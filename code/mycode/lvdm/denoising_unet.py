"""
MotionCtrl referred LVDM denoising U-Net, implemented as a runnable skeleton following VideoCrafter1 block ordering.
Pixel video is compressed into latent space by a 3D autoencoder; that VAE is not implemented here.
This module only performs noise prediction ε_θ(z_t, t, c): inputs are noisy latent video, diffusion timestep, text tokens.
Tensor layout matches official implementation: [B, C, F, H, W], channels precede frame dimension.
y_t ∈ R^{H×W×L×C} in MotionCtrl paper is a view after spatial unfolding inside temporal attention.

Block ordering with attention:
    Spatial ResBlock         Single-frame convolution with diffusion timestep embedding injected
    SpatialTransformer       Intra-frame self-attention, followed by cross-attention with text
    TemporalTransformer      Two rounds of self-attention along frame axis for each spatial pixel

Resolution schedule follows VideoCrafter1: channel_mult=[1,2,4,4], num_res_blocks=2.
At each level, stack ResBlocks at identical spatial size before downsampling between levels; the bottom level has no further downsampling.
attention_resolutions=[4,2,1] denotes levels with downsample factor ds∈{1,2,4} equip spatial-temporal attention.
The bottom ds=8 level only contains convolutions. Middle block resides at the lowest resolution and has attention.

For real 256 video, latent space is 32×32 with channels 320/640/1280/1280 and 16 frames; four spatial stages: 32,16,8,4.
This skeleton uses reduced dimensions for teaching: spatial size=16, channels=16/32/64/64, frames=4; four spatial stages:16,8,4,2.
The number of downsampling steps remains identical to official.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Teaching dimensions. Real latent: C=4, frames=16, latent spatial ~32
B, C_LATENT, FRAMES, H, W = 1, 4, 4, 16, 16
MODEL_CH = 16
CHANNEL_MULT = (1, 2, 4, 4)
NUM_RES_BLOCKS = 2
ATTN_DS = (1, 2, 4)
TEMB = MODEL_CH * 4
TEXT_LEN, TEXT_DIM = 8, 32
HEADS = 4
GROUPS = 4


def timestep_embedding(t, dim):
    """Diffusion timestep embedding.
    t [B] → [B, dim]. dim = sinusoidal embedding dimension.
    """
    half = dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    # [half] half=dim/2, each entry corresponds to a frequency
    arg = t.float()[:, None] * freq[None, :]
    # [B] × [half] → [B, half] B=batch, half=half embedding dimension
    emb = torch.cat([arg.sin(), arg.cos()], dim=-1)
    # [B, half] → [B, dim] B=batch, dim=full embedding dimension
    return emb


class SelfAttention(nn.Module):
    """Sequence self-attention.
    x [N, S, C] → [N, S, C]. S = sequence length.
    """
    def __init__(self, channels):
        super().__init__()
        self.heads = HEADS
        self.head_dim = channels // HEADS
        self.norm = nn.LayerNorm(channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.out = nn.Linear(channels, channels)

    def forward(self, x):
        n, s, c = x.shape
        y = self.norm(x)
        # [N, S, C] → [N, S, C] N=batch, S=sequence length, C=channel
        qkv = self.to_qkv(y)
        # [N, S, C] → [N, S, 3C] N=batch, S=sequence length, 3C=concatenated q/k/v
        qkv = qkv.reshape(n, s, 3, self.heads, self.head_dim)
        # [N, S, 3C] → [N, S, 3, heads, head_dim] N=batch, S=sequence length, 3=q/k/v, heads=attention heads, head_dim=per-head channel
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # [N, S, 3, heads, head_dim] → [3, N, heads, S, head_dim] 3=q/k/v, N=batch, heads=attention heads, S=sequence length, head_dim=per-head channel
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q [N, heads, S, head_dim]; k [N, heads, S, head_dim]; v [N, heads, S, head_dim] N=batch, heads=attention heads, S=sequence length, head_dim=per-head channel
        y = F.scaled_dot_product_attention(q, k, v)
        # [N, heads, S, head_dim] → [N, heads, S, head_dim] N=batch, heads=attention heads, S=sequence length, head_dim=per-head channel
        y = y.transpose(1, 2)
        # [N, heads, S, head_dim] → [N, S, heads, head_dim] N=batch, S=sequence length, heads=attention heads, head_dim=per-head channel
        y = y.reshape(n, s, c)
        # [N, S, heads, head_dim] → [N, S, C] N=batch, S=sequence length, C=original channel
        y = self.out(y)
        # [N, S, C] → [N, S, C] N=batch, S=sequence length, C=channel
        return x + y


class CrossAttention(nn.Module):
    """Cross attention between query sequence and external condition context.
    x [N, S, C], context [N, S_ctx, C_ctx] → [N, S, C].
    Queries from x; keys and values from context.
    """
    def __init__(self, channels, context_dim):
        super().__init__()
        self.heads = HEADS
        self.head_dim = channels // HEADS
        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(context_dim, channels * 2)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, context):
        n, s, c = x.shape
        s_ctx = context.shape[1]
        y = self.norm(x)
        # [N, S, C] → [N, S, C] N=batch, S=query seq len, C=query channel
        q = self.to_q(y)
        # [N, S, C] → [N, S, C] N=batch, S=query seq len, C=query channel
        q = q.reshape(n, s, self.heads, self.head_dim)
        # [N, S, C] → [N, S, heads, head_dim] N=batch, S=query seq len, heads=attention heads, head_dim=per-head channel
        q = q.permute(0, 2, 1, 3)
        # [N, S, heads, head_dim] → [N, heads, S, head_dim] N=batch, heads=attention heads, S=query seq len, head_dim=per-head channel
        kv = self.to_kv(context)
        # [N, S_ctx, C_ctx] → [N, S_ctx, 2C] N=batch, S_ctx=context seq len, 2C=concatenated k/v
        kv = kv.reshape(n, s_ctx, 2, self.heads, self.head_dim)
        # [N, S_ctx, 2C] → [N, S_ctx, 2, heads, head_dim] N=batch, S_ctx=context seq len, 2=k/v, heads=attention heads, head_dim=per-head channel
        kv = kv.permute(2, 0, 3, 1, 4)
        # [N, S_ctx, 2, heads, head_dim] → [2, N, heads, S_ctx, head_dim] 2=k/v, N=batch, heads=attention heads, S_ctx=context seq len, head_dim=per-head channel
        k, v = kv[0], kv[1]
        # k [N, heads, S_ctx, head_dim]; v [N, heads, S_ctx, head_dim] N=batch, heads=attention heads, S_ctx=context seq len, head_dim=per-head channel
        y = F.scaled_dot_product_attention(q, k, v)
        # [N, heads, S, head_dim] → [N, heads, S, head_dim] N=batch, heads=attention heads, S=query seq len, head_dim=per-head channel
        y = y.transpose(1, 2)
        # [N, heads, S, head_dim] → [N, S, heads, head_dim] N=batch, S=query seq len, heads=attention heads, head_dim=per-head channel
        y = y.reshape(n, s, c)
        # [N, S, heads, head_dim] → [N, S, C] N=batch, S=query seq len, C=query channel
        y = self.out(y)
        # [N, S, C] → [N, S, C] N=batch, S=query seq len, C=query channel
        return x + y


class ResBlock(nn.Module):
    """Single-frame spatial convolution, inject diffusion timestep embedding onto every spatial location.
    x [B, C_in, F, H, W], temb [B, TEMB] → [B, C_out, F, H, W].
    Frame axis is not mixed. MotionCtrl object trajectory is added at input of such conv blocks, only in encoder.
    """
    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, cin)
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(temb_dim, cout)
        self.norm2 = nn.GroupNorm(GROUPS, cout)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(cin, cout, kernel_size=1) if cin != cout else None

    def forward(self, x, temb):
        b, c, f, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4)
        # [B, C_in, F, H, W] → [B, F, C_in, H, W] B=batch, F=frame, C_in=input channel, H=height, W=width
        y = y.reshape(b * f, c, h, w)
        # [B, F, C_in, H, W] → [B*F, C_in, H, W] B*F=flatten batch&frame, C_in=input channel, H=height, W=width; convolution operates per single frame
        residual = y
        y = self.norm1(y)
        # [B*F, C_in, H, W] → [B*F, C_in, H, W] B*F=flatten batch&frame, C_in=input channel, H=height, W=width
        y = F.silu(y)
        y = self.conv1(y)
        # [B*F, C_in, H, W] → [B*F, C_out, H, W] B*F=flatten batch&frame, C_in=input channel, C_out=output channel, H=height, W=width
        t = self.time_proj(temb)
        # [B, TEMB] → [B, C_out] B=batch, TEMB=timestep embedding dim, C_out=output channel
        t = t[:, :, None, None]
        # [B, C_out] → [B, C_out, 1, 1] B=batch, C_out=output channel, 1=broadcast spatial dims
        t = t.repeat_interleave(f, dim=0)
        # [B, C_out, 1, 1] → [B*F, C_out, 1, 1] B*F=flatten batch&frame, replicate same diffusion step embedding to every frame
        y = y + t
        # [B*F, C_out, H, W] → [B*F, C_out, H, W] B*F=flatten batch&frame, C_out=output channel, H=height, W=width; add time embedding to each spatial position
        y = self.norm2(y)
        y = F.silu(y)
        y = self.conv2(y)
        # [B*F, C_out, H, W] → [B*F, C_out, H, W] B*F=flatten batch&frame, C_out=output channel, H=height, W=width
        if self.skip is not None:
            residual = self.skip(residual)
            # [B*F, C_in, H, W] → [B*F, C_out, H, W] B*F=flatten batch&frame, C_in=input channel, C_out=output channel, H=height, W=width
        y = y + residual
        # [B*F, C_out, H, W] → [B*F, C_out, H, W] B*F=flatten batch&frame, C_out=output channel, H=height, W=width
        y = y.reshape(b, f, -1, h, w)
        # [B*F, C_out, H, W] → [B, F, C_out, H, W] B=batch, F=frame, C_out=output channel, H=height, W=width
        y = y.permute(0, 2, 1, 3, 4)
        # [B, F, C_out, H, W] → [B, C_out, F, H, W] B=batch, C_out=output channel, F=frame, H=height, W=width
        return y


class SpatialTransformer(nn.Module):
    """Intra-frame attention: pixels attend to each other then attend to text. Channel & spatial size unchanged.
    x [B, C, F, H, W], text [B, N_text, C_text] → [B, C, F, H, W].
    """
    def __init__(self, channels):
        super().__init__()
        self.self_attn = SelfAttention(channels)
        self.cross_attn = CrossAttention(channels, TEXT_DIM)
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels))

    def forward(self, x, text):
        b, c, f, h, w = x.shape
        tokens = x.permute(0, 2, 3, 4, 1)
        # [B, C, F, H, W] → [B, F, H, W, C] B=batch, C=channel, F=frame, H=height, W=width
        tokens = tokens.reshape(b * f, h * w, c)
        # [B, F, H, W, C] → [B*F, H*W, C] B*F=flatten batch&frame, H*W=spatial tokens per frame, C=channel
        tokens = self.self_attn(tokens)
        # [B*F, H*W, C] → [B*F, H*W, C] B*F=flatten batch&frame, H*W=spatial tokens per frame, C=channel; self-attention among pixels inside single frame
        ctx = text.repeat_interleave(f, dim=0)
        # [B, N_text, C_text] → [B*F, N_text, C_text] B*F=flatten batch&frame, N_text=text token count, C_text=text channel; same text condition shared by all frames
        tokens = self.cross_attn(tokens, ctx)
        # [B*F, H*W, C] → [B*F, H*W, C] B*F=flatten batch&frame, H*W=spatial tokens per frame, C=channel; query=pixels, key/value=text
        residual = tokens
        tokens = self.ff_norm(tokens)
        tokens = self.ff(tokens)
        # [B*F, H*W, C] → [B*F, H*W, C] B*F=flatten batch&frame, H*W=spatial tokens per frame, C=channel
        tokens = tokens + residual
        tokens = tokens.reshape(b, f, h, w, c)
        # [B*F, H*W, C] → [B, F, H, W, C] B=batch, F=frame, H=height, W=width, C=channel
        tokens = tokens.permute(0, 4, 1, 2, 3)
        # [B, F, H, W, C] → [B, C, F, H, W] B=batch, C=channel, F=frame, H=height, W=width
        return tokens


class TemporalTransformer(nn.Module):
    """Two self-attention passes along frame axis for each fixed spatial pixel. Channel and frame count unchanged.
    x [B, C, F, H, W] → [B, C, F, H, W].
    In VideoCrafter1 only_self_att mode, one BasicTransformerBlock contains exactly these two attention layers.
    MotionCtrl camera pose is injected after first attention and before second attention.
    """
    def __init__(self, channels):
        super().__init__()
        self.attn1 = SelfAttention(channels)
        self.attn2 = SelfAttention(channels)
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels))

    def forward(self, x):
        b, c, f, h, w = x.shape
        tokens = x.permute(0, 3, 4, 2, 1)
        # [B, C, F, H, W] → [B, H, W, F, C] B=batch, C=channel, F=frame, H=height, W=width
        # This matches paper view H×W×L×C where L=F=frame count
        tokens = tokens.reshape(b * h * w, f, c)
        # [B, H, W, F, C] → [B*H*W, F, C] B*H*W=flatten batch&spatial locations, F=frame, C=channel; each spatial position forms one time sequence of length F
        tokens = self.attn1(tokens)
        # [B*H*W, F, C] → [B*H*W, F, C] B*H*W=flatten batch&spatial locations, F=frame, C=channel; first temporal self-attention
        tokens = self.attn2(tokens)
        # [B*H*W, F, C] → [B*H*W, F, C] B*H*W=flatten batch&spatial locations, F=frame, C=channel; second temporal self-attention
        residual = tokens
        tokens = self.ff_norm(tokens)
        tokens = self.ff(tokens)
        # [B*H*W, F, C] → [B*H*W, F, C] B*H*W=flatten batch&spatial locations, F=frame, C=channel
        tokens = tokens + residual
        tokens = tokens.reshape(b, h, w, f, c)
        # [B*H*W, F, C] → [B, H, W, F, C] B=batch, H=height, W=width, F=frame, C=channel
        tokens = tokens.permute(0, 4, 3, 1, 2)
        # [B, H, W, F, C] → [B, C, F, H, W] B=batch, C=channel, F=frame, H=height, W=width
        return tokens


class UNetUnit(nn.Module):
    """Single cell inside official input/output blocks.
    Normal cell: ResBlock; if ds∈{1,2,4}, attach spatial transformer then temporal transformer. Spatial size unchanged.
    Downsample cell: stride-2 conv only, spatial size halved. Its output is also a skip connection.
    Upsample cell: run ResBlock (and attention) first, then spatially upsample by factor 2.
    x [B, C_in, F, H, W], temb [B, TEMB], text [B, N_text, C_text]
    → [B, C_out, F, H', W']. H'=H/2 for downsample, H'=2H for upsample, H'=H otherwise.
    """
    def __init__(self, cin, cout, temb_dim, attn=False, down=False, up=False):
        super().__init__()
        self.res = None if down else ResBlock(cin, cout, temb_dim)
        self.spatial = SpatialTransformer(cout) if attn else None
        self.temporal = TemporalTransformer(cout) if attn else None
        self.down = Downsample(cout) if down else None
        self.up = Upsample(cout) if up else None

    def forward(self, x, temb, text):
        if self.res is not None:
            x = self.res(x, temb)
            # [B, C_in, F, H, W] → [B, C_out, F, H, W] B=batch, C_in=input channel, C_out=output channel, F=frame, H=height, W=width
        if self.spatial is not None:
            x = self.spatial(x, text)
            # [B, C_out, F, H, W] → [B, C_out, F, H, W] B=batch, C_out=output channel, F=frame, H=height, W=width
            x = self.temporal(x)
            # [B, C_out, F, H, W] → [B, C_out, F, H, W] B=batch, C_out=output channel, F=frame, H=height, W=width
        if self.down is not None:
            x = self.down(x)
            # [B, C, F, H, W] → [B, C, F, H/2, W/2] B=batch, C=channel, F=frame, H=height, W=width
        if self.up is not None:
            x = self.up(x)
            # [B, C, F, H, W] → [B, C, F, 2H, 2W] B=batch, C=channel, F=frame, H=height, W=width
        return x


class MiddleBlock(nn.Module):
    """Middle block at lowest resolution: conv, spatial-temporal attention, conv. Spatial size unchanged.
    x [B, C, F, H, W], temb [B, TEMB], text [B, N_text, C_text] → same shape.
    """
    def __init__(self, channels, temb_dim):
        super().__init__()
        self.res1 = ResBlock(channels, channels, temb_dim)
        self.spatial = SpatialTransformer(channels)
        self.temporal = TemporalTransformer(channels)
        self.res2 = ResBlock(channels, channels, temb_dim)

    def forward(self, x, temb, text):
        x = self.res1(x, temb)
        # [B, C, F, H, W] → [B, C, F, H, W] B=batch, C=channel, F=frame, H=height, W=width
        x = self.spatial(x, text)
        # [B, C, F, H, W] → [B, C, F, H, W] B=batch, C=channel, F=frame, H=height, W=width
        x = self.temporal(x)
        # [B, C, F, H, W] → [B, C, F, H, W] B=batch, C=channel, F=frame, H=height, W=width
        x = self.res2(x, temb)
        # [B, C, F, H, W] → [B, C, F, H, W] B=batch, C=channel, F=frame, H=height, W=width
        return x


class Downsample(nn.Module):
    """Spatially halve resolution only; frame count unchanged.
    x [B, C, F, H, W] → [B, C, F, H/2, W/2].
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4)
        # [B, C, F, H, W] → [B, F, C, H, W] B=batch, C=channel, F=frame, H=height, W=width
        y = y.reshape(b * f, c, h, w)
        # [B, F, C, H, W] → [B*F, C, H, W] B*F=flatten batch&frame, C=channel, H=height, W=width
        y = self.conv(y)
        # [B*F, C, H, W] → [B*F, C, H/2, W/2] B*F=flatten batch&frame, C=channel, H=height, W=width
        y = y.reshape(b, f, c, h // 2, w // 2)
        # [B*F, C, H/2, W/2] → [B, F, C, H/2, W/2] B=batch, F=frame, C=channel, H=height, W=width
        y = y.permute(0, 2, 1, 3, 4)
        # [B, F, C, H/2, W/2] → [B, C, F, H/2, W/2] B=batch, C=channel, F=frame, H=height, W=width
        return y


class Upsample(nn.Module):
    """Spatially double resolution only; frame count unchanged.
    x [B, C, F, H, W] → [B, C, F, 2H, 2W].
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4)
        # [B, C, F, H, W] → [B, F, C, H, W] B=batch, C=channel, F=frame, H=height, W=width
        y = y.reshape(b * f, c, h, w)
        # [B, F, C, H, W] → [B*F, C, H, W] B*F=flatten batch&frame, C=channel, H=height, W=width
        y = F.interpolate(y, scale_factor=2, mode="nearest")
        # [B*F, C, H, W] → [B*F, C, 2H, 2W] B*F=flatten batch&frame, C=channel, H=height, W=width
        y = self.conv(y)
        # [B*F, C, 2H, 2W] → [B*F, C, 2H, 2W] B*F=flatten batch&frame, C=channel, H=height, W=width
        y = y.reshape(b, f, c, h * 2, w * 2)
        # [B*F, C, 2H, 2W] → [B, F, C, 2H, 2W] B=batch, F=frame, C=channel, H=height, W=width
        y = y.permute(0, 2, 1, 3, 4)
        # [B, F, C, 2H, 2W] → [B, C, F, 2H, 2W] B=batch, C=channel, F=frame, H=height, W=width
        return y


class DenoisingUNet(nn.Module):
    """4-level U-Net isomorphic to VideoCrafter1 input_blocks / middle / output_blocks.
    z_t [B, 4, F, H, W], t [B], text [B, N_text, C_text]
    → predicted noise [B, 4, F, H, W], same shape as z_t.
    Output of every encoder unit is saved into skips. Decoder pops skips in reverse order, concatenates along channel dimension,
    then feeds into corresponding decoder unit. The output of downsample unit also counts as a skip; upsampling occurs
    after consuming this coarse-grid skip feature.
    """
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(MODEL_CH, TEMB), nn.SiLU(), nn.Linear(TEMB, TEMB))
        self.conv_in = nn.Conv2d(C_LATENT, MODEL_CH, kernel_size=3, padding=1)
        self.input_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        skip_channels = [MODEL_CH]
        ch = MODEL_CH
        ds = 1
        for level, mult in enumerate(CHANNEL_MULT):
            out_ch = MODEL_CH * mult
            for _ in range(NUM_RES_BLOCKS):
                self.input_blocks.append(UNetUnit(ch, out_ch, TEMB, attn=ds in ATTN_DS))
                ch = out_ch
                skip_channels.append(ch)
            if level != len(CHANNEL_MULT) - 1:
                self.input_blocks.append(UNetUnit(ch, ch, TEMB, down=True))
                skip_channels.append(ch)
                ds *= 2
        self.middle = MiddleBlock(ch, TEMB)
        for level, mult in reversed(list(enumerate(CHANNEL_MULT))):
            out_ch = MODEL_CH * mult
            for i in range(NUM_RES_BLOCKS + 1):
                skip_ch = skip_channels.pop()
                self.output_blocks.append(
                    UNetUnit(
                        ch + skip_ch,
                        out_ch,
                        TEMB,
                        attn=ds in ATTN_DS,
                        up=level != 0 and i == NUM_RES_BLOCKS,
                    )
                )
                ch = out_ch
                if level != 0 and i == NUM_RES_BLOCKS:
                    ds //= 2
        self.conv_out = nn.Conv2d(MODEL_CH, C_LATENT, kernel_size=3, padding=1)

    def _conv2d_frames(self, conv, x):
        b, c, f, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4)
        # [B, C, F, H, W] → [B, F, C, H, W] B=batch, C=channel, F=frame, H=height, W=width
        y = y.reshape(b * f, c, h, w)
        # [B, F, C, H, W] → [B*F, C, H, W] B*F=flatten batch&frame, C=channel, H=height, W=width
        y = conv(y)
        # [B*F, C, H, W] → [B*F, C_out, H, W] B*F=flatten batch&frame, C=input channel, C_out=output channel, H=height, W=width
        c_out = y.shape[1]
        y = y.reshape(b, f, c_out, h, w)
        # [B*F, C_out, H, W] → [B, F, C_out, H, W] B=batch, F=frame, C_out=output channel, H=height, W=width
        y = y.permute(0, 2, 1, 3, 4)
        # [B, F, C_out, H, W] → [B, C_out, F, H, W] B=batch, C_out=output channel, F=frame, H=height, W=width
        return y

    def forward(self, z_t, t, text, trace=None):
        temb = timestep_embedding(t, MODEL_CH)
        # [B] → [B, MODEL_CH] B=batch, MODEL_CH=base model channel
        temb = self.time_mlp(temb)
        # [B, MODEL_CH] → [B, TEMB] B=batch, MODEL_CH=base model channel, TEMB=final timestep embedding dim
        h = self._conv2d_frames(self.conv_in, z_t)
        # [B, 4, F, H, W] → [B, MODEL_CH, F, H, W] B=batch,4=latent channel, F=frame, H=height, W=width, MODEL_CH=base model channel
        skips = [h]
        if trace is not None:
            trace.append(("conv_in", tuple(h.shape)))
        for i, unit in enumerate(self.input_blocks):
            h = unit(h, temb, text)
            skips.append(h)
            if trace is not None:
                kind = "down" if unit.down is not None else "enc"
                trace.append((f"in{i}:{kind}", tuple(h.shape)))
        h = self.middle(h, temb, text)
        # [B, C, F, H, W] → [B, C, F, H, W] B=batch, C=channel, F=frame, H=height, W=width; lowest spatial resolution
        if trace is not None:
            trace.append(("middle", tuple(h.shape)))
        for i, unit in enumerate(self.output_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            # [B, C_dec, F, H, W] + [B, C_skip, F, H, W] → [B, C_dec + C_skip, F, H, W] concatenate encoder skip features along channel dimension at same spatial resolution
            h = unit(h, temb, text)
            if trace is not None:
                kind = "up" if unit.up is not None else "dec"
                trace.append((f"out{i}:{kind}", tuple(h.shape)))
        eps = self._conv2d_frames(self.conv_out, h)
        # [B, MODEL_CH, F, H, W] → [B, 4, F, H, W] B=batch, MODEL_CH=base model channel,4=latent channel, F=frame, H=height, W=width; predicted noise same shape as z_t
        return eps


def main():
    torch.manual_seed(0)
    net = DenoisingUNet()
    z_t = torch.randn(B, C_LATENT, FRAMES, H, W)
    t = torch.tensor([500])
    text = torch.randn(B, TEXT_LEN, TEXT_DIM)
    trace = []
    eps = net(z_t, t, text, trace=trace)
    print("z_t ", tuple(z_t.shape), " B, latent channel, frame, height, width")
    for name, shape in trace:
        print(f"{name:12s} {shape}")
    print("eps ", tuple(eps.shape), " predicted noise with same shape as z_t")
    loss = F.mse_loss(eps, torch.randn_like(eps))
    print("loss", float(loss.detach()))


if __name__ == "__main__":
    main()
