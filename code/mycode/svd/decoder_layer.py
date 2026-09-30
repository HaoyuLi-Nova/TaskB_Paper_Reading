"""SVD 解码器每一层的结构，以及 ControlNet 的输出加在哪。

ControlNet 不在解码器中间插入新层。解码器仍是原来的 12 个 ResBlock。
ControlNet 的 block 是编码器块的副本：输入通道等于它自己的特征通道，
后面接一个 1×1 零卷积。零卷积的输出与跳跃连接逐元素相加。

解码器每一层的第一行是沿通道拼接，不是把控制残差再拼成第三份：

    skip = skip + control          # 同形状，逐元素相加
    x = cat([h, skip], dim=通道)   # 解码器 ResBlock 预训练时的输入宽度
    x = 空间 ResBlock(x)
    x = 时间 ResBlock(x)
    x = 空间注意力 + 时间注意力    # 最深的 3 层没有注意力

官方 ControlNet 把这两步写成一句：cat([h, hs.pop() + control.pop()], dim=1)。
diffusers 的 SVD U-Net 先把 down_block_additional_residuals 加到全部 skip 上，
再在 CrossAttnUpBlockSpatioTemporal 里做 cat。

下面 DECODER_LAYERS 是真实 SVD（block_out_channels=320,640,1280,1280，
layers_per_block=2，所以上采样块各有 2+1=3 个 ResBlock）的 12 层。
可运行的类只演示第 7 层：真实 cat(1280, 640)=1920 → 640，这里通道除以 40。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 真实第 7 层：隐藏 1280，跳连 640，输出 640。教学通道 /40。
C_H, C_SKIP, C_OUT = 32, 16, 16
TEMB = 32
HEADS = 4
CLIP_DIM = 32  # 真实 SVD 的 CLIP 图像嵌入是 [B, 1, 1024]
B, FRAMES, H, W = 1, 4, 4, 4
GROUPS = 8

# 从深到浅。skip 通道是这一层 cat 进来的编码器特征。
# attn=False 的三层属于最深的 UpBlockSpatioTemporal，与最深的 DownBlock 对应。
# up 标记这一层跑完后、整个 up block 才做的 ×2 上采样，不在 ResBlock 内部。
DECODER_LAYERS = [
    # idx, 空间, h 通道, skip 通道, 拼接后, 输出, 注意力, 层后上采样
    (1, "H/8", 1280, 1280, 2560, 1280, False, False),
    (2, "H/8", 1280, 1280, 2560, 1280, False, False),
    (3, "H/8", 1280, 1280, 2560, 1280, False, True),
    (4, "H/4", 1280, 1280, 2560, 1280, True, False),
    (5, "H/4", 1280, 1280, 2560, 1280, True, False),
    (6, "H/4", 1280, 640, 1920, 1280, True, True),
    (7, "H/2", 1280, 640, 1920, 640, True, False),
    (8, "H/2", 640, 640, 1280, 640, True, False),
    (9, "H/2", 640, 320, 960, 640, True, True),
    (10, "H", 640, 320, 960, 320, True, False),
    (11, "H", 320, 320, 640, 320, True, False),
    (12, "H", 320, 320, 640, 320, True, False),
]


def print_decoder_layers():
    print("原 U-Net 解码器 12 层（中间块的输出是第 1 层的 h，不在这张表里）")
    print("层  空间   h      skip    cat 后   输出   注意力  该 up 块结束后上采样")
    for idx, res, h_ch, skip_ch, cat_ch, out_ch, attn, up in DECODER_LAYERS:
        print(
            f"{idx:2d}  {res:4s}  {h_ch:4d}  {skip_ch:4d}  {cat_ch:4d}  {out_ch:4d}"
            f"   {'有' if attn else '无'}      {'是' if up else '否'}"
        )


class ZeroConv(nn.Module):
    """ControlNet 每个编码器输出上的 1×1 卷积，权重与偏置初始为 0。

    x [B, F, C, H, W] → [B, F, C, H, W]。C 与跳跃连接通道相同，空间不变。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)
        # [B, F, C, H, W] → [B*F, C, H, W]
        # B=batch, F=帧, C=与跳连相同的通道, H/W=该层空间
        y = self.conv(x)
        # [B*F, C, H, W] 形状不变。初始为 0
        y = y.reshape(b, f, c, h, w)
        # → [B, F, C, H, W]
        return y


class SpatialResBlock(nn.Module):
    """ResnetBlock2D。解码器这一层的 in_channels = C_h + C_skip；ControlNet 块的 in_channels = C_skip。

    x [B, F, C_in, H, W], temb [B, TEMB] → [B, F, C_out, H, W]
    """

    def __init__(self, cin, cout):
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, cin)
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(TEMB, cout)
        self.norm2 = nn.GroupNorm(GROUPS, cout)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(cin, cout, kernel_size=1) if cin != cout else None

    def forward(self, x, temb):
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)
        # [B, F, C_in, H, W] → [B*F, C_in, H, W]
        # B=batch, F=帧, C_in=该 ResBlock 的输入通道, H/W=该层空间
        residual = x
        y = self.norm1(x)
        # [B*F, C_in, H, W] 形状不变
        y = F.silu(y)
        # 形状不变
        y = self.conv1(y)
        # [B*F, C_in, H, W] → [B*F, C_out, H, W]
        t = F.silu(temb)
        # [B, TEMB] 形状不变。TEMB=时间嵌入维
        t = self.time_proj(t)
        # [B, TEMB] → [B, C_out]
        t = t.repeat_interleave(f, dim=0)
        # [B, C_out] → [B*F, C_out]  同一时间步复制到每一帧
        t = t[:, :, None, None]
        # → [B*F, C_out, 1, 1]  后两维留给空间广播
        y = y + t
        # [B*F, C_out, H, W]  时间嵌入加到每个空间位置
        y = self.norm2(y)
        y = F.silu(y)
        y = self.conv2(y)
        # [B*F, C_out, H, W] 形状不变
        if self.skip is not None:
            residual = self.skip(residual)
            # [B*F, C_in, H, W] → [B*F, C_out, H, W]  通道不同时用 1×1 对齐残差
        y = y + residual
        # [B*F, C_out, H, W]
        y = y.reshape(b, f, -1, h, w)
        # → [B, F, C_out, H, W]
        return y


class TemporalResBlock(nn.Module):
    """TemporalResnetBlock。Conv3d 核 (3,1,1)，只在帧轴上混合，空间 1×1。

    x [B, F, C, H, W], temb [B, TEMB] → [B, F, C, H, W]。通道与空间都不变。
    """

    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.time_proj = nn.Linear(TEMB, channels)
        self.norm2 = nn.GroupNorm(GROUPS, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x, temb):
        b, f, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        # [B, F, C, H, W] → [B, C, F, H, W]
        # Conv3d 把通道放在 dim1，帧放在 dim2
        residual = x
        y = self.norm1(x)
        # [B, C, F, H, W] 形状不变
        y = F.silu(y)
        y = self.conv1(y)
        # [B, C, F, H, W] 形状不变；核 (3,1,1) 混合相邻帧
        t = F.silu(temb)
        # [B, TEMB]
        t = self.time_proj(t)
        # [B, TEMB] → [B, C]
        t = t[:, :, None, None, None]
        # → [B, C, 1, 1, 1]  广播到帧和空间
        y = y + t
        # [B, C, F, H, W]
        y = self.norm2(y)
        y = F.silu(y)
        y = self.conv2(y)
        # [B, C, F, H, W] 形状不变
        y = y + residual
        y = y.permute(0, 2, 1, 3, 4)
        # [B, C, F, H, W] → [B, F, C, H, W]
        return y


class SpatioTemporalResBlock(nn.Module):
    """SpatioTemporalResBlock：空间 ResBlock，再时间 ResBlock，再与纯空间结果混合。

    真实实现用 AlphaBlender。image_only 的样本会把混合系数收到 0，时间支路不生效。
    """

    def __init__(self, cin, cout):
        super().__init__()
        self.spatial = SpatialResBlock(cin, cout)
        self.temporal = TemporalResBlock(cout)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, temb):
        spatial = self.spatial(x, temb)
        # [B, F, C_in, H, W] → [B, F, C_out, H, W]
        temporal = self.temporal(spatial, temb)
        # [B, F, C_out, H, W] 形状不变
        mix = torch.sigmoid(self.alpha)
        # 标量。真实 AlphaBlender 还会看 image_only_indicator
        y = spatial + mix * (temporal - spatial)
        # [B, F, C_out, H, W]  空间结果与时间结果按 alpha 混合
        return y


def sdpa(q, k, v):
    """q,k,v: [B, n, S, d] → [B, S, n, d]。n=头, S=序列, d=头维。"""
    y = F.scaled_dot_product_attention(q, k, v)
    # [B, n, S, d] 形状不变
    y = y.transpose(1, 2)
    # [B, n, S, d] → [B, S, n, d]
    return y


class SpatioTemporalTransformer(nn.Module):
    """TransformerSpatioTemporalModel 的一层。

    先在每一帧的空间 token 上做自注意力和与 CLIP 图像 token 的交叉注意力，
    再把同一像素的各帧当成序列做时间自注意力。通道不变。
    x [B, F, C, H, W], clip [B, 1, CLIP_DIM] → [B, F, C, H, W]
    """

    def __init__(self, channels):
        super().__init__()
        self.heads = HEADS
        self.head_dim = channels // HEADS
        self.norm_s = nn.LayerNorm(channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.norm_c = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.clip_kv = nn.Linear(CLIP_DIM, channels * 2)
        self.norm_t = nn.LayerNorm(channels)
        self.temp_qkv = nn.Linear(channels, channels * 3)
        self.out = nn.Linear(channels, channels)

    def forward(self, x, clip):
        b, f, c, h, w = x.shape
        residual = x
        tokens = x.permute(0, 1, 3, 4, 2)
        # [B, F, C, H, W] → [B, F, H, W, C]
        tokens = tokens.reshape(b * f, h * w, c)
        # → [B*F, H*W, C]  H*W=这一帧的空间 token 数
        tokens = self.norm_s(tokens)
        qkv = self.to_qkv(tokens)
        # [B*F, H*W, C] → [B*F, H*W, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        # 各 [B*F, H*W, C]
        q = q.reshape(b * f, h * w, self.heads, self.head_dim).transpose(1, 2)
        # → [B*F, n, H*W, d]  n=头, d=头维
        k = k.reshape(b * f, h * w, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b * f, h * w, self.heads, self.head_dim).transpose(1, 2)
        tokens = sdpa(q, k, v)
        # [B*F, n, H*W, d] → [B*F, H*W, n, d]
        tokens = tokens.reshape(b * f, h * w, c)
        # → [B*F, H*W, C]

        clip_f = clip.repeat_interleave(f, dim=0)
        # [B, 1, CLIP_DIM] → [B*F, 1, CLIP_DIM]  同一张条件图复制到每一帧
        tokens = self.norm_c(tokens)
        q = self.to_q(tokens)
        # [B*F, H*W, C]
        kv = self.clip_kv(clip_f)
        # [B*F, 1, CLIP_DIM] → [B*F, 1, 2C]
        k, v = kv.chunk(2, dim=-1)
        # 各 [B*F, 1, C]  1=CLIP 图像 token
        q = q.reshape(b * f, h * w, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b * f, 1, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b * f, 1, self.heads, self.head_dim).transpose(1, 2)
        cross = sdpa(q, k, v)
        # → [B*F, H*W, n, d]
        cross = cross.reshape(b * f, h * w, c)
        # → [B*F, H*W, C]
        tokens = tokens + cross
        # 交叉注意力的残差。空间 token 数不变

        tokens = tokens.reshape(b, f, h * w, c)
        # → [B, F, H*W, C]
        tokens = tokens.permute(0, 2, 1, 3)
        # → [B, H*W, F, C]  同一像素的帧排成时间序列
        tokens = tokens.reshape(b * h * w, f, c)
        # → [B*H*W, F, C]
        tokens = self.norm_t(tokens)
        qkv = self.temp_qkv(tokens)
        # [B*H*W, F, C] → [B*H*W, F, 3C]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(b * h * w, f, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b * h * w, f, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b * h * w, f, self.heads, self.head_dim).transpose(1, 2)
        tokens = sdpa(q, k, v)
        tokens = tokens.reshape(b * h * w, f, c)
        # → [B*H*W, F, C]
        tokens = self.out(tokens)
        tokens = tokens.reshape(b, h * w, f, c)
        # → [B, H*W, F, C]
        tokens = tokens.permute(0, 2, 1, 3)
        # → [B, F, H*W, C]
        tokens = tokens.reshape(b, f, h, w, c)
        # → [B, F, H, W, C]
        y = tokens.permute(0, 1, 4, 2, 3)
        # → [B, F, C, H, W]
        y = y + residual
        # 整个注意力块的残差
        return y


class DecoderLayer(nn.Module):
    """解码器的一层。对应 CrossAttnUpBlockSpatioTemporal 里的一次循环。

    进入本层之前，control 已经加到 skip 上。本层只做 cat，再跑时空块。
    h [B, F, C_H, H, W], skip [B, F, C_SKIP, H, W] → [B, F, C_OUT, H, W]
    """

    def __init__(self):
        super().__init__()
        self.res = SpatioTemporalResBlock(C_H + C_SKIP, C_OUT)
        self.attn = SpatioTemporalTransformer(C_OUT)

    def forward(self, h, skip, temb, clip):
        x = torch.cat([h, skip], dim=2)
        # [B, F, C_H, H, W] 与 [B, F, C_SKIP, H, W] 沿通道拼接
        # → [B, F, C_H+C_SKIP, H, W]
        # 第一层卷积权重是 [C_OUT, C_H+C_SKIP, 3, 3]，宽度在预训练时就定死了
        x = self.res(x, temb)
        # → [B, F, C_OUT, H, W]
        x = self.attn(x, clip)
        # [B, F, C_OUT, H, W] 形状不变
        return x


class ControlBlock(nn.Module):
    """编码器副本的一块，再加零卷积。没有 cat。

    输入通道是控制支路自己的 C_SKIP，不是解码器的 C_H+C_SKIP。
    h [B, F, C_SKIP, H, W] → 残差 [B, F, C_SKIP, H, W]，与跳连同形。
    """

    def __init__(self):
        super().__init__()
        self.res = SpatioTemporalResBlock(C_SKIP, C_SKIP)
        self.attn = SpatioTemporalTransformer(C_SKIP)
        self.zero = ZeroConv(C_SKIP)

    def forward(self, h, temb, clip):
        h = self.res(h, temb)
        # [B, F, C_SKIP, H, W] 形状不变
        h = self.attn(h, clip)
        # 形状不变
        residual = self.zero(h)
        # → [B, F, C_SKIP, H, W]  初始为 0，加到解码器的 skip 上
        return residual


def main():
    print_decoder_layers()
    print()
    print("下面只跑第 7 层。真实 cat(1280, 640)=1920 → 640；这里 /40：cat(32, 16)=48 → 16。")

    torch.manual_seed(0)
    decoder = DecoderLayer()
    control = ControlBlock()
    temb = torch.randn(B, TEMB)
    # [B, TEMB]  扩散时间步嵌入，解码器与 ControlNet 共用
    clip = torch.randn(B, 1, CLIP_DIM)
    # [B, 1, CLIP_DIM]  1=条件图像的一个 CLIP token
    h = torch.randn(B, FRAMES, C_H, H, W)
    # [B, F, 32, H, W]  上一解码层的输出。32 对应真实 1280
    skip = torch.randn(B, FRAMES, C_SKIP, H, W)
    # [B, F, 16, H, W]  编码器留下的跳连。16 对应真实 640
    feat = torch.randn(B, FRAMES, C_SKIP, H, W)
    # [B, F, 16, H, W]  ControlNet 编码器在这一级的特征，通道与 skip 相同

    residual = control(feat, temb, clip)
    # → [B, F, 16, H, W]
    print(f"ControlNet 零卷积输出: {tuple(residual.shape)}  最大绝对值 {residual.abs().max().item():.3e}")
    print(f"解码器第一层卷积权重: {tuple(decoder.res.spatial.conv1.weight.shape)}")
    print("  含义 [C_out, C_h+C_skip, 3, 3] = [16, 48, 3, 3]，对应真实 [640, 1920, 3, 3]")
    print(f"ControlNet 块卷积权重: {tuple(control.res.spatial.conv1.weight.shape)}")
    print("  含义 [C_skip, C_skip, 3, 3] = [16, 16, 3, 3]，对应真实 [640, 640, 3, 3]")

    skip_in = skip + residual
    # [B, F, 16, H, W]  逐元素相加，通道数仍是 16
    with torch.no_grad():
        y0 = decoder(h, skip_in, temb, clip)
        y_ref = decoder(h, skip, temb, clip)
    # 两者都是 [B, F, 16, H, W]
    print(f"加进解码器之后: {tuple(y0.shape)}")
    print(f"零残差时与原 skip 的最大差: {(y0 - y_ref).abs().max().item():.3e}")

    with torch.no_grad():
        mid = torch.randn(B, FRAMES, C_H, H, W)
        # [B, F, 32, H, W]  中间块输出，是解码器的起始 h
        mid_residual = torch.zeros_like(mid)
        # ControlNet 中间块的零卷积，与 mid 同形，加在 h 上，不再做 cat
        h0 = mid + mid_residual
        # [B, F, 32, H, W]
    print(f"中间块: h = mid + control_mid → {tuple(h0.shape)}，然后才进入第 1 层的 cat")


if __name__ == "__main__":
    main()
