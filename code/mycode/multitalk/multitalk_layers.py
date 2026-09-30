"""Tiny MultiTalk DiT. Tensor ranks/sizes match wan/modules/multitalk_model.py (1 block, small HxW)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

C, T, H, W = 3, 81, 64, 64                           # RGB, 像素帧, 像素高宽
c, vae_scale, pt, ph, pw = 16, 4, 1, 2, 2            # 潜通道, 时间压缩, patch 核 (t,h,w)
t_lat, h, w = 21, 8, 8                               # VAE 时间格, 潜高宽
N_t, N_h, N_w = 21, 4, 4                             # patch 后时间/空间格
L = N_t * N_h * N_w                                  # 336=视频 token 数
DIM, HEADS, HEAD = 5120, 40, 128                     # DiT 宽, 头数, 头维
FFN, FREQ, TEXT_LEN, TEXT_DIM = 13824, 256, 512, 4096
CLIP_LEN, CLIP_DIM = 257, 1280                       # CLIP token(含 CLS), CLIP 宽
WAV_BLOCKS, WAV_DIM, K, MID = 12, 768, 5, 2          # wav2vec 层, 特征, 窗长, 窗中心
CTX, A_OUT, A_HID = 32, 768, 512                     # 音频 context token, 输出/隐维
IN_DIM = c + c + vae_scale                           # 36=噪声潜+条件潜+mask
R1, R2, R_BG = (0, 4), (20, 24), 12                  # 人1/人2/背景的 1D RoPE 区间


def sdpa(q, k, v):
    """缩放点积注意力。q,k,v: [B, S, n, d] → [B, S, n, d]。B=batch, S=序列, n=头, d=头维。"""
    q = q.transpose(1, 2)                            # [B, S, n, d] → [B, n, S, d]
    k = k.transpose(1, 2)                            # [B, S, n, d] → [B, n, S, d]
    v = v.transpose(1, 2)                            # [B, S, n, d] → [B, n, S, d]
    y = F.scaled_dot_product_attention(q, k, v)      # [B, n, S, d] 形状不变
    y = y.transpose(1, 2)                            # [B, n, S, d] → [B, S, n, d]
    return y


def sinusoidal_embedding_1d(dim, position):
    """正弦时间嵌入。position [B] → [B, dim]。B=batch, dim=FREQ。"""
    half = dim // 2
    scale = torch.pow(10000, -torch.arange(half).float().div(half))
    # [half]  half=频率个数
    sinusoid = torch.outer(position.float(), scale)
    # [B] × [half] → [B, half]
    emb = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    # [B, half] 拼 cos/sin → [B, dim]
    return emb


def rope_params(max_seq_len, dim, theta=10000.0):
    """RoPE 频率表。[max_seq_len, dim/2] 复数；max_seq_len=位置, dim/2=复通道。"""
    freqs = torch.outer(
        torch.arange(max_seq_len).float(),
        1.0 / theta ** (torch.arange(0, dim, 2).float() / dim),
    )
    # [max_seq_len] × [dim/2] → [max_seq_len, dim/2]
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply(x, grid_sizes, freqs):
    """3D RoPE。x [B, L, n, d] → [B, L, n, d]。B=batch, L=token(含 pad), n=头, d=头维。
    只旋转前 f*hh*ww 个有效 token。本脚本 B=1 且 L=有效长度，与官方 x[i,:s] 再拼 pad 等价。"""
    n = x.size(2)                                    # 头数
    half = x.size(3) // 2                            # d/2=复通道
    freqs_t, freqs_h, freqs_w = freqs.split(
        [half - 2 * (half // 3), half // 3, half // 3], dim=1
    )
    # freqs [1024, d/2] → 三段 [1024, ·]  时间/高/宽
    f, hh, ww = grid_sizes[0].tolist()               # 有效网格 (N_t, N_h, N_w)
    seq_len = f * hh * ww                            # 有效 token 数
    x_i = x[0, :seq_len].float()
    # [B, L, n, d] 取 batch0 有效段 → [seq_len, n, d]
    x_i = x_i.reshape(seq_len, n, -1, 2)
    # → [seq_len, n, d/2, 2]
    x_i = torch.view_as_complex(x_i)
    # → [seq_len, n, d/2] 复数
    freqs_t = freqs_t[:f].view(f, 1, 1, -1).expand(f, hh, ww, -1)
    # [f, d_t] → [f, hh, ww, d_t]  时间频率铺到网格
    freqs_h = freqs_h[:hh].view(1, hh, 1, -1).expand(f, hh, ww, -1)
    # → [f, hh, ww, d_h]
    freqs_w = freqs_w[:ww].view(1, 1, ww, -1).expand(f, hh, ww, -1)
    # → [f, hh, ww, d_w]
    freqs_i = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)
    # → [f, hh, ww, d/2]
    freqs_i = freqs_i.reshape(seq_len, 1, -1)
    # → [seq_len, 1, d/2]  1 广播到头维
    x_i = x_i * freqs_i
    # [seq_len, n, d/2] 逐位置旋转
    x_i = torch.view_as_real(x_i)
    # → [seq_len, n, d/2, 2]
    x_i = x_i.flatten(2)
    # → [seq_len, n, d]
    x_pad = x[0, seq_len:]
    # [L-seq_len, n, d]  padding token 不旋转
    x_i = torch.cat([x_i, x_pad], dim=0)
    # → [L, n, d]
    return torch.stack([x_i]).type_as(x)
    # → [1, L, n, d]


def rotate_half(x):
    """最后一维成对 (x1,x2)→(-x2,x1)。[..., d] 形状不变。"""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    # [..., d] → [..., d/2, 2]
    x1, x2 = x.unbind(dim=-1)
    # 各 [..., d/2]
    x = torch.stack((-x2, x1), dim=-1)
    # → [..., d/2, 2]
    x = rearrange(x, "... d r -> ... (d r)")
    # → [..., d]
    return x


def rope_1d(x, pos, base=10000.0):
    """1D RoPE。x [B, n, L, d]，pos [L]。B=batch, n=头, L=序列, d=头维。形状不变。"""
    dim = x.shape[-1]                                # d=头维
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # [d/2]  频率
    ang = torch.einsum("..., f -> ... f", pos.float(), freqs)
    # pos [L] × freqs [d/2] → [L, d/2]
    ang = ang.repeat_interleave(2, -1)
    # → [L, d]  每个频率复用到 sin/cos 对
    cos = ang.cos()[None, None]
    # → [1, 1, L, d]  对齐 B、头维
    sin = ang.sin()[None, None]
    # → [1, 1, L, d]
    x_f = x.float()
    x_rot = rotate_half(x_f)
    y = x_f * cos + x_rot * sin
    # [B, n, L, d] 形状不变
    return y.type_as(x)


def get_attn_map_with_target(q, k, shape, ref_target_masks):
    """用首帧空间 token 作 key，按身份掩码汇总注意力。
    q,k: [B, L, n, d]  B=batch, L=视频 token, n=头, d=头维
    shape: (N_t, N_h, N_w)  patch 网格
    ref_target_masks: [3, S]  3=人1/人2/背景, S=N_h*N_w 每帧空间 token
    → [3, L]
    """
    N_t, N_h, N_w = shape
    S = N_h * N_w                                    # 16=每帧空间 token
    scale = HEAD ** -0.5
    k_first = k[:, :S]
    # [B, L, n, d] → [B, S, n, d]  只用首帧空间格当 key
    attn = torch.einsum("blhd,bshd->blhs", q, k_first) * scale
    # q [B,L,n,d] × k [B,S,n,d] → [B, L, n, S]
    attn = attn.softmax(-1)
    # 沿空间维 S 归一化
    maps = []
    for m in ref_target_masks:
        # m [S]  当前身份的空间掩码
        w = attn * m.view(1, 1, 1, S)
        # [B, L, n, S]  掩码外置 0
        w = w.sum(-1)
        # → [B, L, n]  对该身份空间求和
        w = w.div(m.sum())
        w = w.mean(-1)
        # → [B, L]  对头平均
        maps.append(w[0])
        # → [L]  去掉 batch
    return torch.stack(maps, 0)
    # → [3, L]  3=人1/人2/背景, L=视频 token


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # [..., dim] 形状不变；dim=RMS 归一化通道
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms.type_as(x) * self.weight


class SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(DIM, DIM)
        self.k = nn.Linear(DIM, DIM)
        self.v = nn.Linear(DIM, DIM)
        self.o = nn.Linear(DIM, DIM)
        self.norm_q = WanRMSNorm(DIM)
        self.norm_k = WanRMSNorm(DIM)
        d = HEAD
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], 1)
        # 三段复频率沿通道拼 → [1024, d/2]  1024=最大位置, d/2=复通道

    def forward(self, x, grid_sizes, ref_target_masks):
        """x [B, L, DIM] → [B, L, DIM]，amap [3, L]。B=batch, L=token, DIM=通道。"""
        b, s, n, d = x.shape[0], x.shape[1], HEADS, HEAD
        q = self.q(x)                                # [B, L, DIM] → [B, L, DIM]
        q = self.norm_q(q)
        q = q.view(b, s, n, d)
        # → [B, L, 40, 128]  40=头, 128=头维
        k = self.k(x)
        k = self.norm_k(k)
        k = k.view(b, s, n, d)
        # → [B, L, 40, 128]
        v = self.v(x)
        v = v.view(b, s, n, d)
        # → [B, L, 40, 128]
        q = rope_apply(q, grid_sizes, self.freqs)
        # [B, L, 40, 128] 形状不变
        k = rope_apply(k, grid_sizes, self.freqs)
        amap = get_attn_map_with_target(q, k, grid_sizes[0].tolist(), ref_target_masks)
        # q,k [B,L,40,128], masks [3,S] → [3, L]
        y = sdpa(q, k, v)
        # → [B, L, 40, 128]
        y = y.flatten(2)
        # → [B, L, DIM]
        y = self.o(y)
        # → [B, L, DIM]
        return y, amap


class WanI2VCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(DIM, DIM)
        self.k = nn.Linear(DIM, DIM)
        self.v = nn.Linear(DIM, DIM)
        self.k_img = nn.Linear(DIM, DIM)
        self.v_img = nn.Linear(DIM, DIM)
        self.o = nn.Linear(DIM, DIM)
        self.norm_q = WanRMSNorm(DIM)
        self.norm_k = WanRMSNorm(DIM)
        self.norm_k_img = WanRMSNorm(DIM)

    def forward(self, x, context):
        """x [B, L, DIM]，context [B, 257+512, DIM] → [B, L, DIM]。
        257=CLIP token, 512=T5 token, L=视频 token。"""
        context_img = context[:, :CLIP_LEN]
        # [B, 769, DIM] → [B, 257, DIM]  CLIP 段
        context_txt = context[:, CLIP_LEN:]
        # → [B, 512, DIM]  T5 段
        b, n, d = x.size(0), HEADS, HEAD
        q = self.q(x)
        q = self.norm_q(q)
        q = q.view(b, -1, n, d)
        # [B, L, DIM] → [B, L, 40, 128]
        k = self.k(context_txt)
        k = self.norm_k(k)
        k = k.view(b, -1, n, d)
        # [B, 512, DIM] → [B, 512, 40, 128]
        v = self.v(context_txt)
        v = v.view(b, -1, n, d)
        # → [B, 512, 40, 128]
        k_img = self.k_img(context_img)
        k_img = self.norm_k_img(k_img)
        k_img = k_img.view(b, -1, n, d)
        # [B, 257, DIM] → [B, 257, 40, 128]
        v_img = self.v_img(context_img)
        v_img = v_img.view(b, -1, n, d)
        # → [B, 257, 40, 128]
        y_txt = sdpa(q, k, v)
        # → [B, L, 40, 128]
        y_img = sdpa(q, k_img, v_img)
        # → [B, L, 40, 128]
        y = y_txt + y_img
        y = y.flatten(2)
        # → [B, L, DIM]
        return self.o(y)


class AudioCrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(DIM, DIM, bias=True)
        self.kv = nn.Linear(A_OUT, DIM * 2, bias=True)
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x, encoder_hidden_states, shape, x_ref_attn_map):
        """视频 token 对音频 token 做交叉注意力。
        x: [B, L, DIM]  L=N_t*S
        encoder_hidden_states: [B, N_t, 64, 768]  64=两人×32 context
        shape: (N_t, N_h, N_w)
        x_ref_attn_map: [3, L]  3=人1/人2/背景
        → [B, L, DIM]
        """
        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        # [1, N_t, 64, 768] → [N_t, 64, 768]  本脚本 B=1，去掉 batch
        N_t, N_h, N_w = shape
        S = N_h * N_w                                # 16=每帧空间 token
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)
        # [1, 336, 5120] → [21, 16, 5120]  21=B*N_t 按帧展开, 16=S
        Bf, Ns, _ = x.shape                          # Bf=按帧 batch, Ns=空间 token
        q = self.q(x)
        # [21, 16, 5120] → [21, 16, 5120]
        q = q.view(Bf, Ns, HEADS, HEAD)
        # → [21, 16, 40, 128]
        q = q.permute(0, 2, 1, 3)
        # → [21, 40, 16, 128]  40=头

        h1_max = x_ref_attn_map[0].max()
        h1_min = x_ref_attn_map[0].min()
        h2_max = x_ref_attn_map[1].max()
        h2_min = x_ref_attn_map[1].min()
        # 官方把 max/min 扩成 [3,1,1] 再 cat 取值；与沿 L 直接 max/min 等价
        human1 = (x_ref_attn_map[0] - h1_min) / (h1_max - h1_min) * (R1[1] - R1[0]) + R1[0]
        # [L] 人1 注意力 → RoPE 位置落在 [0, 4]
        human2 = (x_ref_attn_map[1] - h2_min) / (h2_max - h2_min) * (R2[1] - R2[0]) + R2[0]
        # [L] 人2 → [20, 24]
        back = torch.full((x_ref_attn_map.size(1),), float(R_BG))
        # [L] 背景固定 12
        normalized_map = torch.stack([human1, human2, back], dim=1)
        # [L]×3 → [L, 3]
        person_id = x_ref_attn_map.argmax(0)
        # [3, L] 沿身份维 → [L]  每个 token 归属
        pos_q = normalized_map[range(L), person_id]
        # [L, 3] 按 dim1=身份取值 → [L]  该 token 的 1D RoPE 位置

        q = rearrange(q, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        # [21, 40, 16, 128] → [1, 40, 336, 128]  拼回全序列再加 RoPE
        q = rope_1d(q, pos_q)
        # 形状不变；pos_q [336]
        q = rearrange(q, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)
        # → [21, 40, 16, 128]

        Na = encoder_hidden_states.shape[1]
        # 64=两人音频 token
        kv = self.kv(encoder_hidden_states)
        # [21, 64, 768] → [21, 64, 10240]  10240=2*DIM
        kv = kv.view(Bf, Na, 2, HEADS, HEAD)
        # → [21, 64, 2, 40, 128]  2=K/V
        kv = kv.permute(2, 0, 3, 1, 4)
        # → [2, 21, 40, 64, 128]
        k, v = kv.unbind(0)
        # 各 [21, 40, 64, 128]

        per_frame = torch.zeros(Na)
        # [64]
        per_frame[: Na // 2] = (R1[0] + R1[1]) / 2
        # 前 32 个 token → 2（人1 区间中点）
        per_frame[Na // 2:] = (R2[0] + R2[1]) / 2
        # 后 32 个 → 22（人2 中点）
        pos_k = torch.cat([per_frame] * N_t, dim=0)
        # [64] → [1344]  每帧音频位置复制到 N_t 个时间格
        k = rearrange(k, "(B N_t) H S C -> B H (N_t S) C", N_t=N_t)
        # [21, 40, 64, 128] → [1, 40, 1344, 128]
        k = rope_1d(k, pos_k)
        k = rearrange(k, "B H (N_t S) C -> (B N_t) H S C", N_t=N_t)
        # → [21, 40, 64, 128]

        q = rearrange(q, "B H M K -> B M H K")
        # [21, 40, 16, 128] → [21, 16, 40, 128]  sdpa 约定 [B, S, n, d]
        k = rearrange(k, "B H M K -> B M H K")
        # [21, 40, 64, 128] → [21, 64, 40, 128]
        v = rearrange(v, "B H M K -> B M H K")
        y = sdpa(q, k, v)
        # → [21, 16, 40, 128]
        y = y.flatten(2)
        # → [21, 16, 5120]
        y = self.proj(y)
        # → [21, 16, 5120]
        y = rearrange(y, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
        # → [1, 336, 5120]
        return y


class WanAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttn()
        self.norm3 = nn.LayerNorm(DIM, eps=1e-6)
        self.cross_attn = WanI2VCrossAttention()
        self.norm2 = nn.LayerNorm(DIM, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(DIM, FFN), nn.GELU(approximate="tanh"), nn.Linear(FFN, DIM))
        self.modulation = nn.Parameter(torch.randn(1, 6, DIM) / DIM ** 0.5)
        self.audio_cross_attn = AudioCrossAttn()
        self.norm_x = nn.LayerNorm(DIM, eps=1e-6)

    def forward(self, x, e, grid_sizes, context, audio_embedding, ref_target_masks):
        """x [B, L, DIM]，e [B, 6, DIM] AdaLN 调制，→ [B, L, DIM]。"""
        e = self.modulation + e
        # [1, 6, DIM] + [B, 6, DIM] → [B, 6, DIM]
        e0, e1, e2, e3, e4, e5 = e.chunk(6, dim=1)
        # 各 [B, 1, DIM]
        h = self.norm1(x)
        h = h * (1 + e1) + e0
        y, x_ref_attn_map = self.self_attn(h, grid_sizes, ref_target_masks)
        # y [B, L, DIM], x_ref_attn_map [3, L]
        x = x + y * e2
        h = self.norm3(x)
        h = self.cross_attn(h, context)
        # context [B, 769, DIM] → h [B, L, DIM]
        x = x + h
        h = self.norm_x(x)
        h = self.audio_cross_attn(h, audio_embedding, grid_sizes[0].tolist(), x_ref_attn_map)
        # audio_embedding [1, N_t, 64, 768] → h [B, L, DIM]
        x = x + h
        h = self.norm2(x)
        h = h * (1 + e4) + e3
        h = self.ffn(h)
        # [B, L, DIM] → [B, L, DIM]
        x = x + h * e5
        return x


class AudioProjModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(K * WAV_BLOCKS * WAV_DIM, A_HID)
        self.proj1_vf = nn.Linear((K + vae_scale - 1) * WAV_BLOCKS * WAV_DIM, A_HID)
        self.proj2 = nn.Linear(A_HID, A_HID)
        self.proj3 = nn.Linear(A_HID, CTX * A_OUT)
        self.norm = nn.LayerNorm(A_OUT)

    def forward(self, audio_embeds, audio_embeds_vf):
        """音频窗投影为每帧 context token。
        audio_embeds: [H, 1, 5, 12, 768]  H=说话人, 1=首帧, 5=窗 k, 12=层, 768=特征
        audio_embeds_vf: [H, 20, 8, 12, 768]  20=后续 VAE 格, 8=打包窗数
        → [H, 21, 32, 768]  21=全时长, 32=context token, 768=A_OUT
        """
        H = audio_embeds.shape[0]
        n_first = audio_embeds.shape[1]              # 1=首帧
        n_rest = audio_embeds_vf.shape[1]            # 20=后续 VAE 格
        video_length = n_first + n_rest              # 21

        audio_embeds = audio_embeds.reshape(H * n_first, K, WAV_BLOCKS, WAV_DIM)
        # [H, 1, 5, 12, 768] → [H, 5, 12, 768]
        audio_embeds = audio_embeds.flatten(1)
        # → [H, 46080]  46080=5×12×768
        audio_embeds = self.proj1(audio_embeds)
        audio_embeds = torch.relu(audio_embeds)
        # → [H, 512]
        audio_embeds = audio_embeds.view(H, n_first, A_HID)
        # → [H, 1, 512]

        audio_embeds_vf = audio_embeds_vf.reshape(
            H * n_rest, K + vae_scale - 1, WAV_BLOCKS, WAV_DIM
        )
        # [H, 20, 8, 12, 768] → [40, 8, 12, 768]  40=H×20
        audio_embeds_vf = audio_embeds_vf.flatten(1)
        # → [40, 73728]  73728=8×12×768
        audio_embeds_vf = self.proj1_vf(audio_embeds_vf)
        audio_embeds_vf = torch.relu(audio_embeds_vf)
        # → [40, 512]
        audio_embeds_vf = audio_embeds_vf.view(H, n_rest, A_HID)
        # → [H, 20, 512]

        audio_embeds = torch.cat([audio_embeds, audio_embeds_vf], dim=1)
        # [H, 1, 512] + [H, 20, 512] → [H, 21, 512]
        Hf, n_t, hid = audio_embeds.shape
        audio_embeds = audio_embeds.view(Hf * n_t, hid)
        # → [42, 512]  42=H×21
        audio_embeds = self.proj2(audio_embeds)
        audio_embeds = torch.relu(audio_embeds)
        # → [42, 512]
        tok = self.proj3(audio_embeds)
        # → [42, 24576]  24576=32×768
        tok = tok.view(Hf * n_t, CTX, A_OUT)
        # → [42, 32, 768]
        tok = self.norm(tok)
        tok = tok.view(H, video_length, CTX, A_OUT)
        # → [H, 21, 32, 768]
        return tok


class TinyMultiTalk(nn.Module):
    def __init__(self, num_layers=1):
        super().__init__()
        self.patch_embedding = nn.Conv3d(IN_DIM, DIM, kernel_size=(pt, ph, pw), stride=(pt, ph, pw))
        self.text_embedding = nn.Sequential(nn.Linear(TEXT_DIM, DIM), nn.GELU(approximate="tanh"), nn.Linear(DIM, DIM))
        self.time_embedding = nn.Sequential(nn.Linear(FREQ, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(DIM, DIM * 6))
        self.img_emb = nn.Sequential(
            nn.LayerNorm(CLIP_DIM), nn.Linear(CLIP_DIM, CLIP_DIM), nn.GELU(), nn.Linear(CLIP_DIM, DIM), nn.LayerNorm(DIM)
        )
        self.audio_proj = AudioProjModel()
        self.blocks = nn.ModuleList([WanAttentionBlock() for _ in range(num_layers)])
        self.head_n = nn.LayerNorm(DIM, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(DIM, pt * ph * pw * c)
        self.head_mod = nn.Parameter(torch.randn(1, 2, DIM) / DIM ** 0.5)

    def forward(self, x, t, context, seq_len, clip_fea, y, audio, ref_target_masks):
        """I2V MultiTalk 一步。
        x: list[[c, n_t, h, w]]  噪声潜；c=16, n_t=21, h=w=8
        t: [B]  流匹配时间
        context: list[[L_txt, 4096]]  T5 token
        seq_len: pad 后视频 token 数
        clip_fea: [B, 257, 1280]  CLIP
        y: [B, 20, n_t, h, w]  20=4(mask)+16(条件潜)
        audio: [H, T, k, 12, 768]  H=说话人, T=81 像素帧
        ref_target_masks: [3, h, w]  人1/人2/背景
        → [c, n_t, h, w]  预测速度场
        """
        u = x[0]
        # [16, 21, 8, 8]  16=潜通道, 21=VAE 时间格, 8=潜高宽
        v = y[0]
        # y 为 [1, 20, 21, 8, 8]，沿 dim0 取 → [20, 21, 8, 8]
        # 20=4(mask 通道)+16(条件潜)；官方 zip(list, list)，本脚本 B=1 直取
        x = torch.cat([u, v], dim=0)
        # → [36, 21, 8, 8]  36=IN_DIM
        x = x.unsqueeze(0)
        # → [1, 36, 21, 8, 8]  1=batch
        x = self.patch_embedding(x)
        # → [1, 5120, 21, 4, 4]  5120=DIM, 21 时间不变(pt=1), 4=8/2 空间格
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long).unsqueeze(0)
        # → [1, 3]  3=(N_t, N_h, N_w)
        x = x.flatten(2)
        # [1, 5120, 21, 4, 4] → [1, 5120, 336]  336=21×4×4
        x = x.transpose(1, 2)
        # → [1, 336, 5120]
        n_pad = seq_len - x.size(1)
        pad = x.new_zeros(1, n_pad, x.size(2))
        # [1, n_pad, 5120]  本脚本 seq_len=L，n_pad=0
        x = torch.cat([x, pad], dim=1)
        # → [1, seq_len, 5120]

        e = sinusoidal_embedding_1d(FREQ, t)
        # t [B] → [B, 256]  256=FREQ
        e = self.time_embedding(e)
        # → [B, 5120]
        e0 = self.time_projection(e)
        # → [B, 30720]  30720=6×DIM
        e0 = e0.unflatten(1, (6, DIM))
        # → [B, 6, 5120]

        txt = context[0]
        # [L_txt, 4096]  L_txt≤512, 4096=T5 宽
        txt_pad = txt.new_zeros(TEXT_LEN - txt.size(0), txt.size(1))
        # [512-L_txt, 4096]
        txt = torch.cat([txt, txt_pad], dim=0)
        # → [512, 4096]
        txt = txt.unsqueeze(0)
        # → [1, 512, 4096]
        context = self.text_embedding(txt)
        # → [1, 512, 5120]
        clip_tok = self.img_emb(clip_fea)
        # [1, 257, 1280] → [1, 257, 5120]  257=CLIP token(含 CLS)
        context = torch.cat([clip_tok, context], dim=1)
        # → [1, 769, 5120]  769=257+512

        first = audio[:, :1]
        # [H, 81, 5, 12, 768] → [H, 1, 5, 12, 768]
        # H=说话人, 1=首帧(VAE 不压缩), 5=窗 k, 12=wav2vec 层, 768=特征
        latter = audio[:, 1:]
        # → [H, 80, 5, 12, 768]  80=后续像素帧
        H_a, t_rest, k, n_layer, c_a = latter.shape
        latter = latter.view(H_a, t_rest // vae_scale, vae_scale, k, n_layer, c_a)
        # [H, 80, 5, 12, 768] → [H, 20, 4, 5, 12, 768]
        # 20=后续 VAE 时间格, 4=vae_scale 每格对应像素帧
        latter_first = latter[:, :, 0, :MID + 1]
        # 官方 [:, :, :1, :MID+1] 再 (n w)；n=1 时等价于按下标 0 取
        # → [H, 20, 3, 12, 768]  3=窗 [0, MID]
        latter_mid = latter[:, :, 1:-1, MID]
        # 官方 [:, :, 1:-1, MID:MID+1] 再 (n w)；w=1 时等价于取 MID
        # → [H, 20, 2, 12, 768]  2=中间两个 VAE 子帧, 只取窗中心
        latter_last = latter[:, :, -1, MID:]
        # 官方 [:, :, -1:, MID:] 再 (n w)；n=1 时等价于按下标 -1 取
        # → [H, 20, 3, 12, 768]  3=窗 [MID, k)
        latter = torch.cat([latter_first, latter_mid, latter_last], dim=2)
        # → [H, 20, 8, 12, 768]  8=3+2+3=k+vae_scale-1 打包窗数
        audio_embedding = self.audio_proj(first, latter)
        # first [H, 1, 5, 12, 768], latter [H, 20, 8, 12, 768]
        # → [H, 21, 32, 768]  21=全时长, 32=context token, 768=A_OUT
        person_toks = audio_embedding.split(1, dim=0)
        # [2, 21, 32, 768] → 两个 [1, 21, 32, 768]
        audio_embedding = torch.cat(list(person_toks), dim=2)
        # → [1, 21, 64, 768]  64=32+32 两人拼到同一序列

        ref_target_masks = ref_target_masks.unsqueeze(0).float()
        # [3, 8, 8] → [1, 3, 8, 8]  1=伪 batch, 3=人1/人2/背景, 8=潜高宽
        ref_target_masks = F.interpolate(ref_target_masks, size=(N_h, N_w), mode="nearest")
        # → [1, 3, 4, 4]  4=patch 空间格
        ref_target_masks = ref_target_masks.squeeze(0)
        # → [3, 4, 4]
        ref_target_masks = ref_target_masks > 0
        ref_target_masks = ref_target_masks.view(ref_target_masks.shape[0], -1)
        # → [3, 16]  16=N_h×N_w
        ref_target_masks = ref_target_masks.type_as(x)

        for blk in self.blocks:
            x = blk(x, e0, grid_sizes, context, audio_embedding, ref_target_masks)
            # x [1, 336, 5120] 形状不变

        em = e.unsqueeze(1)
        # [B, 5120] → [B, 1, 5120]
        em = self.head_mod + em
        # [1, 2, DIM] + [B, 1, DIM] → [B, 2, DIM]
        em0, em1 = em.chunk(2, dim=1)
        # 各 [B, 1, DIM]
        x = self.head_n(x)
        x = x * (1 + em1) + em0
        x = self.head(x)
        # [B, L, 5120] → [B, L, 64]  64=pt×ph×pw×c
        x = x[0, :L]
        # → [336, 64]  去掉 batch 与 padding
        x = x.view(N_t, N_h, N_w, pt, ph, pw, c)
        # → [21, 4, 4, 1, 2, 2, 16]  patch 格 × patch 内像素 × 潜通道
        x = torch.einsum("fhwpqrc->cfphqwr", x)
        # → [16, 21, 1, 4, 2, 4, 2]  把 patch 内像素插回格点
        x = x.reshape(c, N_t * pt, N_h * ph, N_w * pw)
        # → [16, 21, 8, 8]
        return x


def causal_vae_encode(video):
    """伪因果 VAE：空间 8× 池化，时间首帧保留、其余每 4 帧平均，通道重复到 16。
    video [C, T, H, W]=[3, 81, 64, 64] → [c, n_t, h, w]=[16, 21, 8, 8]。"""
    video = video.unsqueeze(0)
    # [3, 81, 64, 64] → [1, 3, 81, 64, 64]
    pix = F.avg_pool3d(video, (1, 8, 8))
    # → [1, 3, 81, 8, 8]  空间 /8，时间不变
    pix = pix.squeeze(0)
    # → [3, 81, 8, 8]
    z_first = pix[:, :1]
    # → [3, 1, 8, 8]  首帧不压缩
    z_rest = pix[:, 1:]
    # → [3, 80, 8, 8]
    z_rest = z_rest.reshape(C, t_lat - 1, vae_scale, h, w)
    # → [3, 20, 4, 8, 8]
    z_rest = z_rest.mean(2)
    # → [3, 20, 8, 8]  每 4 像素帧平均成 1 个 VAE 格
    z = torch.cat([z_first, z_rest], dim=1)
    # → [3, 21, 8, 8]
    z = z.repeat((c + C - 1) // C, 1, 1, 1)
    # 沿通道重复 6 次 → [18, 21, 8, 8]
    return z[:c]
    # → [16, 21, 8, 8]


def pack_mask(keep):
    """像素帧掩码打成 VAE 格上的 4 通道。
    keep [T]=[81] → [1, 4, 21, 8, 8]；4=vae_scale 作为通道拼进 IN_DIM。"""
    m = keep.view(1, T, 1, 1)
    # [81] → [1, 81, 1, 1]
    m = m.expand(1, T, h, w).clone()
    # → [1, 81, 8, 8]
    m_first = m[:, :1]
    # → [1, 1, 8, 8]
    m_first = torch.repeat_interleave(m_first, repeats=vae_scale, dim=1)
    # → [1, 4, 8, 8]  首帧重复 vae_scale 次再与后续拼接
    m = torch.cat([m_first, m[:, 1:]], dim=1)
    # [1, 4, 8, 8] + [1, 80, 8, 8] → [1, 84, 8, 8]
    m = m.view(1, t_lat, vae_scale, h, w)
    # → [1, 21, 4, 8, 8]
    m = m.transpose(1, 2)
    # → [1, 4, 21, 8, 8]  4 放到通道维
    return m


if __name__ == "__main__":
    video = torch.randn(C, T, H, W)
    # [3, 81, 64, 64]  3=RGB, 81=像素帧, 64=像素高宽
    keep = torch.zeros(T)
    # [81]
    keep[0] = 1                                      # 仅首帧为已知条件
    x_1 = causal_vae_encode(video)
    # → [16, 21, 8, 8]
    cond_video = torch.cat([video[:, :1], torch.zeros(C, T - 1, H, W)], dim=1)
    # [3, 1, 64, 64] + [3, 80, 64, 64] → [3, 81, 64, 64]  只留首帧像素
    z_c = causal_vae_encode(cond_video)
    # → [16, 21, 8, 8]
    z_c = z_c.unsqueeze(0)
    # → [1, 16, 21, 8, 8]
    mask = pack_mask(keep)
    # → [1, 4, 21, 8, 8]
    y = torch.cat([mask, z_c], dim=1)
    # → [1, 20, 21, 8, 8]  20=4+16
    X_t = 0.3 * x_1 + 0.7 * torch.randn_like(x_1)
    # [16, 21, 8, 8]  流匹配插值潜变量
    audio = torch.randn(2, T, K, WAV_BLOCKS, WAV_DIM)
    # [2, 81, 5, 12, 768]  2=说话人
    masks = torch.zeros(3, h, w)
    # [3, 8, 8]  3=人1/人2/背景
    masks[0, :, : w // 2] = 1
    masks[1, :, w // 2:] = 1
    masks[2] = 1 - (masks[0] + masks[1]).clamp(0, 1)
    v = TinyMultiTalk()(
        [X_t], torch.tensor([300.0]),
        [torch.randn(TEXT_LEN, TEXT_DIM)], L,
        torch.randn(1, CLIP_LEN, CLIP_DIM), y, audio, masks,
    )
    # → [16, 21, 8, 8]  预测速度场
