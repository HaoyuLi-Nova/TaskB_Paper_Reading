"""
AdaLN（Adaptive Layer Normalization）玩具实现。

谱系：
  LayerNorm / RMSNorm  固定仿射 γ, β
       ↓ 仿射参数改由条件 c 生成（FiLM）
  AdaLN                γ(c), β(c)
       ↓ 残差门控 + 零初始化
  AdaLN-Zero           DiT（Peebles & Xie, 2023）每块独立 MLP
       ↓ 全层共享 MLP + 每块 bias 表
  AdaLN-Single         PixArt-α / LTX 系列
       ↓ 一流的 scale/shift/gate 由另一流条件化
  Cross-modality AdaLN LTX-2

核心公式（与 DiT / LTX-2 一致，scale 写成 1+scale 以便零初始化即恒等）：

    h = Norm(x) * (1 + scale(c)) + shift(c)
    x ← x + gate(c) ⊙ SubLayer(h)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 无仿射 Norm（仿射交给 AdaLN）
# ---------------------------------------------------------------------------


def layer_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """沿最后一维做 LayerNorm，不含可学习 γ, β。"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps)


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm：不做均值中心化。LTX-2 Transformer 用这个。"""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """FiLM / AdaLN 仿射：x * (1 + scale) + shift。

    写成 ``1 + scale`` 而不是直接乘 scale，是为了让 scale=0 时仿射为恒等。
    scale / shift 的广播形状通常是 (B, 1, C)，作用在序列的每一个 token 上。
    """
    return x * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# 2. 时间步嵌入（正弦，与 Transformer 位置编码同族）
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """标量 t ∈ [0, 1] 或整数扩散步 → (B, dim) 向量。"""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) 浮点时间步
        Returns:
            (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# 3. AdaLN-Zero（DiT）：每块自己的条件 MLP，零初始化 → 块起步为恒等
# ---------------------------------------------------------------------------


class AdaLNZero(nn.Module):
    """从条件向量 c 回归 6 组调制参数：MSA 与 MLP 各 (scale, shift, gate)。

    DiT 把最后一层 Linear 零初始化，于是训练开始时：
        scale = 0, shift = 0, gate = 0
        → modulate(Norm(x), 0, 0) = Norm(x)
        → x + 0 * SubLayer(...) = x
    深层 Transformer 因此能从**恒等映射**慢慢打开，而不是随机扰动。
    """

    NUM_CHUNKS = 6  # scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.proj = nn.Linear(cond_dim, self.NUM_CHUNKS * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            c: (B, cond_dim)
        Returns:
            6 个 (B, 1, dim) 张量，可直接对 (B, L, dim) 广播。
        """
        chunks = self.proj(self.silu(c)).chunk(self.NUM_CHUNKS, dim=-1)
        return tuple(ch[:, None, :] for ch in chunks)


class AdaLNZeroBlock(nn.Module):
    """一个最小 DiT 块：AdaLN → Self-Attn → AdaLN → FFN，两处残差都带 gate。"""

    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = layer_norm
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.adaln = AdaLNZero(dim, cond_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, dim) token 序列
            c: (B, cond_dim) 条件（时间步 / 类别嵌入之和）
        """
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.adaln(c)

        h = modulate(self.norm(x), scale_msa, shift_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa * attn_out

        h = modulate(self.norm(x), scale_mlp, shift_mlp)
        x = x + gate_mlp * self.ff(h)
        return x


# ---------------------------------------------------------------------------
# 4. AdaLN-Single（PixArt-α / LTX）：全层共享 MLP + 每块可学习表
# ---------------------------------------------------------------------------


class AdaLNSingle(nn.Module):
    """共享的条件 MLP：所有 Transformer 块共用这一份。

    输出 (B, n_params * dim)，再按块切分。PixArt-α / LTX 默认 n_params=6；
    若还有文本交叉注意力 AdaLN，LTX 会扩到 9。
    """

    def __init__(self, dim: int, n_params: int = 6):
        super().__init__()
        self.n_params = n_params
        self.t_embed = TimestepEmbedder(dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, n_params * dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) 时间步
        Returns:
            (B, n_params * dim) 全局调制向量
        """
        return self.linear(self.silu(self.t_embed(t)))


class AdaLNSingleBlock(nn.Module):
    """块内只有一张可学习 scale_shift_table，没有独立的条件 MLP。

        ada = table + shared_mlp(t)

    28 层 × 6 × dim 的 Linear 变成 1 份共享 Linear + 28 张小表，省参数。
    LTX-2 的 ``scale_shift_table + timestep`` 就是这个。
    """

    def __init__(self, dim: int, n_heads: int, n_params: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.n_params = n_params
        self.norm = rms_norm
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # (n_params, dim)：每块自己的偏置，训练中吸收**这一层偏好的调制**
        self.scale_shift_table = nn.Parameter(torch.zeros(n_params, dim))

    def _ada(self, shared: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """shared: (B, n_params * dim) → n_params 个 (B, 1, dim)。"""
        b, _ = shared.shape
        merged = self.scale_shift_table[None] + shared.view(b, self.n_params, -1)
        return tuple(merged[:, i, None, :] for i in range(self.n_params))

    def forward(self, x: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self._ada(shared)

        h = modulate(self.norm(x), scale_msa, shift_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa * attn_out

        h = modulate(self.norm(x), scale_mlp, shift_mlp)
        x = x + gate_mlp * self.ff(h)
        return x


# ---------------------------------------------------------------------------
# 5. Cross-modality AdaLN（LTX-2）：门控来自另一模态的时间步
# ---------------------------------------------------------------------------


class CrossModalityAdaLN(nn.Module):
    """跨模态交叉注意力上的 AdaLN（结构对齐 LTX-2 论文 3.1.2，刻意缩小）。

    对**视频 query 看音频 KV**这一方向：
      - Q  的 scale/shift 由 **视频时间步** t_v 生成 → 本模态决定**我现在接多少**
      - KV 的 scale/shift 由 **音频时间步** t_a 生成 → 对方决定**我现在暴露多少**
      - 输出 gate              由 **音频时间步** t_a 生成 → 在该去噪阶段掺入多少跨模态信息

    反向（音频 query 看视频 KV）对称。两模态扩散步 / 时间分辨率可以不同。
    """

    def __init__(self, dim_q: int, dim_kv: int, n_heads: int):
        super().__init__()
        self.norm_q = rms_norm
        self.norm_kv = rms_norm
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_q,
            num_heads=n_heads,
            kdim=dim_kv,
            vdim=dim_kv,
            batch_first=True,
        )
        # 本模态：scale, shift（作用在 Q）
        self.q_table = nn.Parameter(torch.zeros(2, dim_q))
        self.q_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim_q, 2 * dim_q))
        # 对方模态：scale, shift（作用在 KV）+ gate（作用在注意力输出）
        self.kv_table = nn.Parameter(torch.zeros(2, dim_kv))
        self.kv_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim_kv, 2 * dim_kv))
        self.gate_table = nn.Parameter(torch.zeros(1, dim_q))
        self.gate_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim_kv, dim_q))

        nn.init.zeros_(self.q_mlp[1].weight)
        nn.init.zeros_(self.q_mlp[1].bias)
        nn.init.zeros_(self.kv_mlp[1].weight)
        nn.init.zeros_(self.kv_mlp[1].bias)
        nn.init.zeros_(self.gate_mlp[1].weight)
        nn.init.zeros_(self.gate_mlp[1].bias)

    def _affine(self, table: nn.Parameter, mlp: nn.Module, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n = table.shape[0]
        merged = table[None] + mlp(t_emb).view(t_emb.size(0), n, -1)
        return tuple(merged[:, i, None, :] for i in range(n))

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        t_query: torch.Tensor,
        t_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:     (B, Lq, dim_q)   本模态隐状态
            context:   (B, Lk, dim_kv)  另一模态隐状态
            t_query:   (B, dim_q)       本模态时间步嵌入
            t_context: (B, dim_kv)      另一模态时间步嵌入
        """
        scale_q, shift_q = self._affine(self.q_table, self.q_mlp, t_query)
        scale_kv, shift_kv = self._affine(self.kv_table, self.kv_mlp, t_context)
        (gate,) = self._affine(self.gate_table, self.gate_mlp, t_context)

        q = modulate(self.norm_q(query), scale_q, shift_q)
        kv = modulate(self.norm_kv(context), scale_kv, shift_kv)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        return query + gate * attn_out
