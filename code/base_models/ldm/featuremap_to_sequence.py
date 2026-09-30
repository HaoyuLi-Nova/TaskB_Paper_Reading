#!/usr/bin/env python3
"""LDM §3.3：特征图如何展平为序列，再做交叉注意力（最小可运行实现）。

对照 translations/base_models/ldm/paper_zh.md：

  φ_i(z_t) ∈ R^{N × d}   ← UNet 中间特征图展平后的序列（Q 来源）
  τ_θ(y)  ∈ R^{M × d_τ} ← 文本等条件的 token 序列（K/V 来源）

  Q = W_Q · φ_i(z_t)
  K = W_K · τ_θ(y)
  V = W_V · τ_θ(y)

与 CompVis SpatialTransformer 同构：b c h w → b (h·w) c → attn → 再 reshape 回去。

运行：
  python code/base_models/ldm/featuremap_to_sequence.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def featuremap_to_sequence(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """(B, C, H, W) → (B, N, C)，N = H·W。每个空间位置是一个 token。"""
    b, c, h, w = x.shape
    seq = x.flatten(2).transpose(1, 2)  # B, HW, C
    return seq, h, w


def sequence_to_featuremap(seq: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, N, C) → (B, C, H, W)。"""
    b, n, c = seq.shape
    assert n == h * w
    return seq.transpose(1, 2).reshape(b, c, h, w)


class CrossAttention(nn.Module):
    """论文式交叉注意力：Q←图像序列，K/V←条件序列。"""

    def __init__(self, query_dim: int, context_dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) 图像 token；context: (B, M, C_ctx) 条件 token
        b, n, _ = x.shape
        m = context.shape[1]
        h = self.heads
        d = self.to_q.out_features // h

        q = self.to_q(x).reshape(b, n, h, d).permute(0, 2, 1, 3)  # B,H,N,D
        k = self.to_k(context).reshape(b, m, h, d).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(b, m, h, d).permute(0, 2, 1, 3)

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, h * d)
        return self.to_out(out)


class SpatialTransformerBlock(nn.Module):
    """UNet 里插的一块：特征图 → 序列 → (自注意力 + 交叉注意力) → 特征图。"""

    def __init__(self, channels: int, context_dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.attn1 = CrossAttention(channels, channels, heads, dim_head)  # self-attn：context=x
        self.attn2 = CrossAttention(channels, context_dim, heads, dim_head)  # cross-attn
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.proj_in(self.norm(x))
        seq, h, w = featuremap_to_sequence(x)  # φ_i(z_t): B, N, C

        seq = seq + self.attn1(self.norm1(seq), context=seq)       # 自注意力
        seq = seq + self.attn2(self.norm2(seq), context=context)   # 交叉注意力 ← 文本
        seq = seq + self.ff(seq)

        x = sequence_to_featuremap(seq, h, w)
        return residual + self.proj_out(x)


def demo() -> None:
    torch.manual_seed(0)
    # 模拟 UNet 某一层：B=2, C=128, H=W=16 → N=256 个空间 token
    feat = torch.randn(2, 128, 16, 16)
    # 模拟 τ_θ(y)：一句 prompt 的 M=77 个 token（如 CLIP/BERT）
    text = torch.randn(2, 77, 768)

    # 1) 只看展平
    seq, h, w = featuremap_to_sequence(feat)
    print(f"feature map  {tuple(feat.shape)}  →  sequence {tuple(seq.shape)}  (N=H·W={h*w})")
    back = sequence_to_featuremap(seq, h, w)
    assert torch.allclose(feat, back)

    # 2) 交叉注意力（需把 text 先投到与通道可对齐的空间；此处用线性模拟）
    ctx_proj = nn.Linear(768, 128)
    context = ctx_proj(text)  # B, 77, 128  ~ τ 后再进 attn 的 K/V 侧
    block = SpatialTransformerBlock(channels=128, context_dim=128)
    out = block(feat, context)
    print(f"after SpatialTransformer  {tuple(out.shape)}  (仍是特征图，可继续卷积)")
    print("OK")


if __name__ == "__main__":
    demo()
