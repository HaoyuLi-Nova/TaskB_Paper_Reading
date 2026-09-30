"""
迷你 Transformer Encoder（结构对齐原版 / PyTorch 习惯）。

- batch_first=True
- 多头注意力 + 残差 + LayerNorm(post-norm，与原论文一致)
- FFN: d_model -> d_ff -> d_model
- key_padding_mask: (B, L) True=pad，与 nn.MultiheadAttention 一致
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from masks import apply_attn_mask, apply_key_padding_mask


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 与常见实现一致：一次投影再拆头（等价于每头一套 W）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            key_padding_mask: (B, L) bool, True=pad
            attn_mask: (L, L) optional causal / 其他
        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_k

        # (B, L, d) -> (B, H, L, d_k)
        q = self.w_q(x).view(B, L, H, D).transpose(1, 2)
        k = self.w_k(x).view(B, L, H, D).transpose(1, 2)
        v = self.w_v(x).view(B, L, H, D).transpose(1, 2)

        # scores: (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        scores = apply_key_padding_mask(scores, key_padding_mask)
        scores = apply_attn_mask(scores, attn_mask)

        attn = F.softmax(scores, dim=-1)
        # pad 行可能全是 -inf → nan；有效实现里常再 mask，这里用 nan_to_num 保稳定
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Post-LN Encoder 层（Attention Is All You Need）。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 子层1: self-attn
        x = self.norm1(x + self.dropout(self.self_attn(x, key_padding_mask)))
        # 子层2: FFN
        x = self.norm2(x + self.ff(x))
        return x


class PositionalEncoding(nn.Module):
    """正弦位置编码（与论文一致，无学习参数）。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ToyTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_len: int = 128,
        pad_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        # 论文：embedding 乘 sqrt(d_model)；与共享 softmax 配套，玩具里也保留
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) long，右侧 padding
        Returns:
            (B, L, d_model)
        """
        from masks import make_key_padding_mask

        key_padding_mask = make_key_padding_mask(token_ids, self.pad_id)  # (B, L)
        x = self.embed(token_ids) * self.scale
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x
