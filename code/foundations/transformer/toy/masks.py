"""
Padding / causal masks — 与 PyTorch nn.MultiheadAttention / nn.Transformer 约定一致。

官方约定（batch_first=True）:
  key_padding_mask: (B, L), dtype=bool
      True  = pad，该 key 位置在注意力中被忽略
      False = 有效 token

实现方式: 在 softmax 前把对应 logit 填成 -inf。
"""

from __future__ import annotations

import torch


def make_key_padding_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Args:
        token_ids: (B, L)  long
        pad_id:    pad token id
    Returns:
        key_padding_mask: (B, L) bool, True 表示 pad
    """
    return token_ids.eq(pad_id)


def apply_key_padding_mask(
    scores: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    把 pad 位置的 attention logit 设为 -inf。

    Args:
        scores: (B, H, Lq, Lk) 或 (B, Lq, Lk)
        key_padding_mask: (B, Lk), True = pad
    """
    if key_padding_mask is None:
        return scores

    # (B, Lk) -> (B, 1, 1, Lk) 以便广播到 (B, H, Lq, Lk)
    mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
    return scores.masked_fill(mask, float("-inf"))


def make_causal_mask(size: int, device=None, dtype=torch.bool) -> torch.Tensor:
    """
    Decoder 自注意力用的因果 mask（与 PyTorch 布尔约定一致）:
      True  = 屏蔽（不可看未来）
      False = 允许

    返回 (L, L)，上三角（不含对角）为 True。
    """
    # torch.nn.Transformer.generate_square_subsequent_mask 返回的是 float 加性 mask；
    # 这里给布尔版，语义与 key_padding_mask 一致（True=屏蔽）。
    return torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)


def apply_attn_mask(
    scores: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Args:
        scores: (B, H, L, L)
        attn_mask: (L, L) bool, True=屏蔽；或 (L, L) float 加性 mask
    """
    if attn_mask is None:
        return scores
    if attn_mask.dtype == torch.bool:
        return scores.masked_fill(attn_mask, float("-inf"))
    # float additive mask（官方 generate_square_subsequent_mask 风格）
    return scores + attn_mask
