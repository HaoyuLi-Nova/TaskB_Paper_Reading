"""
玩具演示：batch=4、句长不等 → pad → key_padding_mask → Encoder。

运行:
  cd code/foundations/transformer/toy
  python demo.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from masks import make_key_padding_mask
from model import ToyTransformerEncoder


PAD_ID = 0
VOCAB = 50


def collate_variable_lengths(seqs: list[torch.Tensor], pad_id: int = PAD_ID):
    """
    官方常见写法：pad_sequence(..., batch_first=True, padding_value=pad_id)
    得到 (B, L_max)，右侧 pad。
    """
    return pad_sequence(seqs, batch_first=True, padding_value=pad_id)


def show_padding_mask_math():
    """用数值说明 mask 如何把 pad 列变成权重 0。"""
    print("=" * 60)
    print("1) Padding mask 数值示意（单头、1 条样本）")
    print("=" * 60)

    # 有效长度 3，pad 到 5
    # scores: (Lq=5, Lk=5)，最后两列对应 pad keys
    scores = torch.tensor(
        [
            [1.0, 0.5, 0.2, 9.0, 9.0],  # 若无 mask，会错误地猛看 pad
            [0.3, 1.2, 0.4, 9.0, 9.0],
            [0.1, 0.2, 1.0, 9.0, 9.0],
            [0.0, 0.0, 0.0, 9.0, 9.0],  # pad query（可选处理）
            [0.0, 0.0, 0.0, 9.0, 9.0],
        ]
    )
    key_padding_mask = torch.tensor([False, False, False, True, True])  # True=pad

    masked = scores.masked_fill(key_padding_mask.unsqueeze(0), float("-inf"))
    attn = F.softmax(masked, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    print("key_padding_mask (True=pad):", key_padding_mask.tolist())
    print("masked scores[0]:", masked[0].tolist())
    print("softmax 后 attn[0]（pad 列应为 0）:", [round(x, 4) for x in attn[0].tolist()])
    print("pad 列权重和:", float(attn[0, 3:].sum()))


def main():
    show_padding_mask_math()

    print("\n" + "=" * 60)
    print("2) 变长 batch → pad → Encoder（对齐官方 pad 习惯）")
    print("=" * 60)

    # 4 条句子，长度不同（token id 从 1 开始，0 留给 PAD）
    raw = [
        torch.tensor([3, 8, 2]),
        torch.tensor([5, 1, 9, 4, 7, 2]),
        torch.tensor([6, 2]),
        torch.tensor([4, 4, 4, 4, 4, 4, 4, 2]),
    ]
    lengths = [t.numel() for t in raw]
    batch = collate_variable_lengths(raw, PAD_ID)  # (4, L_max)
    mask = make_key_padding_mask(batch, PAD_ID)

    print("lengths:", lengths)
    print("batch token_ids shape:", tuple(batch.shape))
    print("batch:\n", batch)
    print("key_padding_mask (True=pad):\n", mask)

    model = ToyTransformerEncoder(
        vocab_size=VOCAB,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        pad_id=PAD_ID,
    )
    model.eval()
    with torch.no_grad():
        out = model(batch)  # (B, L, d)

    print("encoder out shape:", tuple(out.shape))
    # 有效位置与 pad 位置都有向量；语义上应只使用 ~mask 的位置
    print("有效位置数:", int((~mask).sum()))
    print("pad 位置数:", int(mask.sum()))

    # 与 PyTorch 官方 MultiheadAttention 对照：同一 mask 约定
    print("\n" + "=" * 60)
    print("3) 与 nn.MultiheadAttention 的 mask 约定对照")
    print("=" * 60)
    mha = torch.nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
    x = torch.randn(4, batch.size(1), 64)
    # 官方: key_padding_mask True = ignore
    y, w = mha(x, x, x, key_padding_mask=mask, need_weights=True, average_attn_weights=True)
    # w: (B, L, L)；看第 0 句 pad 列
    L0 = lengths[0]
    pad_cols = w[0, :L0, L0:].abs().sum().item()
    print("官方 MHA: 有效 query 对 pad key 的权重绝对值和 ≈", round(pad_cols, 6), "(应接近 0)")


if __name__ == "__main__":
    main()
