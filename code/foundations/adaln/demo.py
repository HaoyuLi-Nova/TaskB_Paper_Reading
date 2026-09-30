"""
AdaLN 数值演示：恒等初始化、时间步调制、跨模态门控。

运行:
  cd code/foundations/adaln
  python demo.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adaln import (
    AdaLNSingle,
    AdaLNSingleBlock,
    AdaLNZero,
    AdaLNZeroBlock,
    CrossModalityAdaLN,
    TimestepEmbedder,
    layer_norm,
    modulate,
    rms_norm,
)


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def demo_ln_vs_adaln():
    """同一条序列、两个时间步：LayerNorm 不变，AdaLN 的仿射随 t 变。"""
    banner("1) LayerNorm 固定仿射 vs AdaLN 条件仿射")

    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)  # (B, L, C)
    ln_out = layer_norm(x)

    adaln = AdaLNZero(dim=8, cond_dim=8)
    # 故意不零初始化，方便看出**不同 t → 不同仿射**
    nn.init.normal_(adaln.proj.weight, std=0.3)
    nn.init.zeros_(adaln.proj.bias)

    c0 = torch.zeros(2, 8)
    c1 = torch.ones(2, 8)
    s0, sh0, *_ = adaln(c0)
    s1, sh1, *_ = adaln(c1)

    y0 = modulate(ln_out, s0, sh0)
    y1 = modulate(ln_out, s1, sh1)

    print("x  shape:", tuple(x.shape))
    print("LN(x) 不依赖条件，两次调用之差:", float((ln_out - layer_norm(x)).abs().max()))
    print("AdaLN(x | t=0) vs AdaLN(x | t=1) 最大差:", float((y0 - y1).detach().abs().max()))
    print("scale(t=0)[:4]:", s0[0, 0, :4].detach().tolist())
    print("scale(t=1)[:4]:", s1[0, 0, :4].detach().tolist())
    print("→ 条件变了，同一 token 被拉到不同的尺度/偏置。")


def demo_adaln_zero_identity():
    """AdaLN-Zero 零初始化：整块是恒等映射。"""
    banner("2) AdaLN-Zero：零初始化 ⇒ 块 = 恒等")

    torch.manual_seed(1)
    dim, n_heads, cond_dim = 32, 4, 32
    block = AdaLNZeroBlock(dim, n_heads, cond_dim)
    x = torch.randn(3, 6, dim)
    c = torch.randn(3, cond_dim)

    with torch.no_grad():
        y = block(x, c)
        delta = (y - x).abs().max().item()

    chunks = block.adaln(c)
    names = ["scale_msa", "shift_msa", "gate_msa", "scale_mlp", "shift_mlp", "gate_mlp"]
    print("初始化后 6 组调制的 max|·|（都应 ≈ 0）:")
    for name, ch in zip(names, chunks):
        print(f"  {name:10s} {float(ch.abs().max()):.2e}")
    print(f"forward 后 ||y - x||_∞ = {delta:.2e}  （应为 0）")
    print("→ 深层 DiT 从恒等残差起步，训练中 gate 慢慢打开。")


def demo_adaln_single_shared():
    """AdaLN-Single：一份共享 MLP，两块各有自己的 table。"""
    banner("3) AdaLN-Single：共享 MLP + 每块 table")

    torch.manual_seed(2)
    dim, n_heads = 16, 4
    shared = AdaLNSingle(dim, n_params=6)
    b0 = AdaLNSingleBlock(dim, n_heads)
    b1 = AdaLNSingleBlock(dim, n_heads)
    # 给两块不同的 table，模拟**层间偏好不同**
    nn.init.normal_(b0.scale_shift_table, std=0.2)
    nn.init.normal_(b1.scale_shift_table, std=0.2)

    t = torch.tensor([0.3, 0.9])
    x = torch.randn(2, 5, dim)
    emb = shared(t)  # 只算一次，送给每一层

    y0 = b0(x, emb)
    y1 = b1(x, emb)
    print("shared MLP 输出 shape:", tuple(emb.shape), "  # (B, 6*dim)，全层共用")
    print("块0 / 块1 对同一 x, t 的输出差:", float((y0 - y1).abs().mean()))
    print("table0[scale_msa][:4]:", b0.scale_shift_table[0, :4].detach().tolist())
    print("table1[scale_msa][:4]:", b1.scale_shift_table[0, :4].detach().tolist())
    print("→ 时间步嵌入算一次；层间差异全部来自那张小表。这是 PixArt / LTX 的做法。")


def demo_cross_modality_gate():
    """跨模态 AdaLN：对方时间步决定 gate；gate=0 时跨模态通路关闭。"""
    banner("4) Cross-modality AdaLN：门控来自另一模态")

    torch.manual_seed(3)
    dim_v, dim_a, n_heads = 16, 12, 4
    a2v = CrossModalityAdaLN(dim_q=dim_v, dim_kv=dim_a, n_heads=n_heads)
    # 构造时 gate 是 0（恒等）。这里把 gate 打开，模拟训练中途。
    nn.init.constant_(a2v.gate_table, 0.5)

    t_embed_v = TimestepEmbedder(dim_v)
    t_embed_a = TimestepEmbedder(dim_a)

    video = torch.randn(2, 8, dim_v)
    audio = torch.randn(2, 5, dim_a)
    t_v = t_embed_v(torch.tensor([0.4, 0.4]))
    t_a = t_embed_a(torch.tensor([0.7, 0.7]))

    with torch.no_grad():
        out_open = a2v(video, audio, t_v, t_a)
        # 强制关闭跨模态通路：模拟**另一模态在该步不注入**
        a2v.gate_table.zero_()
        out_closed = a2v(video, audio, t_v, t_a)

    print("视频 token:", tuple(video.shape), " 音频 token:", tuple(audio.shape))
    print("正常前向  ||out - video||_∞:", float((out_open - video).abs().max()))
    print("gate≡0    ||out - video||_∞:", float((out_closed - video).abs().max()), "  （应为 0）")
    print("→ 音频时间步通过 gate 决定这一去噪阶段掺入多少声音信息。")
    print("   LTX-2 用它处理**两流扩散步 / 时间分辨率不一致**的同步问题。")


def demo_broadcast_shapes():
    """强调调制是 per-channel、对序列广播，不是 per-token。"""
    banner("5) 广播形状：条件是全局的，作用在每个 token 的通道上")

    x = torch.randn(2, 5, 8)
    scale = torch.arange(8).float().view(1, 1, 8) * 0.1  # (1, 1, C)
    shift = torch.zeros(1, 1, 8)
    y = modulate(rms_norm(x), scale, shift)

    # 同一通道上，所有 token 乘同一个 (1+scale_c)
    ratio = y[0, :, 3] / (rms_norm(x)[0, :, 3] + 1e-8)
    print("通道 3 上 5 个 token 的 (1+scale) 比值:", [round(v, 4) for v in ratio.tolist()])
    print("应全部等于 1 + 0.3 =", 1.3)
    print("→ AdaLN 不给每个 token 单独的 γ；条件是样本级（或时间步级）的。")


def main():
    demo_ln_vs_adaln()
    demo_adaln_zero_identity()
    demo_adaln_single_shared()
    demo_cross_modality_gate()
    demo_broadcast_shapes()
    print()


if __name__ == "__main__":
    main()
