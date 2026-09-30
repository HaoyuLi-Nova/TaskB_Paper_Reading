"""
LayerNorm 逐步实现：输入 x 形状为 (B, N, D)。

B = batch，N = token 数（序列长 / patch 数），D = 每个 token 的通道维。

关键事实：统计量只沿最后一维 D 计算。B 与 N 完全不混合——
每个 token 的 D 维向量单独标准化，互不影响。

运行:
  cd code/foundations/adaln
  python layernorm_demo.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """与 ``nn.LayerNorm(D)`` 对齐的手写版，专门吃 (B, N, D)。

    公式（对每一个 token 的 D 维向量独立）：

        μ    = mean(x, dim=-1)                         # (B, N, 1)
        σ²   = mean((x - μ)², dim=-1)                  # (B, N, 1)
        x̂    = (x - μ) / sqrt(σ² + ε)                  # (B, N, D)
        y    = γ ⊙ x̂ + β                               # γ, β: (D,)
    """

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))   # γ
            self.bias = nn.Parameter(torch.zeros(dim))    # β
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)  — 也兼容 (..., D)，只要最后一维是 D
        Returns:
            同形状
        """
        if x.size(-1) != self.dim:
            raise ValueError(f"last dim {x.size(-1)} != LayerNorm.dim {self.dim}")

        # ---- 1. 每个 token 自己的均值：把 D 维压成标量，keepdim 方便广播 ----
        # x:    (B, N, D)
        # mean: (B, N, 1)   ← 一共 B*N 个独立的 μ，不是 1 个全局 μ
        mean = x.mean(dim=-1, keepdim=True)

        # ---- 2. 总体方差（unbiased=False，与 nn.LayerNorm 一致，分母是 D 不是 D-1）----
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, N, 1)

        # ---- 3. 标准化：mean/var 沿 D 广播回 (B, N, D) ----
        x_hat = (x - mean) * torch.rsqrt(var + self.eps)

        # ---- 4. 仿射：γ, β 形状 (D,)，广播到每个 token ----
        if self.elementwise_affine:
            return x_hat * self.weight + self.bias
        return x_hat


def per_token_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回每个 token 沿 D 的均值 / 方差，形状 (B, N)。"""
    mean = x.mean(dim=-1)
    var = x.var(dim=-1, unbiased=False)
    return mean, var


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def demo_shapes():
    banner("1) 形状逐步跟踪  x: (B=2, N=3, D=4)")

    torch.manual_seed(0)
    x = torch.tensor(
        [
            [  # batch 0：3 个 token
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 10.0, 10.0, 10.0],  # 常数 token，方差为 0
                [-1.0, 0.0, 1.0, 2.0],
            ],
            [  # batch 1
                [0.0, 0.0, 0.0, 1.0],
                [5.0, 6.0, 7.0, 8.0],
                [1.0, 1.0, 2.0, 3.0],
            ],
        ]
    )
    print("x.shape =", tuple(x.shape))

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    print("mean.shape =", tuple(mean.shape), "  # (B, N, 1)，每个 token 一个 μ")
    print("var.shape  =", tuple(var.shape))
    print("token[0,0] 的 4 个数:", x[0, 0].tolist(), " μ =", float(mean[0, 0]))
    print("token[0,1] 的 4 个数:", x[0, 1].tolist(), " μ =", float(mean[0, 1]), "  (常数 → μ 就是它自己)")

    ln = LayerNorm(4, elementwise_affine=False)
    y = ln(x)
    m, v = per_token_stats(y)
    print("标准化后每个 token 的均值（应 ≈ 0）:\n", m)
    print("标准化后每个 token 的方差（应 ≈ 1；常数 token 因 ε 变成 0）:\n", v)


def demo_tokens_are_independent():
    banner("2) token 之间互不混合")

    torch.manual_seed(1)
    x = torch.randn(1, 2, 8)
    ln = LayerNorm(8, elementwise_affine=False)

    y_both = ln(x)
    y_only0 = ln(x[:, :1, :])          # 只送第 0 个 token
    y_only1 = ln(x[:, 1:, :])          # 只送第 1 个 token

    d0 = float((y_both[:, 0:1, :] - y_only0).abs().max())
    d1 = float((y_both[:, 1:2, :] - y_only1).abs().max())
    print("一起归一化 vs 单独归一化 token0 之差:", d0)
    print("一起归一化 vs 单独归一化 token1 之差:", d1)
    print("→ 改旁边那个 token、甚至改 batch 里别的样本，都不会动当前 token 的输出。")


def demo_affine_broadcast():
    banner("3) γ, β 只有 (D,)，对 B 和 N 广播")

    x = torch.randn(2, 5, 4)
    ln = LayerNorm(4)
    with torch.no_grad():
        ln.weight.copy_(torch.tensor([2.0, 0.5, 1.0, 3.0]))
        ln.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))

    y = ln(x)
    # 关掉仿射的标准化结果
    x_hat = LayerNorm(4, elementwise_affine=False)(x)
    reconstructed = x_hat * ln.weight + ln.bias
    print("γ =", ln.weight.detach().tolist())
    print("β =", ln.bias.detach().tolist())
    print("手写 y 与 x̂*γ+β 之差:", float((y - reconstructed).detach().abs().max()))
    print("γ.shape =", tuple(ln.weight.shape), "  不是 (B,N,D)；同一通道上所有 token 共用一个 γ_d")


def demo_vs_pytorch():
    banner("4) 对齐 nn.LayerNorm(D)")

    torch.manual_seed(2)
    x = torch.randn(3, 7, 16)
    ours = LayerNorm(16)
    official = nn.LayerNorm(16)
    with torch.no_grad():
        official.weight.copy_(ours.weight)
        official.bias.copy_(ours.bias)

    diff = float((ours(x) - official(x)).abs().max())
    print("手写 vs nn.LayerNorm 最大差:", diff, "  （应在 1e-6 量级）")
    print("normalized_shape=16  ⇒  PyTorch 对最后 1 维做 LN，(B,N) 当 batch 轴看待")


def demo_not_batchnorm():
    banner("5) 对照：若误对 B 做归一化（那是 BatchNorm 的思路）")

    torch.manual_seed(3)
    x = torch.randn(4, 6, 8)
    # LayerNorm：每个 (b,n) 沿 D
    ln_mean = x.mean(dim=-1)  # (B, N)
    # **错误的**沿 batch 归一化某个通道：每个 d 沿 (B, N)
    bn_like_mean = x.mean(dim=(0, 1))  # (D,)

    print("LayerNorm 的 μ 形状:", tuple(ln_mean.shape), "  # 每个 token 一个")
    print("若沿 (B,N) 求均值的形状:", tuple(bn_like_mean.shape), "  # 每个通道一个，这是 BN")
    print("→ Transformer 里 LN 回答的是：这个 token 自己的 D 维要均值为 0；")
    print("   不回答**这个通道在整个 batch 上均值为 0**。所以 LN 推理时不依赖 batch 统计。")


def main():
    demo_shapes()
    demo_tokens_are_independent()
    demo_affine_broadcast()
    demo_vs_pytorch()
    demo_not_batchnorm()
    print()


if __name__ == "__main__":
    main()
