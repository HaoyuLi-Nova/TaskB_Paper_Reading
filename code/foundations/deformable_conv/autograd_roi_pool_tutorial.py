"""
PyTorch 自动求导 + RoI 池化循环 —— 教学脚本
==========================================

核心问题
--------
**一张图里多个 RoI、Python for 循环逐个池化，梯度还能传回去吗？**

答案：能。因为：
  1. autograd 跟踪的是**张量运算**，不是**Python 控制流**；
  2. for 循环只是按顺序往计算图里多挂了若干节点；
  3. 多个 RoI 采到同一特征像素时，反向时梯度会在该像素处相加（accumulate）。

运行：
  python scripts/autograd_roi_pool_tutorial.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# =============================================================================
# 0. 最小计算图：forward 建图，backward 沿图反传
# =============================================================================


def demo_basic_graph() -> None:
    print("\n" + "=" * 60)
    print("0. 最小计算图")
    print("=" * 60)

    x = torch.tensor(2.0, requires_grad=True)  # leaf
    y = x * x + 3 * x                          # y = x^2 + 3x
    # 反向：dy/dx = 2x + 3 = 7

    print(f"  x = {x.item()}, y = {y.item()}")
    print(f"  y.grad_fn = {y.grad_fn}")  # AddBackward → 说明 y 不是叶子，由运算产生

    y.backward()
    print(f"  x.grad = {x.grad.item()}  (期望 2*2+3=7)")

    # 关键：requires_grad=True 的叶子才在 .grad 里积累梯度；
    # 中间量默认不保留，除非 retain_grad()。


# =============================================================================
# 1. Python for 循环不会断梯度
# =============================================================================


def demo_loop_still_differentiable() -> None:
    print("\n" + "=" * 60)
    print("1. for 循环只是**多次挂节点**，不断图")
    print("=" * 60)

    w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    s = torch.tensor(0.0)
    for i in range(3):
        # 每次迭代：s = s + w[i]^2
        # 图上等价于：s0=0 → s1=s0+w0^2 → s2=s1+w1^2 → s3=s2+w2^2
        s = s + w[i] * w[i]

    print(f"  s = {s.item()}  (期望 1+4+9=14)")
    s.backward()
    print(f"  w.grad = {w.grad.tolist()}  (期望 [2,4,6] = 2w)")

    # 对比：写成向量化 s = (w*w).sum()，数学与图结构等价，只是节点更少。


# =============================================================================
# 2. 多个**消费者**共享同一张量 → 梯度相加
# =============================================================================


def demo_grad_accumulate() -> None:
    print("\n" + "=" * 60)
    print("2. 共享输入时，反向梯度自动相加")
    print("=" * 60)

    feat = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # 两个**RoI**都用到 feat[0,0]：
    #   a = feat[0,0]
    #   b = feat[0,0] * 10
    #   loss = a + b = 11 * feat[0,0]
    a = feat[0, 0]
    b = feat[0, 0] * 10.0
    loss = a + b
    loss.backward()

    print(f"  loss = {loss.item()}")
    print(f"  feat.grad =\n{feat.grad}")
    print("  → 位置 (0,0) 的梯度 = 1 + 10 = 11（两路贡献相加）")
    print("  → 其它位置为 0（没被用到）")


# =============================================================================
# 3. 迷你 RoI 平均池化：循环实现 + 手算对照梯度
# =============================================================================


def toy_roi_avg_pool(
    feat: torch.Tensor,
    rois: list[tuple[int, int, int, int]],
) -> torch.Tensor:
    """极简 2×2 RoI 平均池化（教学用）。

    feat: [H, W] 单通道特征图
    rois: 每个 RoI = (y1, x1, y2, x2)，半开区间 [y1:y2, x1:x2]
    返回: [R, 2, 2] —— 每个 RoI 切成 2×2 bin，每 bin 做 mean
    """
    outs = []
    for y1, x1, y2, x2 in rois:
        h = y2 - y1
        w = x2 - x1
        # 每个 bin 的高/宽（不同 RoI 可以不同！）
        bh, bw = h // 2, w // 2
        bins = []
        for i in range(2):
            row = []
            for j in range(2):
                ys, ye = y1 + i * bh, y1 + (i + 1) * bh
                xs, xe = x1 + j * bw, x1 + (j + 1) * bw
                # 关键：切片 + mean 都是可微运算（对参与平均的像素，梯度均分）
                row.append(feat[ys:ye, xs:xe].mean())
            bins.append(torch.stack(row))
        outs.append(torch.stack(bins))
    return torch.stack(outs)  # [R, 2, 2]


def demo_roi_pool_grad() -> None:
    print("\n" + "=" * 60)
    print("3. 循环 RoI 池化：梯度如何流回特征图")
    print("=" * 60)

    # 4×4 特征图
    feat = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    feat = feat.clone().requires_grad_(True)
    print(f"  feat =\n{feat.detach()}")

    # 两个不同大小的 RoI（bin 像素数因此不同）
    # RoI0: 整图 4×4 → 每 bin 2×2
    # RoI1: 左上 2×2 → 每 bin 1×1
    rois = [
        (0, 0, 4, 4),
        (0, 0, 2, 2),
    ]
    pooled = toy_roi_avg_pool(feat, rois)
    print(f"  pooled.shape = {tuple(pooled.shape)}  # 两个 RoI 输出形状相同！")
    print(f"  pooled[0] (大框) =\n{pooled[0].detach()}")
    print(f"  pooled[1] (小框) =\n{pooled[1].detach()}")

    # 假装下游 loss = 所有池化值之和
    loss = pooled.sum()
    loss.backward()

    print(f"  feat.grad =\n{feat.grad}")
    print(
        """
  读梯度表（以左上角像素 feat[0,0]=0 为例）：
    - 它属于 RoI0 的 bin(0,0)，该 bin 有 4 个像素做 mean → 贡献 1/4
    - 它属于 RoI1 的 bin(0,0)，该 bin 有 1 个像素做 mean → 贡献 1/1
    - 总梯度 = 0.25 + 1.0 = 1.25
  右下角 feat[3,3] 只被 RoI0 的 bin(1,1) 用到 → 梯度 0.25
  这就是**循环 + 共享特征图**时 autograd 自动做的事：按使用次数/权重累加。
"""
    )


# =============================================================================
# 4. 双线性采样也可微（可变形 RoI / 可变形卷积的关键）
# =============================================================================


def demo_bilinear_grad() -> None:
    print("\n" + "=" * 60)
    print("4. 双线性采样：对 feat 和 坐标 都可反传")
    print("=" * 60)

    # 1×1×2×2 特征
    feat = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    # 在 (0.5, 0.5) 采一点（像素坐标，对齐 align_corners 时需映射到 [-1,1]）
    # grid_sample: -1=左/上, +1=右/下；对 2×2、align_corners=True：
    #   像素 (0,0)→(-1,-1), (1,1)→(1,1), 中心 (0.5,0.5)→(0,0)
    grid = torch.tensor([[[[0.0, 0.0]]]], requires_grad=True)  # [1,1,1,2] = (x,y) in [-1,1]

    sampled = F.grid_sample(
        feat, grid, mode="bilinear", align_corners=True, padding_mode="zeros"
    )
    # 中心双线性：四周各 0.25 → (1+2+3+4)/4 = 2.5
    print(f"  sampled = {sampled.item():.4f}  (期望 2.5)")

    sampled.backward()
    print(f"  feat.grad =\n{feat.grad}")
    print("  → 四个角各分到 0.25（双线性权重）")
    print(f"  grid.grad = {grid.grad}")  # 对采样坐标也可导 → 偏移才能学


# =============================================================================
# 5. 自己写一个 Function：看清 forward / backward 分工
# =============================================================================


class SquareFunction(torch.autograd.Function):
    """演示自定义算子：forward 存 ctx，backward 用 ctx 算梯度。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        # d(x^2)/dx = 2x，链式法则再乘上游 grad_output
        return grad_output * 2 * x


def demo_custom_function() -> None:
    print("\n" + "=" * 60)
    print("5. 自定义 Function = 你告诉 PyTorch**怎么反传**")
    print("=" * 60)

    x = torch.tensor(3.0, requires_grad=True)
    y = SquareFunction.apply(x)
    y.backward()
    print(f"  x={x.item()}, y={y.item()}, x.grad={x.grad.item()}  (期望 6)")

    print(
        """
  内置的 + * mean slice grid_sample ... 都自带这种 forward/backward。
  RoI 池化的 CUDA 实现也一样：forward 记下来采了哪些像素，
  backward 把上游梯度按权重 scatter 回特征图。
  Python for 循环版只是把这些小算子串起来，图更长，但规则相同。
"""
    )


# =============================================================================
# 6. 一张**心智图**
# =============================================================================


def print_mental_model() -> None:
    print("\n" + "=" * 60)
    print("6. 心智模型（请对着看）")
    print("=" * 60)
    print(
        r"""
  forward（从左到右建图）:

    feat (requires_grad)
       │
       ├─► RoI0: slice/mean/bin → pooled0 ─┐
       ├─► RoI1: slice/mean/bin → pooled1 ─┼─► loss
       └─► RoI2: ...            → pooled2 ─┘

  backward（从右到左沿边推梯度）:

    dL/d(pooled*) ──► 每个 bin 的 mean 把梯度均分给 bin 内像素
                   ──► 若同一像素被多个 RoI/bin 用到，梯度相加
                   ──► 最终得到 feat.grad

  因此：
    • **循环**只是 Python 写法，图里是一串可微节点；
    • 不同 RoI 的 bin 像素数可以不同 —— 只影响 mean 的分母，
      不影响**输出仍是固定 2×2 / 7×7**；
    • 梯度传播靠的是每个算子的 backward，不是靠循环本身。
"""
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_basic_graph()
    demo_loop_still_differentiable()
    demo_grad_accumulate()
    demo_roi_pool_grad()
    demo_bilinear_grad()
    demo_custom_function()
    print_mental_model()
    print("全部 demo 跑完。建议打断点看 y.grad_fn / feat.grad。")
