"""
可变形卷积 / 可变形 RoI 池化 —— 对照 ICCV 2017 论文的教学实现
================================================================

对应译本：translations/foundations/deformable_conv/paper_zh.md
原文 TeX：arxiv/foundations/deformable_conv/extracted/egpaper_for_arxiv.tex

本文件目标
----------
用**能逐步打印、逐步调试**的纯 PyTorch 代码，把论文三个模块讲清楚：

  1. 可变形卷积（图 2）
  2. 可变形 RoI 池化（图 3，Faster R-CNN 风格）
  3. 可变形 PS RoI 池化（图 4，R-FCN 风格）

阅读建议
--------
- 先看每个模块顶部的**在说什么**，再看代码里标了 STEP 的函数。
- 运行本文件：``python scripts/deformable_conv_tutorial.py``
  demo 会打印张量形状，并验证**零偏移 ≈ 普通卷积**。

说明
----
- 这是教学参考实现（循环多、偏慢），不是工程最优实现。
- 工程常用：torchvision.ops.DeformConv2d / mmcv CUDA 算子。
- DCNv1（本文）只有 offset；DCNv2 的 modulation 本文件不实现。

依赖：pip install torch
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 公共工具
# =============================================================================


def bilinear_sample(
    feat: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """在特征图上按像素坐标 (x, y) 做双线性采样（论文 bilinear interpolation）。

    Args:
        feat: [B, C, H, W]
        x, y: [B, N]  像素坐标（可为分数）；原点在左上，x 向右、y 向下

    Returns:
        [B, C, N]
    """
    b, _c, h, w = feat.shape
    # grid_sample 要求坐标落在 [-1, 1]
    gx = 2.0 * x / max(w - 1, 1) - 1.0
    gy = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(b, -1, 1, 2)  # [B, N, 1, 2]
    out = F.grid_sample(
        feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # [B, C, N, 1]
    return out.squeeze(-1)


def sample_one_point(
    feat_map: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """从单张图 feat_map [C, H, W] 的 (x, y) 处采一个向量，返回 [C]。"""
    sampled = bilinear_sample(
        feat_map.unsqueeze(0),  # [1, C, H, W]
        x.reshape(1, 1),
        y.reshape(1, 1),
    )
    return sampled[0, :, 0]


@dataclass
class RoiBox:
    """一个 RoI 在特征图像素坐标系下的几何信息。"""

    batch_idx: int
    x1: torch.Tensor
    y1: torch.Tensor
    x2: torch.Tensor
    y2: torch.Tensor

    @classmethod
    def from_row(cls, roi: torch.Tensor) -> "RoiBox":
        # roi: (batch_idx, x1, y1, x2, y2)
        return cls(
            batch_idx=int(roi[0].item()),
            x1=roi[1],
            y1=roi[2],
            x2=roi[3],
            y2=roi[4],
        )

    @property
    def width(self) -> torch.Tensor:
        return (self.x2 - self.x1).clamp(min=1e-3)

    @property
    def height(self) -> torch.Tensor:
        return (self.y2 - self.y1).clamp(min=1e-3)

    def bin_center(self, i: int, j: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """第 (i, j) 个 bin 的规则网格中心（行 i、列 j，0-based）。"""
        cx = self.x1 + (j + 0.5) * self.width / k
        cy = self.y1 + (i + 0.5) * self.height / k
        return cx, cy


# =============================================================================
# 1. 可变形卷积（Deformable Convolution）—— 论文图 2
# =============================================================================
#
# 在说什么
# --------
# 标准 3×3 卷积：在固定 9 个格子上采样，再与权重加权求和。
# 可变形卷积：每个格子再加一个学出来的偏移 Δp，采样点可以**拧歪**。
#
#   标准:  y(p0) = Σ_n  w(pn) · x(p0 + pn)
#   可变形: y(p0) = Σ_n  w(pn) · x(p0 + pn + Δpn)
#
# 一张图对应三个角色：
#   • weight          —— 卷积核权重 w（和普通 Conv2d 一样）
#   • offset_conv(x)  —— 额外卷积，输出 2N 通道（N=核点数；每点一对 dx,dy）
#   • 采样 + 加权求和 —— 实现公式本身
#
# 通道排列（本文件约定，交错）：
#   offset[:, 0]=dx0, [:,1]=dy0, [:,2]=dx1, [:,3]=dy1, ...


class DeformConv2dPack(nn.Module):
    """带偏移分支的可变形卷积（输入/输出形状与普通 Conv2d 相同，可直接替换）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.num_kernel_points = kernel_size * kernel_size  # N
        # 主分支权重：对应公式里的 w(pn)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        # 偏移分支：与当前层同核大小 / 同 dilation；零初始化 → 训练初期 ≈ 普通卷积
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * self.num_kernel_points,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)  # [B, 2N, H_out, W_out]
        return deform_conv2d_ref(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


def deform_conv2d_ref(
    x: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """可变形卷积前向（教学版：按**每个核点**逐步采样，再做一次矩阵乘）。

    Args:
        x:      [B, C_in, H, W]
        offset: [B, 2N, H_out, W_out]
        weight: [C_out, C_in, kH, kW]
    """
    b, c_in, h, w = x.shape
    c_out, _, k_h, k_w = weight.shape
    n = k_h * k_w

    # ----- STEP 1: 输出特征图空间尺寸（与普通卷积公式相同）-----
    h_out = (h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    w_out = (w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1

    # ----- STEP 2: 规则核点相对位移 pn（尚未加学到的 Δp）-----
    # 对输出位置 (oh, ow)，第 (kh, kw) 个核点的输入坐标：
    #   ih = oh*stride - padding + kh*dilation + Δy
    #   iw = ow*stride - padding + kw*dilation + Δx
    kh, kw = torch.meshgrid(
        torch.arange(k_h, device=x.device, dtype=x.dtype),
        torch.arange(k_w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    pn_y = (kh * dilation).reshape(-1)  # [N]
    pn_x = (kw * dilation).reshape(-1)

    # ----- STEP 3: 每个输出位置的锚点（左上角对应的输入坐标）-----
    out_y, out_x = torch.meshgrid(
        torch.arange(h_out, device=x.device, dtype=x.dtype),
        torch.arange(w_out, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    base_y = out_y * stride - padding  # [H_out, W_out]
    base_x = out_x * stride - padding

    # ----- STEP 4: 对每个核点 n，在 p0+pn+Δpn 处采样 → 拼成 im2col -----
    # cols[n] 形状 [B, C_in, H_out*W_out]
    cols = []
    for n_idx in range(n):
        dx = offset[:, 2 * n_idx]  # [B, H_out, W_out]
        dy = offset[:, 2 * n_idx + 1]

        sample_x = base_x[None] + pn_x[n_idx] + dx
        sample_y = base_y[None] + pn_y[n_idx] + dy

        sampled = bilinear_sample(
            x,
            sample_x.reshape(b, -1),
            sample_y.reshape(b, -1),
        )
        cols.append(sampled)

    # [B, C_in * N, H_out*W_out] —— 类似 im2col
    col = torch.stack(cols, dim=2).reshape(b, c_in * n, h_out * w_out)

    # ----- STEP 5: 与权重做矩阵乘（等价于 Σ w(pn)·采样值）-----
    w_mat = weight.reshape(c_out, -1)  # [C_out, C_in*N]
    out = torch.einsum("oc,bcn->bon", w_mat, col)
    out = out.view(b, c_out, h_out, w_out)

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


# =============================================================================
# 2. 可变形 RoI 池化（Faster R-CNN 风格）—— 论文图 3
# =============================================================================
#
# 在说什么
# --------
# 标准 RoI 池化：把框切成 k×k 个 bin，每 bin 取一个代表值（这里用 bin 中心采样示意）。
# 可变形：每个 bin 中心再平移 Δp_ij。
#
# 偏移怎么来（图 3 三步）：
#   1) 先做一次普通 RoI 池化 → 固定长度特征
#   2) FC 预测**归一化偏移**Δp̂（与框大小无关，便于学习）
#   3) 还原真实偏移：Δp = γ · Δp̂ ⊙ (w, h)，再按偏移采样


class DeformRoIPoolPack(nn.Module):
    """图 3 完整模块：普通池化 → FC 预测归一化偏移 → 带偏移再池化。"""

    def __init__(self, channels: int, pooled_size: int = 7, gamma: float = 0.1):
        super().__init__()
        self.k = pooled_size
        self.gamma = gamma
        # 输出 2*k*k：每个 bin 一对 (dx̂, dŷ)
        self.offset_fc = nn.Linear(
            channels * pooled_size * pooled_size,
            2 * pooled_size * pooled_size,
        )
        nn.init.zeros_(self.offset_fc.weight)
        nn.init.zeros_(self.offset_fc.bias)

    def forward(self, feat: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W]
            rois: [R, 5]，每行 (batch_idx, x1, y1, x2, y2)，特征图像素坐标
        Returns:
            [R, C, k, k]
        """
        # STEP 1: 普通 RoI 池化
        pooled = roi_pool_at_bin_centers(feat, rois, self.k)  # [R, C, k, k]

        # STEP 2: FC → 归一化偏移 Δp̂，形状 [R, 2, k, k]
        num_rois = pooled.shape[0]
        delta_hat = self.offset_fc(pooled.flatten(1))
        delta_hat = delta_hat.view(num_rois, 2, self.k, self.k)

        # STEP 3: 还原真实偏移并采样
        return deform_roi_pool_ref(feat, rois, delta_hat, self.k, self.gamma)


def roi_pool_at_bin_centers(
    feat: torch.Tensor,
    rois: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """极简 RoI 池化：每个 bin 只在中心采一点（示意用；真实实现会在 bin 内多样本）。"""
    outputs = []
    for roi_row in rois:
        box = RoiBox.from_row(roi_row)
        feat_map = feat[box.batch_idx]  # [C, H, W]

        # grid[i, j] = 该 bin 中心采到的 [C] 向量
        grid = []
        for i in range(k):
            row = []
            for j in range(k):
                cx, cy = box.bin_center(i, j, k)
                row.append(sample_one_point(feat_map, cx, cy))
            grid.append(torch.stack(row, dim=1))  # [C, k]
        outputs.append(torch.stack(grid, dim=1))  # [C, k, k]
    return torch.stack(outputs, dim=0)


def deform_roi_pool_ref(
    feat: torch.Tensor,
    rois: torch.Tensor,
    delta_hat: torch.Tensor,
    k: int,
    gamma: float = 0.1,
) -> torch.Tensor:
    """给定归一化偏移，做可变形 RoI 池化。

    Args:
        delta_hat: [R, 2, k, k]，通道 0=dx̂，1=dŷ
    """
    outputs = []
    for r, roi_row in enumerate(rois):
        box = RoiBox.from_row(roi_row)
        feat_map = feat[box.batch_idx]

        grid = []
        for i in range(k):
            row = []
            for j in range(k):
                cx, cy = box.bin_center(i, j, k)
                # Δp = γ * Δp̂ ⊙ (w, h)
                dx = gamma * delta_hat[r, 0, i, j] * box.width
                dy = gamma * delta_hat[r, 1, i, j] * box.height
                row.append(sample_one_point(feat_map, cx + dx, cy + dy))
            grid.append(torch.stack(row, dim=1))
        outputs.append(torch.stack(grid, dim=1))
    return torch.stack(outputs, dim=0)


# =============================================================================
# 3. 可变形 PS RoI 池化（R-FCN 风格）—— 论文图 4
# =============================================================================
#
# 在说什么
# --------
# 普通 RoI：所有 bin 从**同一张**特征图 x 上取。
# PS RoI：第 (i,j) 个 bin 只从**专属于它**的得分图 x_{i,j} 上取。
#
# 通道布局（整图一次算完，空间尺寸仍是 H×W）：
#
#   score maps  通道数 = k² · (C+1)
#     按 bin 分组，每组 (C+1) 张：
#       组 0 (bin 左上) → 类 0, 类 1, ..., 类 C
#       组 1            → 类 0, 类 1, ..., 类 C
#       ...
#
#   offset fields 通道数 = 2 · k² · (C+1)
#     同样按 bin 分组，每组 2·(C+1)：
#       对每个类存 (dx̂, dŷ)
#
# 图 4 的两条支路：
#   上：conv → offset fields →（对每个 RoI/类）读出 k×k 个归一化偏移
#   下：conv → score maps    → 用偏移在对应 x_{i,j} 上采样 → 得 k×k 得分
#   最后：对 k×k 取平均，得到该 RoI、该类的一个标量分数


def ps_score_channel(bin_id: int, cls: int, num_classes_plus_bg: int) -> int:
    """score maps 里：bin_id 组、类别 cls 对应的通道下标。"""
    return bin_id * num_classes_plus_bg + cls


def ps_offset_channels(
    bin_id: int, cls: int, num_classes_plus_bg: int
) -> Tuple[int, int]:
    """offset fields 里：bin_id 组、类别 cls 的 (dx 通道, dy 通道)。"""
    # 每组宽度 = 2 * (C+1)；组内按类交错存 (dx, dy)
    base = bin_id * (2 * num_classes_plus_bg) + cls * 2
    return base, base + 1


class DeformPSRoIPoolHead(nn.Module):
    """图 4：上支路学偏移，下支路出位置敏感得分，再按 RoI/类做可变形 PS 池化。"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        k: int = 3,
        gamma: float = 0.1,
    ):
        super().__init__()
        self.k = k
        self.num_classes_plus_bg = num_classes + 1  # C+1
        self.gamma = gamma

        # 下支路：位置敏感得分图
        self.score_conv = nn.Conv2d(
            in_channels, k * k * self.num_classes_plus_bg, kernel_size=1
        )
        # 上支路：全图 offset field（零初始化 → 起步无偏移）
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * k * k * self.num_classes_plus_bg, kernel_size=1
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, feat: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            [R, C+1] —— 每个 RoI、每个类一个标量（对 k×k bin 得分取平均）
        """
        score_maps = self.score_conv(feat)  # [B, k²(C+1), H, W]
        offset_fields = self.offset_conv(feat)  # [B, 2k²(C+1), H, W]

        roi_logits = []
        for roi_row in rois:
            roi_logits.append(
                self._pool_one_roi(score_maps, offset_fields, RoiBox.from_row(roi_row))
            )
        return torch.stack(roi_logits, dim=0)

    def _pool_one_roi(
        self,
        score_maps: torch.Tensor,
        offset_fields: torch.Tensor,
        box: RoiBox,
    ) -> torch.Tensor:
        """对单个 RoI：遍历每个类，做可变形 PS 池化，返回 [C+1]。"""
        class_scores = []
        for cls in range(self.num_classes_plus_bg):
            bin_scores = self._pool_one_class(score_maps, offset_fields, box, cls)
            # R-FCN：k×k 个位置敏感得分再平均（vote）成一个类分数
            class_scores.append(bin_scores.mean())
        return torch.stack(class_scores)

    def _pool_one_class(
        self,
        score_maps: torch.Tensor,
        offset_fields: torch.Tensor,
        box: RoiBox,
        cls: int,
    ) -> torch.Tensor:
        """对单个 RoI、单个类：得到 k×k 的位置敏感得分网格。"""
        k = self.k
        c1 = self.num_classes_plus_bg
        scores_b = score_maps[box.batch_idx]  # [k²(C+1), H, W]
        offsets_b = offset_fields[box.batch_idx]

        grid = []
        for i in range(k):
            row = []
            for j in range(k):
                bin_id = i * k + j
                cx, cy = box.bin_center(i, j, k)

                # ----- 上支路：在规则中心读出归一化偏移（示意版 PS 池化）-----
                dx_ch, dy_ch = ps_offset_channels(bin_id, cls, c1)
                dx_hat = sample_one_point(offsets_b[dx_ch : dx_ch + 1], cx, cy)[0]
                dy_hat = sample_one_point(offsets_b[dy_ch : dy_ch + 1], cx, cy)[0]

                # Δp = γ · Δp̂ ⊙ (w, h)
                sx = cx + self.gamma * dx_hat * box.width
                sy = cy + self.gamma * dy_hat * box.height

                # ----- 下支路：在偏移后的位置，从**该 bin 专属**得分图采样 -----
                score_ch = ps_score_channel(bin_id, cls, c1)
                val = sample_one_point(scores_b[score_ch : score_ch + 1], sx, sy)[0]
                row.append(val)
            grid.append(torch.stack(row))
        return torch.stack(grid)  # [k, k]


# =============================================================================
# Demo：形状检查 + **零偏移 ≈ 普通卷积**
# =============================================================================


def demo() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32, 32)

    print("=" * 64)
    print("1) 可变形卷积  Deformable Convolution")
    print("=" * 64)
    layer = DeformConv2dPack(16, 32, kernel_size=3, padding=1)
    y = layer(x)
    offset = layer.offset_conv(x)
    print(f"  输入 x          {tuple(x.shape)}     # [B, C_in, H, W]")
    print(f"  偏移 offset     {tuple(offset.shape)}  # [B, 2N, H, W]，3×3 时 2N=18")
    print(f"  输出 y          {tuple(y.shape)}     # [B, C_out, H, W]")

    with torch.no_grad():
        offset0 = torch.zeros_like(offset)
        y_def = deform_conv2d_ref(x, offset0, layer.weight, layer.bias, padding=1)
        y_std = F.conv2d(x, layer.weight, layer.bias, padding=1)
        diff = (y_def - y_std).abs().max().item()
        print(f"  零偏移 vs Conv2d  max|diff| = {diff:.6f}  （应接近 0）")

    print()
    print("=" * 64)
    print("2) 可变形 RoI 池化  Deformable RoI Pooling")
    print("=" * 64)
    pool = DeformRoIPoolPack(channels=16, pooled_size=3)
    rois = torch.tensor(
        [
            [0, 4.0, 4.0, 20.0, 20.0],
            [1, 8.0, 8.0, 28.0, 28.0],
        ]
    )
    z = pool(x, rois)
    print(f"  rois            {tuple(rois.shape)}        # [R, 5]=(batch,x1,y1,x2,y2)")
    print(f"  池化结果        {tuple(z.shape)}     # [R, C, k, k]")

    print()
    print("=" * 64)
    print("3) 可变形 PS RoI 池化  （通道含义）")
    print("=" * 64)
    k, num_classes = 3, 5
    c1 = num_classes + 1
    ps = DeformPSRoIPoolHead(in_channels=16, num_classes=num_classes, k=k)
    logits = ps(x, rois)

    print(f"  k={k}, C={num_classes}, C+1={c1}")
    print(f"  score  通道数   = k²·(C+1)     = {k * k * c1}")
    print(f"  offset 通道数   = 2·k²·(C+1)   = {2 * k * k * c1}")
    print(f"  类分数 logits   {tuple(logits.shape)}      # [R, C+1]")
    print()
    print("  通道索引例子（bin 左上 i=j=0，类 cls=2）：")
    print(f"    score 通道 = {ps_score_channel(0, 2, c1)}")
    dx_ch, dy_ch = ps_offset_channels(0, 2, c1)
    print(f"    offset dx/dy 通道 = {dx_ch}, {dy_ch}")


if __name__ == "__main__":
    demo()
