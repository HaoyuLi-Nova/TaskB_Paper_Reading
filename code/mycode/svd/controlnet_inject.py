"""ControlNet 把控制信号注入 SVD 去噪 U-Net（缩小、可运行）。

SVD 的去噪网络是时空 U-Net。编码器逐级把空间缩小，每级特征留作跳跃连接；
解码器把跳跃连接沿通道拼回去，预测噪声。本文件只保留这条骨架，尺寸缩小。

控制图不进 SVD 原有的两条条件通路（首帧潜变量的通道拼接、CLIP 的交叉注意力）。
ControlNet 另复制一份编码器：

1. hint 编码器把逐帧控制图收到与 conv_in 同形状，最后一层是零卷积 Z1。
2. 可训练副本跑完每一级后，再用零卷积 Z2 把该级特征变成残差。
3. 残差加到冻结 U-Net 的同级跳跃连接和中间块上。解码器读到的是 skip + 残差。

论文里一个块写成
    y = F(x; Θ) + Z2( F(x + Z1(c); Θ_c) ; Θ_z2 )
第 0 步 Z1、Z2 输出都是 0，所以 y 与原 U-Net 相同。源码把 Z1 放在 hint 编码器
的最后一层、只加一次；更深的块因为输入已经带上这个和，所以看得到控制信号。
Z2 则每一级跳连一个，对应官方 ControlNet 的 hs.pop() + control.pop()。

diffusers 里同一加法的参数名是：
    down_block_additional_residuals  加到各层 skip
    mid_block_additional_residual    加到中间块

真实 SVD-img2vid 大约是：像素 576×1024、14 或 25 帧，VAE×8 后潜变量
[B, F, 4, 72, 128]，与复制的首帧潜变量拼成 [B, F, 8, 72, 128]；
block_out_channels = (320, 640, 1280, 1280)，每级 2 个 ResBlock，跳连多于 3 条。
这里 B=2, F=4, 潜空间 8×8, 通道 (16, 32, 64)，每级 1 条跳连。加法与真实模型相同。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

B, F_FRAMES = 2, 4
H_LAT, W_LAT = 8, 8
C_NOISE, C_IMAGE = 4, 4
C_IN = C_NOISE + C_IMAGE  # 8，噪声潜变量与首帧潜变量沿通道拼接
C0, C1, C2 = 16, 32, 64  # 三级通道。真实 SVD 是 320, 640, 1280
PIX_SCALE = 8  # 控制图相对潜空间的空间倍数，对应 VAE 的 ×8


class ZeroConv(nn.Module):
    """1×1 卷积，权重与偏置初始为 0。论文中的零卷积 Z。

    x [B, F, C_in, H, W] → [B, F, C_out, H, W]
    B=batch, F=帧, C_in/C_out=通道, H/W=该层空间。空间尺寸不变。
    """

    def __init__(self, cin, cout=None):
        super().__init__()
        cout = cin if cout is None else cout
        self.conv = nn.Conv2d(cin, cout, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)
        # [B, F, C_in, H, W] → [B*F, C_in, H, W]
        # B=batch, F=帧, C_in=输入通道, H/W=该层高宽。空间卷积逐帧做，与 SVD 把 B、F 摊平一致
        y = self.conv(x)
        # [B*F, C_in, H, W] → [B*F, C_out, H, W]
        # 初始 W=0 且 bias=0，故 y 恒为 0；∂L/∂W = ∂L/∂y · x，梯度可以不为 0
        y = y.reshape(b, f, -1, h, w)
        # → [B, F, C_out, H, W]
        return y


class SpatialRes(nn.Module):
    """逐帧 3×3 卷积残块。真实 SVD 此处是 GroupNorm-SiLU-Conv，并加有 timestep embedding。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(cin, cout, kernel_size=1) if cin != cout else None

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)
        # [B, F, C_in, H, W] → [B*F, C_in, H, W]
        # B=batch, F=帧, C_in=输入通道, H/W=该层空间
        h_in = x
        x = F.silu(x)
        # [B*F, C_in, H, W] 形状不变
        y = self.conv(x)
        # [B*F, C_in, H, W] → [B*F, C_out, H, W]
        if self.proj is None:
            y = y + h_in
            # [B*F, C_out, H, W] 残差，通道未变
        else:
            skip = self.proj(h_in)
            # [B*F, C_in, H, W] → [B*F, C_out, H, W]  1×1 把残差分支对齐到输出通道
            y = y + skip
            # [B*F, C_out, H, W] 两条分支相加
        y = y.reshape(b, f, -1, h, w)
        # → [B, F, C_out, H, W]
        return y


class TemporalMix(nn.Module):
    """沿帧混合。真实 SVD 在空间块之后做时间注意力；这里用核长 3 的 Conv1d，帧轴位置相同。"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.permute(0, 3, 4, 2, 1)
        # [B, F, C, H, W] → [B, H, W, C, F]
        # 把帧轴放到最后，便于每个空间位置单独做一维卷积
        x = x.reshape(b * h * w, c, f)
        # → [B*H*W, C, F]  B*H*W=空间位置个数, C=通道, F=帧
        y = self.conv(x)
        # [B*H*W, C, F] 形状不变；核长 3，混合相邻帧
        y = y.reshape(b, h, w, c, f)
        # → [B, H, W, C, F]
        y = y.permute(0, 4, 3, 1, 2)
        # → [B, F, C, H, W]
        return y


class STBlock(nn.Module):
    """一个时空块：先空间残块，再沿帧混合。cin → cout，空间与帧数不变。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.spatial = SpatialRes(cin, cout)
        self.temporal = TemporalMix(cout)

    def forward(self, x):
        x = self.spatial(x)
        # [B, F, C_in, H, W] → [B, F, C_out, H, W]
        x = self.temporal(x)
        # [B, F, C_out, H, W] 形状不变，帧之间已混合
        return x


class Down2x(nn.Module):
    """空间 /2，帧数与通道不变。"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)
        # [B, F, C, H, W] → [B*F, C, H, W]
        y = self.conv(x)
        # [B*F, C, H, W] → [B*F, C, H/2, W/2]
        h2, w2 = y.shape[-2], y.shape[-1]
        y = y.reshape(b, f, c, h2, w2)
        # → [B, F, C, H/2, W/2]  帧数不变
        return y


class Up2x(nn.Module):
    """空间 ×2，并把拼接后的通道压到 cout。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)
        # [B, F, C_in, H, W] → [B*F, C_in, H, W]
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # [B*F, C_in, H, W] → [B*F, C_in, 2H, 2W]
        y = self.conv(x)
        # → [B*F, C_out, 2H, 2W]
        y = y.reshape(b, f, -1, h * 2, w * 2)
        # → [B, F, C_out, 2H, 2W]
        return y


class Conv2dBF(nn.Module):
    """把 [B, F, C, H, W] 摊成 [B*F, C, H, W] 做二维卷积，再还原。空间尺寸不变。"""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1)

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = x.reshape(b * f, c, h, w)
        # [B, F, C_in, H, W] → [B*F, C_in, H, W]
        y = self.conv(x)
        # → [B*F, C_out, H, W]
        y = y.reshape(b, f, -1, h, w)
        # → [B, F, C_out, H, W]
        return y


class HintEncoder(nn.Module):
    """像素控制图 → 与 U-Net conv_in 输出同形状。最后一层是零卷积，即论文的 Z1。

    cond [B, F, 3, H_pix, W_pix] → [B, F, C0, H_lat, W_lat]
    H_pix = PIX_SCALE * H_lat。三层 stride=2 把空间收到 1/8。
    """

    def __init__(self, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.zero = ZeroConv(64, cout)

    def forward(self, cond):
        b, f, c, h, w = cond.shape
        x = cond.reshape(b * f, c, h, w)
        # [B, F, 3, H_pix, W_pix] → [B*F, 3, H_pix, W_pix]
        # 3=控制图通道（姿态、深度、轨迹热图画成的图）
        x = self.conv1(x)
        # → [B*F, 16, H_pix/2, W_pix/2]
        x = F.silu(x)
        # 形状不变
        x = self.conv2(x)
        # → [B*F, 32, H_pix/4, W_pix/4]
        x = F.silu(x)
        # [B*F, 32, H_pix/4, W_pix/4] 形状不变
        x = self.conv3(x)
        # → [B*F, 64, H_lat, W_lat]  H_lat=H_pix/8，与 VAE 潜空间对齐
        x = F.silu(x)
        # [B*F, 64, H_lat, W_lat] 形状不变
        _, c3, h3, w3 = x.shape
        x = x.reshape(b, f, c3, h3, w3)
        # → [B, F, 64, H_lat, W_lat]
        y = self.zero(x)
        # → [B, F, C0, H_lat, W_lat]  初始为全 0
        return y


class SVDTinyUNet(nn.Module):
    """缩小的 SVD 去噪 U-Net。只含编码器跳连、中间块、解码器。

    输入 z [B, F, 8, H, W]：前 4 通道是噪声潜变量，后 4 通道是沿帧复制的首帧潜变量。
    输出 eps [B, F, 4, H, W]：预测的噪声，通道数回到 VAE 潜通道，不含图像条件那 4 维。
    """

    def __init__(self):
        super().__init__()
        self.conv_in = Conv2dBF(C_IN, C0)
        self.block0 = STBlock(C0, C0)
        self.down0 = Down2x(C0)
        self.block1 = STBlock(C0, C1)
        self.down1 = Down2x(C1)
        self.block2 = STBlock(C1, C2)
        self.mid = STBlock(C2, C2)
        self.up2 = Up2x(C2 + C2, C1)
        self.up1 = Up2x(C1 + C1, C0)
        self.out = Conv2dBF(C0 + C0, C_NOISE)

    def encode(self, z):
        """z [B, F, 8, H, W] → 三条跳连与中间特征。跳连从浅到深。"""
        h = self.conv_in(z)
        # [B, F, 8, H, W] → [B, F, C0, H, W]  8=噪声通道+首帧潜通道
        h = self.block0(h)
        # [B, F, C0, H, W] 形状不变
        skip0 = h
        h = self.down0(h)
        # → [B, F, C0, H/2, W/2]
        h = self.block1(h)
        # → [B, F, C1, H/2, W/2]
        skip1 = h
        h = self.down1(h)
        # → [B, F, C1, H/4, W/4]
        h = self.block2(h)
        # → [B, F, C2, H/4, W/4]
        skip2 = h
        mid = self.mid(h)
        # [B, F, C2, H/4, W/4] 形状不变
        return (skip0, skip1, skip2), mid

    def decode(self, mid, skips):
        """mid 与三条跳连（浅到深）→ 噪声预测 [B, F, 4, H, W]。

        调用前，跳连和 mid 上已经加过 ControlNet 残差。
        """
        skip0, skip1, skip2 = skips
        x = torch.cat([mid, skip2], dim=2)
        # [B, F, C2, H/4, W/4] 与同形状跳连，沿通道维拼接 → [B, F, 2*C2, H/4, W/4]
        x = self.up2(x)
        # → [B, F, C1, H/2, W/2]
        x = torch.cat([x, skip1], dim=2)
        # [B, F, C1, H/2, W/2] 与跳连拼接 → [B, F, 2*C1, H/2, W/2]
        x = self.up1(x)
        # → [B, F, C0, H, W]
        x = torch.cat([x, skip0], dim=2)
        # → [B, F, 2*C0, H, W]
        eps = self.out(x)
        # → [B, F, 4, H, W]  4=预测噪声的潜通道
        return eps

    def forward(self, z):
        skips, mid = self.encode(z)
        return self.decode(mid, skips)


class SVDControlNet(nn.Module):
    """可训练编码器副本 + hint + 每级一个零卷积。

    与 SVDTinyUNet.encode 同结构。forward 不跑解码器，只返回要加到跳连和中间块上的残差。
    """

    def __init__(self):
        super().__init__()
        self.hint = HintEncoder(C0)
        self.conv_in = Conv2dBF(C_IN, C0)
        self.block0 = STBlock(C0, C0)
        self.down0 = Down2x(C0)
        self.block1 = STBlock(C0, C1)
        self.down1 = Down2x(C1)
        self.block2 = STBlock(C1, C2)
        self.mid = STBlock(C2, C2)
        self.zero0 = ZeroConv(C0)
        self.zero1 = ZeroConv(C1)
        self.zero2 = ZeroConv(C2)
        self.zero_mid = ZeroConv(C2)

    def clone_encoder_from(self, unet):
        """复制冻结 U-Net 的编码器权重。论文中的可训练副本 Θ_c，起步时就是预训练骨干。"""
        names = ("conv_in", "block0", "down0", "block1", "down1", "block2", "mid")
        for name in names:
            getattr(self, name).load_state_dict(getattr(unet, name).state_dict())

    def forward(self, z, cond):
        """z [B, F, 8, H, W], cond [B, F, 3, H_pix, W_pix]
        → 三条残差（与跳连同形状）和中间残差。第 0 步四者都是 0。
        """
        h = self.conv_in(z)
        # [B, F, 8, H, W] → [B, F, C0, H, W]
        hint = self.hint(cond)
        # cond [B, F, 3, H_pix, W_pix] → [B, F, C0, H, W]，初始全 0
        h = h + hint
        # 两路同形状相加。对应源码在 input_blocks[0] 之后加上 guided_hint
        h = self.block0(h)
        # [B, F, C0, H, W] 形状不变
        c0 = self.zero0(h)
        # → [B, F, C0, H, W]  加到最浅跳连上的残差，初始为 0
        h = self.down0(h)
        # → [B, F, C0, H/2, W/2]
        h = self.block1(h)
        # → [B, F, C1, H/2, W/2]
        c1 = self.zero1(h)
        # → [B, F, C1, H/2, W/2]
        h = self.down1(h)
        # → [B, F, C1, H/4, W/4]
        h = self.block2(h)
        # → [B, F, C2, H/4, W/4]
        c2 = self.zero2(h)
        # → [B, F, C2, H/4, W/4]
        h = self.mid(h)
        # [B, F, C2, H/4, W/4] 形状不变
        c_mid = self.zero_mid(h)
        # → [B, F, C2, H/4, W/4]  加到中间块上的残差
        return (c0, c1, c2), c_mid


def inject(unet, control, z, cond):
    """冻结编码器不算梯度；残差加到跳连和中间块上，再交给解码器。

    这就是「把控制信号注入去噪 U-Net」的那一行加法。
    返回噪声预测 [B, F, 4, H, W]。
    """
    with torch.no_grad():
        skips, mid = unet.encode(z)
    skip0, skip1, skip2 = (s.detach() for s in skips)
    mid = mid.detach()
    (c0, c1, c2), c_mid = control(z, cond)

    # diffusers: down_block_res_sample + down_block_additional_residual
    skip0 = skip0 + c0
    # [B, F, C0, H, W] + 同形状残差 → [B, F, C0, H, W]
    skip1 = skip1 + c1
    # [B, F, C1, H/2, W/2]
    skip2 = skip2 + c2
    # [B, F, C2, H/4, W/4]
    # diffusers: sample + mid_block_additional_residual
    mid = mid + c_mid
    # [B, F, C2, H/4, W/4]

    eps = unet.decode(mid, (skip0, skip1, skip2))
    # → [B, F, 4, H, W]
    return eps, (c0, c1, c2, c_mid)


def grad_abs_mean(module):
    total = 0.0
    count = 0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().abs().sum().item()
        count += param.numel()
    if count == 0:
        return 0.0
    return total / count


def max_abs_diff(module, snapshot):
    diff = 0.0
    for name, param in module.state_dict().items():
        diff = max(diff, (param - snapshot[name]).abs().max().item())
    return diff


def main():
    torch.manual_seed(0)
    unet = SVDTinyUNet()
    control = SVDControlNet()
    control.clone_encoder_from(unet)
    unet.requires_grad_(False)

    z_noise = torch.randn(B, F_FRAMES, C_NOISE, H_LAT, W_LAT)
    # [B, F, 4, H, W]  4=VAE 潜通道, H/W=潜空间高宽。这是正在去噪的 z_t
    first = torch.randn(B, 1, C_IMAGE, H_LAT, W_LAT)
    # [B, 1, 4, H, W]  首帧潜变量。1=条件帧
    image = first.expand(B, F_FRAMES, C_IMAGE, H_LAT, W_LAT)
    # [B, 1, 4, H, W] → [B, F, 4, H, W]  沿帧复制。1=条件帧, F=视频帧
    image = image.contiguous()
    # [B, F, 4, H, W] 形状不变。SVD 自己的图像条件，与 ControlNet 无关
    z = torch.cat([z_noise, image], dim=2)
    # → [B, F, 8, H, W]  8=4+4，U-Net 的输入

    cond = torch.randn(B, F_FRAMES, 3, H_LAT * PIX_SCALE, W_LAT * PIX_SCALE)
    # [B, F, 3, H_pix, W_pix]  逐帧控制图。H_pix=8*H
    cond_other = torch.randn_like(cond)

    with torch.no_grad():
        locked = unet(z)
        # [B, F, 4, H, W]  没有 ControlNet 时的噪声预测
        pred0, residuals = inject(unet, control, z, cond)
        pred0_other, _ = inject(unet, control, z, cond_other)
    c0, c1, c2, c_mid = residuals
    print("跳连残差形状:")
    print(f"  c0    {tuple(c0.shape)}  最浅，空间与潜变量相同")
    print(f"  c1    {tuple(c1.shape)}  空间 /2")
    print(f"  c2    {tuple(c2.shape)}  空间 /4")
    print(f"  c_mid {tuple(c_mid.shape)}  中间块")
    print(f"第 0 步残差最大绝对值: {max(t.abs().max().item() for t in residuals):.3e}")
    print(f"第 0 步预测与冻结 U-Net 的最大差: {(pred0 - locked).abs().max().item():.3e}")
    print(f"第 0 步换一张控制图后的预测差: {(pred0 - pred0_other).abs().max().item():.3e}")

    target = torch.randn_like(locked)
    # [B, F, 4, H, W]  假想的监督噪声。只为让损失对残差产生梯度
    opt = torch.optim.AdamW(control.parameters(), lr=1e-2)
    unet_before = {k: v.detach().clone() for k, v in unet.state_dict().items()}
    zero_before = control.zero0.conv.weight.detach().clone()

    print("步  残差均值   Z2跳连零卷积  副本编码器   Z1 hint零卷积  hint主干")
    for step in range(1, 6):
        opt.zero_grad(set_to_none=True)
        pred, residuals = inject(unet, control, z, cond)
        loss = F.mse_loss(pred, target)
        # pred、target 均为 [B, F, 4, H, W]，4=噪声潜通道
        loss.backward()
        g_z2 = grad_abs_mean(control.zero0)
        g_copy = grad_abs_mean(control.block1)
        g_z1 = grad_abs_mean(control.hint.zero)
        g_hint = grad_abs_mean(control.hint.conv1)
        res_abs = [r.detach().abs().mean() for r in residuals]
        res_mean = torch.stack(res_abs).mean().item()
        # res_abs 含 4 个标量，对应 c0,c1,c2,c_mid
        print(
            f"{step:2d}  {res_mean:.3e}  {g_z2:.3e}      {g_copy:.3e}   {g_z1:.3e}     {g_hint:.3e}"
        )
        opt.step()

    with torch.no_grad():
        pred_a, _ = inject(unet, control, z, cond)
        pred_b, _ = inject(unet, control, z, cond_other)
    print(f"训练后，两张不同控制图的预测差: {(pred_a - pred_b).abs().max().item():.3e}")
    print(f"冻结 U-Net 权重最大变化: {max_abs_diff(unet, unet_before):.3e}")
    print(f"最浅零卷积权重最大变化: {(control.zero0.conv.weight - zero_before).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
