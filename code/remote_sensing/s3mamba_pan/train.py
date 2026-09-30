import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import os
import yaml
import time
import json
import math
import warnings
import argparse
from typing import Optional

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

import torch.nn.functional as F

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
        if size_average:
            return ssim_map.mean(), cs
        else:
            return ssim_map.mean(1).mean(1).mean(1), cs

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class MS_SSIM_Loss_Custom(nn.Module):
    def __init__(self, data_range=1.0, size_average=True, channel=8):
        super(MS_SSIM_Loss_Custom, self).__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.ssim_module = SSIM(window_size=11, size_average=size_average)
        self.weights = torch.FloatTensor([0.0448, 0.2856, 0.3001])
        self.weights = self.weights / self.weights.sum()

    def forward(self, img1, img2):
        if self.weights.device != img1.device:
            self.weights = self.weights.to(img1.device)
        msssim = []
        for i in range(3):
            ssim_val, _ = self.ssim_module(img1, img2)
            msssim.append(ssim_val)
            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))
        msssim = torch.stack(msssim)
        final_ms_ssim = torch.sum(msssim * self.weights)
        return 1.0 - final_ms_ssim

def get_sobel_kernels(device, channels):
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0)
    sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype=torch.float32).unsqueeze(0)
    sobel_x = sobel_x.repeat(channels, 1, 1, 1).to(device)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1).to(device)
    return sobel_x, sobel_y

def gradient_loss(pred, target, device):
    channels = pred.shape[1]
    sobel_x, sobel_y = get_sobel_kernels(device, channels)
    grad_pred_x = F.conv2d(pred, sobel_x, padding=1, groups=channels)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1, groups=channels)
    grad_target_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
    grad_target_y = F.conv2d(target, sobel_y, padding=1, groups=channels)
    return F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

def compute_sam(img1, img2):
    img1 = img1.reshape(img1.shape[0], -1).astype(np.float64)
    img2 = img2.reshape(img2.shape[0], -1).astype(np.float64)
    dot = np.sum(img1 * img2, axis=0)
    cos_val = np.clip(dot / (np.linalg.norm(img1, axis=0) * np.linalg.norm(img2, axis=0) + 1e-8), -1.0, 1.0)
    return np.mean(np.arccos(cos_val)) * 180.0 / np.pi

def compute_ergas(img_fused, img_ref, scale=4):
    C, H, W = img_fused.shape
    ergas_sum = 0.0
    for c in range(C):
        rmse = np.sqrt(np.mean((img_fused[c] - img_ref[c]) ** 2))
        mu_c = np.mean(img_ref[c])
        if mu_c > 1e-8: ergas_sum += (rmse / mu_c) ** 2
    return 100 / scale * np.sqrt(ergas_sum / C)

def compute_q4(img_fused, img_ref):
    C = img_fused.shape[0]
    q_vals = []
    for c in range(C):
        x, y = img_fused[c].flatten(), img_ref[c].flatten()
        cov = np.cov(x, y)[0, 1]
        mx, my = np.mean(x), np.mean(y)
        vx, vy = np.var(x), np.var(y)
        denom = (vx + vy) * (mx**2 + my**2)
        q_vals.append((4 * cov * mx * my) / denom if denom != 0 else 1.0)
    return np.mean(q_vals)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU()
        )
        for p in self.layers.parameters(): p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        p_rgb = pred.repeat(1,3,1,1) if pred.shape[1]==1 else pred[:, :3]
        t_rgb = target.repeat(1,3,1,1) if target.shape[1]==1 else target[:, :3]
        p_rgb = (p_rgb - self.mean) / self.std
        t_rgb = (t_rgb - self.mean) / self.std
        return F.l1_loss(self.layers(p_rgb), self.layers(t_rgb))

# --- FDFM: Frequency-Decoupled Fusion Module (DWT/IDWT) ---
class DWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('w_ll', torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32))
        self.register_buffer('w_lh', torch.tensor([[[[0.25, 0.25], [-0.25, -0.25]]]], dtype=torch.float32))
        self.register_buffer('w_hl', torch.tensor([[[[0.25, -0.25], [0.25, -0.25]]]], dtype=torch.float32))
        self.register_buffer('w_hh', torch.tensor([[[[0.25, -0.25], [-0.25, 0.25]]]], dtype=torch.float32))

    def forward(self, x):
        if x.shape[-1] % 2 != 0: x = F.pad(x, (0, 1, 0, 1), mode='reflect')
        C = x.shape[1]
        x_ll = F.conv2d(x, self.w_ll.expand(C, -1, -1, -1), stride=2, groups=C)
        x_lh = F.conv2d(x, self.w_lh.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hl = F.conv2d(x, self.w_hl.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hh = F.conv2d(x, self.w_hh.expand(C, -1, -1, -1), stride=2, groups=C)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

class IDWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('w_ll', torch.tensor([[[[1., 1.], [1., 1.]]]], dtype=torch.float32))
        self.register_buffer('w_lh', torch.tensor([[[[1., 1.], [-1., -1.]]]], dtype=torch.float32))
        self.register_buffer('w_hl', torch.tensor([[[[1., -1.], [1., -1.]]]], dtype=torch.float32))
        self.register_buffer('w_hh', torch.tensor([[[[1., -1.], [-1., 1.]]]], dtype=torch.float32))

    def forward(self, x):
        C = x.shape[1] // 4
        x_ll, x_lh, x_hl, x_hh = x[:, 0*C:1*C], x[:, 1*C:2*C], x[:, 2*C:3*C], x[:, 3*C:4*C]
        o_ll = F.conv_transpose2d(x_ll, self.w_ll.expand(C, -1, -1, -1), stride=2, groups=C)
        o_lh = F.conv_transpose2d(x_lh, self.w_lh.expand(C, -1, -1, -1), stride=2, groups=C)
        o_hl = F.conv_transpose2d(x_hl, self.w_hl.expand(C, -1, -1, -1), stride=2, groups=C)
        o_hh = F.conv_transpose2d(x_hh, self.w_hh.expand(C, -1, -1, -1), stride=2, groups=C)
        return o_ll + o_lh + o_hl + o_hh

# --- GSAM: Global Spectral Anchoring Module (ESAM) ---
class GSA_Encoder(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.E = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels // 2, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(hid_channels // 2, hid_channels, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(hid_channels, hid_channels * 4), nn.LeakyReLU(0.1, True),
            nn.Linear(hid_channels * 4, hid_channels)
        )
    def forward(self, x):
        return self.mlp(self.E(x).squeeze(-1).squeeze(-1))

# --- ADR: Adaptive Distribution Recalibration Module ---
class ADR_Module(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(3, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, eps=1e-6):
        b, c, _, _ = x.shape
        mean = x.view(b, c, -1).mean(2, keepdim=True)
        std = (x.view(b, c, -1).var(2, keepdim=True) + eps).sqrt()
        abs_mean = self.avg_pool(torch.abs(x)).view(b, c, 1)
        y = self.conv(torch.cat([abs_mean, mean, std], dim=2).transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        return x * self.sigmoid(y)

# --- S3M: Spectral-Structure Stream Mamba Block ---
class VimBlock(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.d_inner = dim * 2
        self.dt_rank = math.ceil(dim / 16)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 4, groups=self.d_inner, padding=3)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 32, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, 17, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x, z_anchor=None):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        if z_anchor is not None: x_flat = x_flat + z_anchor.unsqueeze(1)
        xz = self.in_proj(x_flat)
        x_in, z_gate = xz.chunk(2, dim=-1)
        x_conv = self.act(self.conv1d(x_in.transpose(1, 2))[:, :, :H*W])
        x_dbl = self.x_proj(x_conv.transpose(1, 2))
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, 16, 16], dim=-1)
        dt = self.dt_proj(dt).transpose(1, 2)
        if selective_scan_fn:
            y = selective_scan_fn(x_conv, dt, -torch.exp(self.A_log), B_ssm.transpose(1,2), C_ssm.transpose(1,2), self.D.float(), z=z_gate.transpose(1,2), delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        else: return x
        return self.out_proj(y.transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)

# --- STS: Spatial-Texture Stream (PID-SSM Block) ---
class PID_SSM_Block(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.d_inner = dim * 2
        self.dt_rank = math.ceil(dim / 16)
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 4, groups=self.d_inner, padding=3)
        self.act = nn.SiLU()
        self.k_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.d_inner), nn.SiLU())
        self.q_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.d_inner), nn.SiLU())
        self.dt_bc_proj = nn.Linear(self.d_inner*2, self.dt_rank + 32, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.gate_proj = nn.Linear(self.d_inner*2, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, 17, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x, K, Q):
        B, C, H, W = x.shape
        x_in = self.in_proj(x.flatten(2).transpose(1, 2)).transpose(1, 2)
        x_conv = self.act(self.conv1d(x_in)[:, :, :H*W])
        K_proj = self.k_proj(K.flatten(2).transpose(1, 2))
        Q_proj = self.q_proj(Q.flatten(2).transpose(1, 2))
        x_dbl = self.dt_bc_proj(torch.cat([x_conv.transpose(1, 2), K_proj], dim=-1))
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, 16, 16], dim=-1)
        z_gate = self.gate_proj(torch.cat([x_conv.transpose(1, 2), Q_proj], dim=-1)).transpose(1, 2)
        if selective_scan_fn:
            y = selective_scan_fn(x_conv, self.dt_proj(dt).transpose(1, 2), -torch.exp(self.A_log), B_ssm.transpose(1, 2), C_ssm.transpose(1, 2), self.D.float(), z=z_gate, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        else: return x
        return self.out_proj(y.transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)

# --- Dual-Stream Mamba Core ---
class DualStreamBlock(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.sss = VimBlock(dim, d_state)
        self.sts = PID_SSM_Block(dim*3, d_state)
    def forward(self, m_ll, m_high, z, K, Q):
        return m_ll + self.sss(m_ll, z), m_high + self.sts(m_high, K, Q)

# --- S3Mamba-Pan Architecture ---
class S3Mamba_Pan(nn.Module):
    def __init__(self, ms_channels=8, pan_channels=1, dim=64, num_layers=4):
        super().__init__()
        self.conv_ms = nn.Conv2d(ms_channels, dim, 3, 1, 1)
        self.conv_pan = nn.Conv2d(pan_channels, dim, 3, 1, 1)
        self.dwt, self.idwt = DWT_2D(), IDWT_2D()
        self.gsa = GSA_Encoder(ms_channels, dim)
        self.k_expand = nn.Conv2d(dim, dim*3, 1)
        self.blocks = nn.ModuleList([DualStreamBlock(dim) for _ in range(num_layers)])
        self.adr = ADR_Module(dim)
        self.tail = nn.Conv2d(dim, ms_channels, 3, 1, 1)

    def forward(self, ms, pan):
        z = self.gsa(ms)
        f_ms, f_pan = self.conv_ms(ms), self.conv_pan(pan)
        ms_dwt, pan_dwt = self.dwt(f_ms), self.dwt(f_pan)
        m_ll, m_high = ms_dwt[:, :f_ms.shape[1]], ms_dwt[:, f_ms.shape[1]:]
        p_ll, p_high = pan_dwt[:, :f_ms.shape[1]], pan_dwt[:, f_ms.shape[1]:]
        p_ll_ex = self.k_expand(p_ll)
        for block in self.blocks: m_ll, m_high = block(m_ll, m_high, z, p_ll_ex, p_high)
        f_fused = self.idwt(torch.cat([m_ll, m_high], 1))
        return self.tail(self.adr(f_fused)) + ms

class CombinedLoss(nn.Module):
    def __init__(self, l_s=0.1, l_g=0.1, c=8, dev='cpu'):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.msssim = MS_SSIM_Loss_Custom(channel=c)
        self.l_s, self.l_g, self.dev = l_s, l_g, dev
    def forward(self, pred, target):
        return self.l1(pred, target) + self.l_s * self.msssim(pred, target) + self.l_g * gradient_loss(pred, target, self.dev)

class PS_Dataset(Dataset):
    def __init__(self, p, g, l, b=10, aug=True):
        self.p, self.g, self.l, self.max_val, self.aug = p, g, l, float(2**b-1), aug
    def __getitem__(self, i):
        p, g, l = (self.p[i]/self.max_val).float(), (self.g[i]/self.max_val).float(), (self.l[i]/self.max_val).float()
        if self.aug:
            if torch.rand(1)<0.5: g, l, p = g.flip(-1), l.flip(-1), p.flip(-1)
            if torch.rand(1)<0.5: g, l, p = g.flip(-2), l.flip(-2), p.flip(-2)
        return p, g, l
    def __len__(self): return self.g.size(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    os.makedirs(config.get('weights_path', 'weights'), exist_ok=True)
    with h5py.File(config['train_data_path'], 'r') as f:
        train_loader = DataLoader(PS_Dataset(torch.from_numpy(f['pan'][:]), torch.from_numpy(f['gt'][:]), torch.from_numpy(f['lms'][:]), b=config.get('bit_depth', 10)), batch_size=config['batch_size'], shuffle=True, drop_last=True)
    model = S3Mamba_Pan(ms_channels=config['lms_channels'], dim=config['hid_channels']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['base_lr'])
    criterion = CombinedLoss(l_s=config.get('lambda_struct', 0.1), l_g=config.get('lambda_grad', 0.1), c=config['lms_channels'], dev=device)
    for epoch in range(1, config['epoch_num'] + 1):
        model.train()
        for pan, gt, lms in train_loader:
            optimizer.zero_grad()
            out = model(lms.to(device), pan.to(device))
            loss = criterion(out, gt.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch % 50 == 0: torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(config['weights_path'], f"s3mamba_epoch{epoch}.pth"))