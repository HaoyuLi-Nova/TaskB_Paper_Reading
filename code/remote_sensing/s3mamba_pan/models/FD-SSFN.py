import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

# --- FDFM: Frequency-Decoupled Fusion Module (DWT/IDWT) ---
class DWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        w_ll = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32)
        w_lh = torch.tensor([[[[0.25, 0.25], [-0.25, -0.25]]]], dtype=torch.float32)
        w_hl = torch.tensor([[[[0.25, -0.25], [0.25, -0.25]]]], dtype=torch.float32)
        w_hh = torch.tensor([[[[0.25, -0.25], [-0.25, 0.25]]]], dtype=torch.float32)
        self.register_buffer('w_ll', w_ll)
        self.register_buffer('w_lh', w_lh)
        self.register_buffer('w_hl', w_hl)
        self.register_buffer('w_hh', w_hh)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ll = F.conv2d(x, self.w_ll.expand(C, -1, -1, -1), stride=2, groups=C)
        x_lh = F.conv2d(x, self.w_lh.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hl = F.conv2d(x, self.w_hl.expand(C, -1, -1, -1), stride=2, groups=C)
        x_hh = F.conv2d(x, self.w_hh.expand(C, -1, -1, -1), stride=2, groups=C)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

class IDWT_2D(nn.Module):
    def __init__(self):
        super().__init__()
        w_ll = torch.tensor([[[[1., 1.], [1., 1.]]]], dtype=torch.float32)
        w_lh = torch.tensor([[[[1., 1.], [-1., -1.]]]], dtype=torch.float32)
        w_hl = torch.tensor([[[[1., -1.], [1., -1.]]]], dtype=torch.float32)
        w_hh = torch.tensor([[[[1., -1.], [-1., 1.]]]], dtype=torch.float32)
        self.register_buffer('w_ll', w_ll)
        self.register_buffer('w_lh', w_lh)
        self.register_buffer('w_hl', w_hl)
        self.register_buffer('w_hh', w_hh)

    def forward(self, x):
        B, C4, H, W = x.shape
        C = C4 // 4
        x_ll, x_lh, x_hl, x_hh = x[:, 0*C:1*C], x[:, 1*C:2*C], x[:, 2*C:3*C], x[:, 3*C:4*C]
        o_ll = F.conv_transpose2d(x_ll, self.w_ll.expand(C, -1, -1, -1), stride=2, groups=C)
        o_lh = F.conv_transpose2d(x_lh, self.w_lh.expand(C, -1, -1, -1), stride=2, groups=C)
        o_hl = F.conv_transpose2d(x_hl, self.w_hl.expand(C, -1, -1, -1), stride=2, groups=C)
        o_hh = F.conv_transpose2d(x_hh, self.w_hh.expand(C, -1, -1, -1), stride=2, groups=C)
        return o_ll + o_lh + o_hl + o_hh

# --- GSAM: Global Spectral Anchoring Module ---
class GSAM_Encoder(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super().__init__()
        self.E = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hid_channels // 2, hid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(hid_channels, hid_channels * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(hid_channels * 4, hid_channels)
        )

    def forward(self, x):
        z = self.mlp(self.E(x).squeeze(-1).squeeze(-1))
        return z

# --- ADR: Adaptive Distribution Recalibration Module ---
class ADR_Module(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(max(1, channels), 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(3, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, eps=1e-6):
        b, c, h, w = x.shape
        mean = x.view(b, c, -1).mean(2, keepdim=True)
        std = (x.view(b, c, -1).var(2, keepdim=True) + eps).sqrt()
        abs_mean = self.avg_pool(torch.abs(x)).view(b, c, 1)
        y = self.conv(torch.cat([abs_mean, mean, std], dim=2).transpose(1, 2))
        return x * self.sigmoid(y.transpose(1, 2).unsqueeze(-1))

# --- SSS: Spectral-Structure Stream (Vision Mamba Block) ---
class SSS_Block(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.d_inner = dim * 2
        self.dt_rank = math.ceil(dim / 16)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 4, groups=self.d_inner, padding=3)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x, z_anchor=None):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        if z_anchor is not None: x_flat = x_flat + z_anchor.unsqueeze(1)
        x_in, z_gate = self.in_proj(x_flat).chunk(2, dim=-1)
        x_conv = self.act(self.conv1d(x_in.transpose(1, 2))[:, :, :H*W])
        x_dbl = self.x_proj(x_conv.transpose(1, 2))
        dt, B_s, C_s = torch.split(x_dbl, [self.dt_rank, 16, 16], dim=-1)
        y = selective_scan_fn(x_conv, self.dt_proj(dt).transpose(1, 2), -torch.exp(self.A_log), B_s.transpose(1, 2), C_s.transpose(1, 2), self.D.float(), z=z_gate.transpose(1, 2), delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        return self.out_proj(y.transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)

# --- STS: Spatial-Texture Stream (PID-SSM Block) ---
class STS_Block(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.d_inner = dim * 2
        self.dt_rank = math.ceil(dim / 16)
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 4, groups=self.d_inner, padding=3)
        self.act = nn.SiLU()
        self.k_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.d_inner), nn.SiLU())
        self.q_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.d_inner), nn.SiLU())
        self.dt_bc_proj = nn.Linear(self.d_inner * 2, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.gate_proj = nn.Linear(self.d_inner * 2, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x, K, Q):
        B, C, H, W = x.shape
        x_conv = self.act(self.conv1d(self.in_proj(x.flatten(2).transpose(1, 2)).transpose(1, 2))[:, :, :H*W])
        x_t, K_p, Q_p = x_conv.transpose(1, 2), self.k_proj(K.flatten(2).transpose(1, 2)), self.q_proj(Q.flatten(2).transpose(1, 2))
        x_dbl = self.dt_bc_proj(torch.cat([x_t, K_p], dim=-1))
        dt, B_s, C_s = torch.split(x_dbl, [self.dt_rank, 16, 16], dim=-1)
        z_g = self.gate_proj(torch.cat([x_t, Q_p], dim=-1)).transpose(1, 2)
        y = selective_scan_fn(x_conv, self.dt_proj(dt).transpose(1, 2), -torch.exp(self.A_log), B_s.transpose(1, 2), C_s.transpose(1, 2), self.D.float(), z=z_g, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        return self.out_proj(y.transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)

# --- Dual-Stream Mamba Block ---
class DualStreamBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sss = SSS_Block(dim)
        self.sts = STS_Block(dim * 3)

    def forward(self, m_ll, m_high, z, K, Q):
        return m_ll + self.sss(m_ll, z), m_high + self.sts(m_high, K, Q)

# --- S3Mamba-Pan: Main Architecture ---

class S3Mamba_Pan(nn.Module):
    def __init__(self, ms_c=8, pan_c=1, dim=64, layers=3):
        super().__init__()
        self.conv_ms, self.conv_pan = nn.Conv2d(ms_c, dim, 3, 1, 1), nn.Conv2d(pan_c, dim, 3, 1, 1)
        self.dwt, self.idwt = DWT_2D(), IDWT_2D()
        self.gsam = GSAM_Encoder(ms_c, dim)
        self.k_ex = nn.Conv2d(dim, dim * 3, 1)
        self.blocks = nn.ModuleList([DualStreamBlock(dim) for _ in range(layers)])
        self.adr, self.tail = ADR_Module(dim), nn.Conv2d(dim, ms_c, 3, 1, 1)

    def forward(self, ms, pan):
        z, f_m, f_p = self.gsam(ms), self.conv_ms(ms), self.conv_pan(pan)
        m_d, p_d = self.dwt(f_m), self.dwt(f_p)
        m_l, m_h = m_d[:, :f_m.shape[1]], m_d[:, f_m.shape[1]:]
        p_l, p_h = p_d[:, :f_m.shape[1]], p_d[:, f_m.shape[1]:]
        k_ex = self.k_ex(p_l)
        for b in self.blocks: m_l, m_h = b(m_l, m_h, z, k_ex, p_h)
        return self.tail(self.adr(self.idwt(torch.cat([m_l, m_h], 1)))) + ms