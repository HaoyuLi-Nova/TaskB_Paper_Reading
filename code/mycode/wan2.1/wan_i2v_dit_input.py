import torch
import torch.nn.functional as F

C, T, H, W = 3, 81, 720, 1280
c, s = 16, 4
t, h, w = 1 + (T - 1) // s, H // 8, W // 8  # 21, 90, 160

img = torch.randn(C, H, W)  # first frame
zeros = torch.zeros((C, T-1, H, W), device=img.device, dtype=img.dtype) # 0-padding frames

video = torch.cat([img[:, None], zeros], 1)  # + 0-padding frames -> [3, 81, 720, 1280]

# Wan-Encoder (fake): spatial /8, then causal temporal /4, then 3 -> 16 channels
pix = video.unsqueeze(0)  # [3, 81, 720, 1280] -> [1, 3, 81, 720, 1280]  avg_pool3d needs N,C,T,H,W
pix = F.avg_pool3d(pix, (1, 8, 8))  # [1, 3, 81, 720, 1280] -> [1, 3, 81, 90, 160]
pix = pix.squeeze(0)  # [1, 3, 81, 90, 160] -> [3, 81, 90, 160]

z0 = pix[:, :1]  # [3, 81, 90, 160] -> [3, 1, 90, 160]  first frame, no temporal compress
z_rest = pix[:, 1:]  # [3, 81, 90, 160] -> [3, 80, 90, 160]
z_rest = z_rest.reshape(C, t - 1, s, h, w)  # [3, 80, 90, 160] -> [3, 20, 4, 90, 160]
z_rest = z_rest.mean(2)  # [3, 20, 4, 90, 160] -> [3, 20, 90, 160]  each 4 frames -> 1 latent
latent = torch.cat([z0, z_rest], 1)  # [3, 1, 90, 160] + [3, 20, 90, 160] -> [3, 21, 90, 160]
latent = latent.repeat((c + C - 1) // C, 1, 1, 1)  # [3, 21, 90, 160] -> [18, 21, 90, 160]  tile RGB 6 times
latent = latent[:c]  # [18, 21, 90, 160] -> [16, 21, 90, 160]  paper c=16

M = torch.zeros(1, T, h, w)  # [1, 81, 90, 160]  all 0 = frames to generate
M[:, 0] = 1  # [1, 81, 90, 160]  frame 0 = 1 (keep), frames 1..80 = 0
first4 = M[:, :1].repeat_interleave(s, 1)  # [1, 1, 90, 160] -> [1, 4, 90, 160]  pad f0 to a pack of 4
rest = M[:, 1:]  # [1, 81, 90, 160] -> [1, 80, 90, 160]
mask = torch.cat([first4, rest], 1)  # [1, 4, 90, 160] + [1, 80, 90, 160] -> [1, 84, 90, 160]
mask = mask.view(1, t, s, h, w)  # [1, 84, 90, 160] -> [1, 21, 4, 90, 160]
mask = mask.transpose(1, 2)  # [1, 21, 4, 90, 160] -> [1, 4, 21, 90, 160]
mask = mask.squeeze(0)  # [1, 4, 21, 90, 160] -> [4, 21, 90, 160]

X_t = torch.randn(c, t, h, w)  # [16, 21, 90, 160]  noisy latent
x = torch.cat([X_t, latent, mask], 0)  # [16,21,90,160] + [16,21,90,160] + [4,21,90,160] -> [36, 21, 90, 160]

clip = torch.randn(1, 257, 1280)  # CLIP image encoder on first frame
t5 = torch.randn(512, 4096)  # umT5 encoder
timestep = torch.tensor([500.0])  # diffusion t
