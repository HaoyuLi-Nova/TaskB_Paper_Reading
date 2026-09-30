"""One two-person train step, from raw inputs. Official: generate_multitalk.py + wan/multitalk.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from multitalk_layers import (
    C, T, H, W, L, h, w,
    CLIP_LEN, CLIP_DIM, TEXT_LEN, TEXT_DIM,
    WAV_BLOCKS, WAV_DIM, K, MID,
    TinyMultiTalk, causal_vae_encode, pack_mask,
)

SR, FPS = 16000, 25
N_SAMPLES = T * SR // FPS                                # 51840  81 frames @ 25fps
CLIP_SIZE, CLIP_PATCH = 224, 14                          # ViT-H/14


# --- raw encoders (weights fake; layout matches official) ---

t5_tok = nn.Embedding(256, TEXT_DIM)                     # stand-in for umT5-xxl


class FakeCLIPVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(C, CLIP_DIM, CLIP_PATCH, CLIP_PATCH)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, CLIP_DIM))
        self.pos_embedding = nn.Parameter(torch.randn(1, CLIP_LEN, CLIP_DIM))

    def forward(self, first_frame):
        x = F.interpolate(first_frame[None], size=(CLIP_SIZE, CLIP_SIZE), mode="bicubic", align_corners=False)
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)  # [1, 256, 1280]
        x = torch.cat([self.cls_embedding.expand(x.size(0), -1, -1), x], 1)
        return x + self.pos_embedding                            # [1, 257, 1280]


clip = FakeCLIPVisual()


def encode_t5(prompt):
    # tokenizer + umT5 encoder. Official returns unpadded [seq, 4096]; DiT later pads to 512.
    ids = torch.tensor([ord(ch) % 256 for ch in prompt])
    return t5_tok(ids)                                   # [seq, 4096]


def encode_clip(first_frame):
    return clip(first_frame)


def encode_wav2vec(wav):
    # feature_extractor + interpolate to fps=25 + stack hidden_states[1:] (12 layers)
    feat = F.interpolate(wav.view(1, 1, -1), size=T, mode="linear", align_corners=True)
    feat = feat.reshape(T, 1, 1).expand(T, WAV_BLOCKS, WAV_DIM)
    return feat + torch.randn_like(feat) * 0.1           # [81, 12, 768]


def window_audio(emb):
    indices = torch.arange(2 * 2 + 1) - 2                # [-2,-1,0,1,2]
    idx = torch.arange(T).unsqueeze(1) + indices.unsqueeze(0)
    idx = idx.clamp(min=0, max=emb.shape[0] - 1)
    return emb[idx]                                      # [81, 5, 12, 768]


def masks_from_bbox(boxes):
    # boxes: pixel [y0, x0, y1, x1] for person1, person2 → latent [3, 8, 8]
    pix = torch.zeros(3, H, W)
    (y0, x0, y1, x1), (y0b, x0b, y1b, x1b) = boxes
    pix[0, y0:y1, x0:x1] = 1
    pix[1, y0b:y1b, x0b:x1b] = 1
    pix[2] = 1 - (pix[0] + pix[1]).clamp(0, 1)
    return F.interpolate(pix.unsqueeze(0), size=(h, w), mode="nearest")[0]


# --- one sample, raw ---

prompt = "two people talking in a car"
video = torch.randn(C, T, H, W)                          # [3, 81, 64, 64]  rgb video
wav1 = torch.randn(N_SAMPLES)                            # person1 16 kHz pcm
wav2 = torch.randn(N_SAMPLES)                            # person2 16 kHz pcm
boxes = [(0, 0, H, W // 2), (0, W // 2, H, W)]           # left / right person

context = encode_t5(prompt).detach()                     # [seq, 4096]  T5 frozen
clip_fea = encode_clip(video[:, 0]).detach()             # [1, 257, 1280]  CLIP frozen
audio = torch.stack([window_audio(encode_wav2vec(wav1)),
                     window_audio(encode_wav2vec(wav2))], 0).detach()  # [2, 81, 5, 12, 768]
masks = masks_from_bbox(boxes)                           # [3, 8, 8]
t = torch.tensor([0.3])

x_1 = causal_vae_encode(video)                           # [16, 21, 8, 8]
x_0 = torch.randn_like(x_1)
X_t = t * x_1 + (1 - t) * x_0

keep = torch.zeros(T)
keep[0] = 1
cond = torch.cat([video[:, :1], torch.zeros(C, T - 1, H, W)], 1) # [3, 81, 64, 64]
y = torch.concat([pack_mask(keep), causal_vae_encode(cond).unsqueeze(0)], dim=1)  # [1, 20, 21, 8, 8]

dit = TinyMultiTalk()
for name, p in dit.named_parameters():
    p.requires_grad = ("audio_proj" in name) or ("audio_cross_attn" in name) or ("norm_x" in name)
opt = torch.optim.AdamW([p for p in dit.parameters() if p.requires_grad], lr=2e-5)

v_pred = dit([X_t], t * 1000, [context], L, clip_fea, y, audio, masks)
loss = F.mse_loss(v_pred, x_1 - x_0)
opt.zero_grad()
loss.backward()
opt.step()
