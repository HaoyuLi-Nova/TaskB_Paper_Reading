"""Wav2Vec window + causal pack + AudioProj. Shapes match generate_multitalk.py / AudioProjModel."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

T, vae_scale, t_lat = 81, 4, 21
WAV_BLOCKS, WAV_DIM = 12, 768
K, MID = 5, 2
CTX, OUT, HID = 32, 768, 512
H_HUMANS = 2


def fake_wav2vec(wav, n_frames):
    feat = F.interpolate(wav.view(1, 1, -1), size=n_frames, mode="linear", align_corners=True)
    feat = feat.reshape(n_frames, 1, 1).expand(n_frames, WAV_BLOCKS, WAV_DIM)
    return feat + torch.randn_like(feat) * 0.1       # [T, 12, 768]  hidden_states[1:]


def window_audio(emb):
    indices = torch.arange(2 * 2 + 1) - 2            # [-2, -1, 0, 1, 2]
    idx = torch.arange(T).unsqueeze(1) + indices.unsqueeze(0)
    idx = idx.clamp(min=0, max=emb.shape[0] - 1)
    return emb[idx]                                  # [81, 5, 12, 768]


def pack_to_latent(win):
    first = win[:, :1]                               # [H, 1, 5, 12, 768]
    latter = rearrange(win[:, 1:], "b (n_t n) w s c -> b n_t n w s c", n=vae_scale)
    pack0 = rearrange(latter[:, :, :1, :MID + 1], "b n_t n w s c -> b n_t (n w) s c")
    packM = rearrange(latter[:, :, 1:-1, MID:MID + 1], "b n_t n w s c -> b n_t (n w) s c")
    packL = rearrange(latter[:, :, -1:, MID:], "b n_t n w s c -> b n_t (n w) s c")
    latter8 = torch.concat([pack0, packM, packL], dim=2)  # [H, 20, 8, 12, 768]
    return first, latter8


class AudioProjModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(K * WAV_BLOCKS * WAV_DIM, HID)                 # 5*12*768 -> 512
        self.proj1_vf = nn.Linear((K + vae_scale - 1) * WAV_BLOCKS * WAV_DIM, HID)  # 8*12*768 -> 512
        self.proj2 = nn.Linear(HID, HID)
        self.proj3 = nn.Linear(HID, CTX * OUT)
        self.norm = nn.LayerNorm(OUT)

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B = audio_embeds.shape[0]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        audio_embeds = audio_embeds.view(audio_embeds.shape[0], -1)
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        audio_embeds_vf = audio_embeds_vf.view(audio_embeds_vf.shape[0], -1)
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c.view(batch_size_c * N_t, C_a)))
        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c * N_t, CTX, OUT)
        context_tokens = self.norm(context_tokens)
        return rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)  # [H, 21, 32, 768]


wav = torch.randn(int(T / 25 * 16000))               # 81 frames @ 25 fps, 16 kHz
audio = torch.concat(
    [window_audio(fake_wav2vec(wav, T))[None] for _ in range(H_HUMANS)], 0
)                                                    # [2, 81, 5, 12, 768]
first, latter8 = pack_to_latent(audio)               # [2, 1, 5, 12, 768], [2, 20, 8, 12, 768]
tokens = AudioProjModel()(first, latter8)            # [2, 21, 32, 768]
kv = torch.concat(tokens.split(1), dim=2)            # [1, 21, 64, 768]
