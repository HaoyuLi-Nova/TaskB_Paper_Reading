#!/usr/bin/env python3
"""Encode videos with Wan2.2 VAE and captions with UMT5. Writes .pt under runs/lora."""
from __future__ import annotations

import json
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import add_wan_path, apply_hf_env, load_cfg, run_dirs


def read_rgb(path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"empty video: {path}")
    return np.stack(frames)


def sample_resize(rgb: np.ndarray, n: int, h: int, w: int) -> torch.Tensor:
    t = rgb.shape[0]
    idx = np.linspace(0, t - 1, n).round().astype(int)
    x = torch.from_numpy(rgb[idx]).float()
    x = x.permute(0, 3, 1, 2) / 255.0
    x = x * 2 - 1
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    x = x.permute(1, 0, 2, 3)
    return x


def load_captions(data_root, trigger: str) -> dict[str, str]:
    names = (data_root / "videos.txt").read_text().strip().splitlines()
    prompts = (data_root / "prompt.txt").read_text().strip().splitlines()
    out = {}
    for name, prompt in zip(names, prompts):
        stem = name.split("/")[-1]
        text = prompt.strip()
        if trigger and trigger not in text:
            text = f"{trigger} {text}"
        out[stem] = text
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_cfg()
    apply_hf_env(cfg)
    add_wan_path(cfg)
    dirs = run_dirs(cfg)
    dirs["latents"].mkdir(parents=True, exist_ok=True)
    dirs["contexts"].mkdir(parents=True, exist_ok=True)

    from wan.configs import WAN_CONFIGS
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_2 import Wan2_2_VAE

    wan_cfg = WAN_CONFIGS["ti2v-5B"]
    ckpt = cfg["ckpt_dir"]
    device = torch.device("cuda:0")

    vae = Wan2_2_VAE(vae_pth=f"{ckpt}/{wan_cfg.vae_checkpoint}", device=device)
    t5 = T5EncoderModel(
        text_len=wan_cfg.text_len,
        dtype=wan_cfg.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=f"{ckpt}/{wan_cfg.t5_checkpoint}",
        tokenizer_path=f"{ckpt}/{wan_cfg.t5_tokenizer}",
    )

    captions = load_captions(dirs["data"], cfg["trigger"])
    index = []
    for mp4 in tqdm(sorted(dirs["videos"].glob("*.mp4")), desc="precompute"):
        video = sample_resize(read_rgb(mp4), cfg["frames"], cfg["height"], cfg["width"])
        with torch.no_grad():
            latent = vae.encode([video.to(device)])[0].cpu().half()
            ctx = t5([captions[mp4.name]], torch.device("cpu"))[0].cpu().half()
        torch.save(latent, dirs["latents"] / f"{mp4.stem}.pt")
        torch.save(ctx, dirs["contexts"] / f"{mp4.stem}.pt")
        index.append({"id": mp4.stem, "video": mp4.name, "caption": captions[mp4.name]})
    dirs["run"].mkdir(parents=True, exist_ok=True)
    (dirs["run"] / "index.json").write_text(json.dumps(index, indent=2))
    (dirs["run"] / "config.yaml").write_text((__import__("pathlib").Path(__file__).with_name("config.yaml")).read_text())
    print(f"wrote {len(index)} samples -> {dirs['run']}")


if __name__ == "__main__":
    main()
