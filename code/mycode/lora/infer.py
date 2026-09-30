#!/usr/bin/env python3
"""Compare base vs LoRA T2V on the crush prompt. Writes mp4 under runs/lora/samples."""
from __future__ import annotations

import argparse

from peft import PeftModel

from common import add_wan_path, apply_hf_env, load_cfg, run_dirs


def generate(pipe, prompt: str, cfg: dict, seed: int):
    return pipe.generate(
        prompt,
        size=(cfg["width"], cfg["height"]),
        frame_num=cfg["frames"],
        sampling_steps=cfg["infer"]["steps"],
        guide_scale=cfg["infer"]["guide_scale"],
        seed=seed,
        offload_model=True,
        shift=cfg["train"]["shift"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "lora", "both"], default="both")
    args = ap.parse_args()

    cfg = load_cfg()
    apply_hf_env(cfg)
    add_wan_path(cfg)
    dirs = run_dirs(cfg)
    (dirs["samples"] / "base").mkdir(parents=True, exist_ok=True)
    (dirs["samples"] / "lora").mkdir(parents=True, exist_ok=True)

    from wan.configs import WAN_CONFIGS
    from wan.textimage2video import WanTI2V
    from wan.utils.utils import save_video

    wan_cfg = WAN_CONFIGS["ti2v-5B"]
    pipe = WanTI2V(
        config=wan_cfg,
        checkpoint_dir=cfg["ckpt_dir"],
        device_id=0,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    prompt = cfg["infer"]["prompt"]
    seed = cfg["infer"]["seed"]

    if args.mode in ("base", "both"):
        video = generate(pipe, prompt, cfg, seed)
        out = dirs["samples"] / "base" / "crush.mp4"
        save_video(video[None], str(out), fps=wan_cfg.sample_fps, nrow=1)
        print("base", out)

    if args.mode in ("lora", "both"):
        pipe.model = PeftModel.from_pretrained(pipe.model, str(dirs["adapter"]))
        pipe.model.eval()
        video = generate(pipe, prompt, cfg, seed)
        out = dirs["samples"] / "lora" / "crush.mp4"
        save_video(video[None], str(out), fps=wan_cfg.sample_fps, nrow=1)
        print("lora", out)


if __name__ == "__main__":
    main()
