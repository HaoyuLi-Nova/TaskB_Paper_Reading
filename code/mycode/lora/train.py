#!/usr/bin/env python3
"""Flow-matching LoRA on Wan2.2-TI2V-5B attention projections only."""
from __future__ import annotations

import json
import logging
import math
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from common import add_wan_path, apply_hf_env, load_cfg, run_dirs


def wrap_ckpt(model) -> None:
    for block in model.blocks:
        orig = block.forward

        def _fwd(x, e, seq_lens, grid_sizes, freqs, context, context_lens, _orig=orig):
            def fn(x_, e_, ctx):
                return _orig(x_, e_, seq_lens, grid_sizes, freqs, ctx, context_lens)

            return checkpoint(fn, x, e, context, use_reentrant=False)

        block.forward = _fwd
    

def sample_t(n: int, shift: float, device) -> torch.Tensor:
    u = torch.rand(n, device=device)
    t = torch.sigmoid(0.5 * torch.logit(u.clamp(1e-4, 1 - 1e-4)))
    return shift * t / (1.0 + (shift - 1.0) * t)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_cfg()
    apply_hf_env(cfg)
    add_wan_path(cfg)
    dirs = run_dirs(cfg)
    dirs["adapter"].mkdir(parents=True, exist_ok=True)

    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel

    wan_cfg = WAN_CONFIGS["ti2v-5B"]
    device = torch.device("cuda:0")
    index = json.loads((dirs["run"] / "index.json").read_text())
    ids = [u["id"] for u in index]

    model = WanModel.from_pretrained(cfg["ckpt_dir"])
    model.requires_grad_(False)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora"]["rank"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=list(cfg["lora"]["targets"]),
            init_lora_weights="gaussian",
            bias="none",
        ),
    )
    wrap_ckpt(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.to(device=device, dtype=wan_cfg.param_dtype)
    model.train()
    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    h, w = cfg["height"], cfg["width"]
    nframes = cfg["frames"]
    stride = wan_cfg.vae_stride
    patch = wan_cfg.patch_size
    lat_f = (nframes - 1) // stride[0] + 1
    lat_h, lat_w = h // stride[1], w // stride[2]
    seq_len = math.ceil(lat_f * lat_h * lat_w / (patch[1] * patch[2]))
    steps = cfg["train"]["steps"]
    shift = cfg["train"]["shift"]
    log_every = cfg["train"]["log_every"]
    save_every = cfg["train"]["save_every"]

    running = 0.0
    pbar = tqdm(range(1, steps + 1), desc="train")
    for step in pbar:
        sid = random.choice(ids)
        x1 = torch.load(dirs["latents"] / f"{sid}.pt", map_location=device).float()
        ctx = torch.load(dirs["contexts"] / f"{sid}.pt", map_location=device).float()
        noise = torch.randn_like(x1)
        t01 = sample_t(1, shift, device).reshape(())
        xt = (1.0 - t01) * x1 + t01 * noise
        target = noise - x1
        timestep = (t01 * wan_cfg.num_train_timesteps).reshape(1)

        with torch.amp.autocast("cuda", dtype=wan_cfg.param_dtype):
            pred = model(
                [xt.to(wan_cfg.param_dtype)],
                t=timestep,
                context=[ctx.to(wan_cfg.param_dtype)],
                seq_len=seq_len,
            )[0]
            loss = F.mse_loss(pred.float(), target.float())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        running += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if step % log_every == 0:
            avg = running / log_every
            running = 0.0
            with open(dirs["log"], "a") as logf:
                logf.write(f"step={step} loss={avg:.4f} t={float(t01):.3f} id={sid}\n")
        if step % save_every == 0 or step == steps:
            dest = dirs["adapter"] if step == steps else dirs["run"] / f"adapter-{step}"
            dest.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(dest))
            print(f"saved {dest}")


if __name__ == "__main__":
    main()
