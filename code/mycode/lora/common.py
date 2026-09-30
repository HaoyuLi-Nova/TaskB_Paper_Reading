"""Shared paths and helpers for the Wan TI2V-5B LoRA lab."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def load_cfg(path: Path | None = None) -> dict:
    path = path or HERE / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def apply_hf_env(cfg: dict) -> None:
    os.environ.setdefault("HF_ENDPOINT", cfg["hf_endpoint"])
    video_root = Path(os.environ.get("VIDEO_ROOT", "/data1/lihaoyu/video_workspace"))
    os.environ.setdefault("HF_HOME", str(video_root / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))


def add_wan_path(cfg: dict) -> None:
    sys.path.insert(0, cfg["wan_code"])


def run_dirs(cfg: dict) -> dict[str, Path]:
    data = Path(cfg["data_root"])
    run = Path(cfg["run_root"])
    return {
        "data": data,
        "videos": data / "videos",
        "run": run,
        "latents": run / "latents",
        "contexts": run / "contexts",
        "adapter": run / "adapter",
        "samples": run / "samples",
        "log": run / "train.log",
    }
