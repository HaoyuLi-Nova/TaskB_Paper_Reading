#!/usr/bin/env python3
"""Download finetrainers/crush-smol via the HF mirror into video_workspace/data."""
from __future__ import annotations

from huggingface_hub import snapshot_download

from common import apply_hf_env, load_cfg, run_dirs


def main() -> None:
    cfg = load_cfg()
    apply_hf_env(cfg)
    dirs = run_dirs(cfg)
    dirs["data"].mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=cfg["dataset_repo"],
        repo_type="dataset",
        local_dir=str(dirs["data"]),
    )
    n = len(list(dirs["videos"].glob("*.mp4")))
    print(f"dataset={path} videos={n}")


if __name__ == "__main__":
    main()
