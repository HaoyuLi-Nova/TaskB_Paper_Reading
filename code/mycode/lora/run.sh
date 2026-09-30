#!/usr/bin/env bash
set -euo pipefail
source /data1/lihaoyu/video_workspace/env.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON=/data1/lihaoyu/miniconda3/envs/videodiff/bin/python
cd "$(dirname "$0")"
$PYTHON download.py
$PYTHON precompute.py
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" $PYTHON train.py
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" $PYTHON infer.py --mode both
