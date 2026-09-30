# Wan2.2-TI2V-5B LoRA lab

Tiny hydraulic-press LoRA on [finetrainers/crush-smol](https://huggingface.co/datasets/finetrainers/crush-smol) (~47 clips, ~55MB).

```
data  /data1/lihaoyu/video_workspace/data/lora/crush-smol
runs  /data1/lihaoyu/video_workspace/runs/lora/crush-smol
code  paper_reading/code/mycode/lora
```

```bash
source /data1/lihaoyu/video_workspace/env.sh
export HF_ENDPOINT=https://hf-mirror.com
cd paper_reading/code/mycode/lora
# or: bash run.sh
python download.py
python precompute.py
CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=0 python infer.py --mode both
```

Uses `videodiff`. LoRA is `r=16` on `self_attn`/`cross_attn` `{q,k,v,o}` only. AdaLN `modulation` is frozen. T5/VAE are precomputed so training keeps only the DiT on GPU.
