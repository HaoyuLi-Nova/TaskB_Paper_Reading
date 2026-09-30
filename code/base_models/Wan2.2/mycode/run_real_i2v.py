"""Wan2.2-TI2V-5B 真实 I2V。

断点：
  wan/textimage2video.py  WanTI2V.i2v     整条 I2V 流程
  mycode/dit_trace.py     dit_forward     patch / 文本 / 时间嵌入
  mycode/dit_trace.py     dit_block       每一层 self-attn / cross-attn / FFN

运行：
  CUDA_VISIBLE_DEVICES=2 python mycode/run_real_i2v.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image

from dit_trace import attach
from wan.configs import WAN_CONFIGS
from wan.textimage2video import WanTI2V
from wan.utils.utils import save_video

CKPT = "/data1/lihaoyu/video_workspace/models/Wan2.2-TI2V-5B"
IMAGE = ROOT / "examples/i2v_input.JPG"
PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
# 官方默认 1280*704 / 50 步 / 121 帧。下面缩小只为把结构看清楚。
MAX_AREA, FRAMES, STEPS = 1280 * 704, 121, 50


def main():
    cfg = WAN_CONFIGS["ti2v-5B"]
    pipe = WanTI2V(
        config=cfg,
        checkpoint_dir=CKPT,
        device_id=0,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    attach(pipe.model)

    video = pipe.generate(
        PROMPT,
        img=Image.open(IMAGE).convert("RGB"),
        max_area=MAX_AREA,
        frame_num=FRAMES,
        sampling_steps=STEPS,
        seed=42,
        offload_model=True,
    )
    print("video", tuple(video.shape))
    out = Path(__file__).parent / "real_i2v.mp4"
    save_video(video[None], str(out), fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
    print("saved", out)


if __name__ == "__main__":
    main()
