# `code/` — 论文学习代码

按 `code/<track>/<slug>/` 与笔记、原文对齐。这里放**可运行的最小实现 / 教程**，不是仓库运维脚本（运维在 `scripts/`）。

## Foundations

| 论文 | 路径 | 运行 |
|------|------|------|
| Transformer | [`foundations/transformer/toy/`](foundations/transformer/toy/) | `cd foundations/transformer/toy && python demo.py` |
| SimCLR | [`foundations/simclr/`](foundations/simclr/) | `python foundations/simclr/simclr_minimal.py` |
| Jigsaw | [`foundations/jigsaw/`](foundations/jigsaw/) | 见脚本内说明 |
| Dynamic Conv | [`foundations/dynamic_conv/`](foundations/dynamic_conv/) | 见脚本内说明 |
| Deformable Conv | [`foundations/deformable_conv/`](foundations/deformable_conv/) | 见脚本内说明 |
| AdaLN | [`foundations/adaln/`](foundations/adaln/) | `cd foundations/adaln && python demo.py` |
| LayerNorm | [`foundations/adaln/layernorm_demo.py`](foundations/adaln/layernorm_demo.py) | `cd foundations/adaln && python layernorm_demo.py` |

## Base models

| 论文 | 路径 | 运行 |
|------|------|------|
| LDM | [`base_models/ldm/`](base_models/ldm/) | `python base_models/ldm/featuremap_to_sequence.py` |
| Wan | [`base_models/wan/`](base_models/wan/) | 官方推理代码（Wan2.1 浅克隆）；笔记见 `notes/base_models/wan2.1/` |
| Ovi | [`base_models/ovi/`](base_models/ovi/) | 官方推理代码；见目录内 `README.md` / `README_LOCAL.md` |

## Audio driven

| 论文 | 路径 | 运行 |
|------|------|------|
| MultiTalk | [`audio_driven/multitalk/`](audio_driven/multitalk/) | 官方推理代码（SSH clone）；见 `README.md` / `README_LOCAL.md` |
| AnyTalker | [`audio_driven/anytalker/`](audio_driven/anytalker/) | 官方推理代码（SSH 完整克隆）；见 `README.md` / `README_LOCAL.md` |
| InterActHuman | [`audio_driven/interacthuman/`](audio_driven/interacthuman/) | 官方 demonstration 代码（SSH 完整克隆）；见 `README.md` / `README_LOCAL.md` |

## Multi view

官方仓库浅克隆（`--depth 1`，`GIT_LFS_SKIP_SMUDGE=1`），路径与 `papers.yaml` 的 `multi_view` slug 对齐。权重不在本仓库，见各目录 `README_LOCAL.md`。

| 论文 | 路径 | 来源 |
|------|------|------|
| QuerySplat | [`multi_view/querysplat/`](multi_view/querysplat/) | [inspatio/QuerySplat](https://github.com/inspatio/QuerySplat) |

## EEG Visual

官方仓库浅克隆（`--depth 1`，`GIT_LFS_SKIP_SMUDGE=1`），路径与 `papers.yaml` 的 `eeg_visual` slug 对齐。

| 论文 | 路径 | 来源 |
|------|------|------|
| NICE | [`eeg_visual/nice/`](eeg_visual/nice/) | [eeyhsong/NICE-EEG](https://github.com/eeyhsong/NICE-EEG) |
| ATM | [`eeg_visual/atm/`](eeg_visual/atm/) | [ncclab-sustech/EEG_Image_decode](https://github.com/ncclab-sustech/EEG_Image_decode) |
| DreamDiffusion | [`eeg_visual/dreamdiffusion/`](eeg_visual/dreamdiffusion/) | [bbaaii/DreamDiffusion](https://github.com/bbaaii/DreamDiffusion) |
| CognitionCapturer | [`eeg_visual/cognitioncapturer/`](eeg_visual/cognitioncapturer/) | [XiaoZhangYES/CognitionCapturer](https://github.com/XiaoZhangYES/CognitionCapturer) |
| NECOMIMI | [`eeg_visual/necomimi/`](eeg_visual/necomimi/) | [ChiShengChen/EEG_gen_img_NECOMIMI](https://github.com/ChiShengChen/EEG_gen_img_NECOMIMI) |
| MindCine | [`eeg_visual/mindcine/`](eeg_visual/mindcine/) | [KevinZhou6/MindCine](https://github.com/KevinZhou6/MindCine) |
| Neuro-3D | [`eeg_visual/neuro_3d/`](eeg_visual/neuro_3d/) | [gzq17/neuro-3D](https://github.com/gzq17/neuro-3D) |
| Mind2Matter | [`eeg_visual/mind2matter/`](eeg_visual/mind2matter/) | [sddwwww/Mind2Matter](https://github.com/sddwwww/Mind2Matter) |
| 3D-Telepathy | [`eeg_visual/eeg_to_3d/`](eeg_visual/eeg_to_3d/) | [gegen666/EEGTo3D](https://github.com/gegen666/EEGTo3D) |

## Remote sensing

官方仓库浅克隆（`--depth 1`），路径与 `papers.yaml` 的 `remote_sensing` slug 对齐。权重与训练数据不在本仓库。

| 论文 | 路径 | 来源 |
|------|------|------|
| S³Mamba-Pan | [`remote_sensing/s3mamba_pan/`](remote_sensing/s3mamba_pan/) | [FreeZS-a/S3Mamba](https://github.com/FreeZS-a/S3Mamba) |

新增代码时：在对应 slug 下建目录，并在 `papers.yaml` 该条目加 `code:` 字段。
