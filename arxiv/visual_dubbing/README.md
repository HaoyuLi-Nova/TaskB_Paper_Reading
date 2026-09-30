# `visual_dubbing/` — 画面译制与跨语言说话脸编辑

服务项目：**Faceglot**（多模态定制译制片，Target SIGGRAPH 2026）。

权威元数据：[`../../papers.yaml`](../../papers.yaml) 中 `track: visual_dubbing`。

**本轨资产优先级：`source.tar.gz` / `extracted/` ≫ `paper.pdf`。**

## 课题把三条线拆开

| 路线 | 代表 | 对 Faceglot 的意义 |
|------|------|-------------------|
| 嘴部 inpaint / 无 mask 但仍「只改嘴」 | Wav2Lip、LatentSync、MuseTalk、**X-Dub**、OmniSync、KeySync | 直接基线；X-Dub 的 bootstrap 骨架可复用，数据假设必须打破 |
| 稀疏帧 / 全身跟音频重生 | **InfiniteTalk**、LongCat-Avatar | 对照「生成 ≠ 锁镜头编辑」 |
| 跨语言说话脸（生成，非剪辑） | DisentTalk、Polyglot | 支撑「中英说话表情不一致」的主张，但任务不是 V2V 译制 |

音频侧电影配音（FunCineForge / Neural Dubber）与生成数字人（LongCat / OmniHuman）不在本轨，见 [`../audio_driven/`](../audio_driven/)。  
数据构造用的姿态控制见 [`../base_models/avcontrol/`](../base_models/avcontrol/)（LTX-2）。

## 阅读优先级（Faceglot）

完整理由与必答问题见 [`../../ROADMAP.md`](../../ROADMAP.md) 的 Faceglot 一节。

| 优先级 | slug | 为何先读 |
|--------|------|----------|
| **P0 立刻精读** | `xdub`, `latentsync`, `omnisync`, `infinitetalk` | 任务定义、数据假设、编辑 vs 生成的差 |
| **P1 方法/主张** | `keysync`, `musetalk`, `videoretalking`, `disenttalk`, `polyglot` | leakage、实时 inpaint、跨语言表情 |
| **P2 背景/数据** | `difftalk`, `diff2lip`, `openhumanvid` + `audio_driven/wav2lip` | 实验表里的旧基线与中文源语料 |

交叉必读（不在本轨）：`base_models/{wan,ltx2,avcontrol}`，`audio_driven/{longcat_avatar_15,funcineforge,wan_s2v}`。

## 下载 TeX

```bash
python scripts/download_catalog.py --track visual_dubbing --with-tex

# P0 单篇
python scripts/download_catalog.py --slug xdub --with-tex
python scripts/download_catalog.py --slug latentsync --with-tex
python scripts/download_catalog.py --slug omnisync --with-tex
python scripts/download_catalog.py --slug infinitetalk --with-tex
```

落盘：`arxiv/visual_dubbing/<slug>/source.tar.gz`，可解则进 `extracted/`。

## 手动补录

- **HDTF**（CVPR 2021，Zhang et al.）：无 arXiv e-print，评测集说明见 CVPR Open Access；不必强行入库 TeX。
- **StyleSync**（CVPR 2023）：通常无 arXiv TeX。
- arXiv 仅 PDF、无 e-print 的条目：脚本会 warn，需另找源。
