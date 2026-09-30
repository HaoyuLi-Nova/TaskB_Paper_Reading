# `audio_driven/` — 语音驱动视频与音色协调

服务项目：**HarmonizedVisualAudio**（音色与人脸和谐的音频驱动人体视频）。

权威元数据：[`../../papers.yaml`](../../papers.yaml) 中 `track: audio_driven`。

**本轨资产优先级：`source.tar.gz` / `extracted/` ≫ `paper.pdf`。**

## 与现有 track 的关系

| 相关论文 | 位置 | 说明 |
|----------|------|------|
| **Ovi** | [`../base_models/ovi/`](../base_models/ovi/) | 联合生成底座；**TeX 已齐** |
| **LTX-2** | [`../base_models/ltx2/`](../base_models/ltx2/) | 非对称双流 T2AV 基座；与 Ovi 对照 |
| **Wan** | [`../base_models/wan/`](../base_models/wan/) | 视频 DiT 基座；Wan-S2V 依赖 |

读本 track 时请同时打开 **Ovi**：`translations/base_models/ovi/paper_zh.md`。  
笔记三件套：`notes/audio_driven/<slug>/{QA,critique,note}.md`。

**Faceglot 交叉阅读**：画面译制线在 [`../visual_dubbing/`](../visual_dubbing/)（X-Dub / LatentSync / InfiniteTalk）。本轨对 Faceglot 最相关的是 `longcat_avatar_15`（生成对照）与 `funcineforge`（可选音色）。

## 阅读优先级

| 优先级 | slug |
|--------|------|
| **P0 先下 TeX** | `omnihuman`, `fantasytalking`, `multitalk`, `anytalker`, `bind_your_avatar`, `dreamvoice`, `longcat_avatar_15` + `base_models/ovi` |
| **P1** | `omniavatar`, `wan_s2v`, `stableavatar`, `interacthuman`, `universe1`, `harmony_av`, `omnihuman15`, `funcineforge`, `neural_dubber` |
| **P2 背景** | `wav2lip`, `sadtalker`, `emo`, `instructdubber` |

## 下载 TeX

```bash
# 整轨 e-print（推荐）
python scripts/download_catalog.py --track audio_driven --with-tex

# P0 单篇
python scripts/download_catalog.py --slug omnihuman --with-tex
python scripts/download_catalog.py --slug fantasytalking --with-tex
python scripts/download_catalog.py --slug multitalk --with-tex
python scripts/download_catalog.py --slug anytalker --with-tex
python scripts/download_catalog.py --slug bind_your_avatar --with-tex
python scripts/download_catalog.py --slug dreamvoice --with-tex
```

落盘：`arxiv/audio_driven/<slug>/source.tar.gz`，可解则进 `extracted/`。

## 手动补录

- **FVMVC**（ACM MM 2023）：通常无 arXiv TeX，需从 ACM / 作者页手动放入 `arxiv/audio_driven/fvmvc/`。
- arXiv 仅 PDF、无 e-print 的条目：脚本会 warn，需另找源。
