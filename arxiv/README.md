# `arxiv/` — 从 arXiv 下载的原文资产

按 **track/slug** 组织。权威元数据：[`../papers.yaml`](../papers.yaml)。

对应译本在 [`../translations/`](../translations/)；阅读笔记三件套在 [`../notes/<track>/<slug>/{QA,critique,note}.md`](../notes/)。

## Tracks

| Track | 目录 |
|-------|------|
| Foundations | [`foundations/`](foundations/) |
| Base Models | [`base_models/`](base_models/) |
| Geometry Control | [`geometry_control/`](geometry_control/) |
| Multi-View | [`multi_view/`](multi_view/) |
| Streaming | [`streaming/`](streaming/) |
| Long Video | [`long_video/`](long_video/) |
| VLA | [`vla/`](vla/) |
| Audio-Driven | [`audio_driven/`](audio_driven/) |
| Visual Dubbing | [`visual_dubbing/`](visual_dubbing/) |
| EEG Visual | [`eeg_visual/`](eeg_visual/) |
| Remote Sensing | [`remote_sensing/`](remote_sensing/) |
| Spatial Understanding | [`spatial/`](spatial/) |

## 单篇约定

```text
arxiv/<track>/<slug>/
├── paper.pdf
├── source.tar.gz      # 可选
└── extracted/         # 可选 TeX / 图
```

## 下载

```bash
python scripts/download_catalog.py --track base_models
python scripts/download_catalog.py --slug ldm --with-tex
python scripts/download_catalog.py --all
# foundations 默认跳过；需要刷新时：
python scripts/download_catalog.py --track foundations --include-foundations
```
