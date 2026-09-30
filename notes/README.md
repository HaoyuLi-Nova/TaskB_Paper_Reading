# `notes/` — 阅读笔记

本目录放**单篇精读笔记**，不是全文译本。

译本在 [`../translations/<track>/<slug>/paper_zh.md`](../translations/)。
原文在 [`../arxiv/`](../arxiv/)。

每篇论文只使用这三个文件名（与 [`README.md`](../README.md) 项目结构一致）：

```text
notes/<track>/<slug>/QA.md        # 考察理解深度的问答
notes/<track>/<slug>/critique.md  # 六维批判精读
notes/<track>/<slug>/note.md      # 动机、创新点、技术方案
```

跨论文专题（如 RoPE、AdaLN）可放在 `notes/foundations/` 根下，不占用 slug 三件套。

写笔记时**不要覆盖** `translations/` 里的 `paper_zh.md`。
