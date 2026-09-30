# Paper_Reading（Codex）

目录约定以 `STRUCTURE.md` 为准，路线图见 `ROADMAP.md`，元数据以 `papers.yaml` 为准。一文四路径：`arxiv/` · `translations/` · `notes/` · `code/`，同一 `track/slug`。

- 全文译本只写 `translations/<track>/<slug>/paper_zh.md`，**人工撰写**，禁止脚本/API 批量生成正文。
- 精读三件套只写 `notes/<track>/<slug>/{QA,critique,note}.md`，**不得覆盖译本**。
- 插图先 `scripts/figures_to_png.py` 转为 PNG，译本只引用 `.png`。
- `paper.pdf` / `source.tar.gz` 默认 gitignore，用 `scripts/download_catalog.py` 拉取。
- 精读工作流与六维模板：`.agents/skills/reading-papers/SKILL.md`。

## 公式定界

对话用 `\(` `\)` 与 `\[` `\]`。写入 `.md` 用 `$` / `$$`。代码块内 `$` 保持字面量。
