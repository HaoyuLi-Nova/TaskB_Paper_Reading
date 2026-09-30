# `translations/` — 中文全文译本

每篇译本是人工撰写的 `paper_zh.md`（**不要**用脚本/API 批量生成正文）：

1. 按原文结构全文翻译  
2. 插图：先用 `scripts/figures_to_png.py` 转为 **PNG**，再引用  
3. 表格改为 Markdown；公式保留 LaTeX  

```text
translations/<track>/<slug>/paper_zh.md
```

原文在 [`../arxiv/`](../arxiv/)；阅读笔记三件套（`QA.md` / `critique.md` / `note.md`）在 [`../notes/<track>/<slug>/`](../notes/)，不要写进本目录。

转图示例：

```bash
python scripts/figures_to_png.py arxiv/base_models/ldm/extracted/img/final_figure.pdf
```

引用示例（从 `translations/<track>/<slug>/` 出发，三级回到仓库根）：

```markdown
![图3：……](../../../arxiv/base_models/ldm/extracted/img/final_figure.png)
```
