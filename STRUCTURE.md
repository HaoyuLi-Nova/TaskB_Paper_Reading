# 仓库结构规范（STRUCTURE）

本仓库长期用于：**arXiv 原文管理、中文全文译本、阅读笔记、代码学习、横向对比**。
所有新内容按本文件约定落盘；元数据以 [`papers.yaml`](papers.yaml) 为准。
目录树与 [`README.md`](README.md) 的项目结构一节对齐。

---

## 1. 核心原则

1. **一文四路径**：`arxiv/` 原文 · `translations/` 译本 · `notes/` 阅读笔记 · `code/` 代码（后两者可选）。
2. **按 track/slug 对齐**：同一篇论文在四处使用相同 `track` 与 `slug`。
3. **译本与笔记分开**：`translations/` 只放全文中文翻译；`notes/` 只放该篇的三件套（见下）。
4. **译文由人（Agent）亲自撰写**：禁止用脚本 / API / 批量工具自动生成 `paper_zh.md` 正文。
5. **精读笔记不得覆盖译本**：三件套写到 `notes/<track>/<slug>/`，不要改 `paper_zh.md`。
6. **插图先转 PNG 再引用**：原图若为 PDF / JPG 等，先用 `scripts/figures_to_png.py`（PyMuPDF + Pillow）转为 PNG，译本中只引用 `.png`。
7. **工具与学习分离**：`scripts/` 只放下载与转图等运维工具；玩具/教程进 `code/`。
8. **大文件不进 git**：`paper.pdf`、`source.tar.gz` 默认 gitignore，本地用脚本拉取。

---

## 2. 目录职责

| 路径                                        | 职责                                                                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------ |
| `papers.yaml`                             | 标题、arXiv、track、slug、status、deps、`translation` / `notes` / `code` 路径  |
| `arxiv/<track>/<slug>/`                   | 从 arXiv 下载的`paper.pdf`、`source.tar.gz`、`extracted/`（含转好的 `.png`） |
| `translations/<track>/<slug>/paper_zh.md` | 全文中文译本（人工撰写）                                                             |
| `notes/<track>/<slug>/`                   | 单篇三件套：`QA.md`、`critique.md`、`note.md`                                     |
| `code/<track>/<slug>/`                    | 最小可运行实现、教程脚本                                                             |
| `comparisons/<track>/`                    | 跨论文对比                                                                           |
| `scripts/`                                | `download_catalog.py`、`figures_to_png.py` 等                                    |
| `ROADMAP.md`                              | 读什么、什么顺序                                                                     |
| `STRUCTURE.md`                            | 本文件：怎么放                                                                       |

专题页（跨论文、不占 slug）可放在 `notes/foundations/` 根下，如 `RoPE.md`、`AdaLN.md`。

---

## 3. 单篇论文目录模板

```text
arxiv/<track>/<slug>/
├── paper.pdf
├── source.tar.gz          # 可选
└── extracted/             # TeX / 原图 / 转好的 PNG

translations/<track>/<slug>/
└── paper_zh.md            # 按原文结构全文翻译（人工撰写）

notes/<track>/<slug>/      # 可选；有则尽量凑齐三件套
├── QA.md                  # 考察理解深度的问答
├── critique.md            # 六维批判精读
└── note.md                # 动机、创新点、技术方案

code/<track>/<slug>/       # 可选
├── README.md
└── *.py / toy/
```

### 译本工作流

```bash
# 1) 下载原文（PDF / TeX）
python scripts/download_catalog.py --slug ldm --with-tex

# 2) 把要引用的图转为 PNG（PDF / JPG / … → PNG）
python scripts/figures_to_png.py arxiv/base_models/ldm/extracted/img/foo.pdf
python scripts/figures_to_png.py arxiv/base_models/ldm/extracted/img/bar.jpg

# 3) 人工撰写 translations/<track>/<slug>/paper_zh.md（不要用代码生成正文）
```

译本约定：

- 章节与原文对齐，内容完整翻译（非摘要改写）
- 图：只引用 PNG，例如 `![图注](../../../arxiv/<track>/<slug>/extracted/.../xxx.png)`
- 表：Markdown 表格；公式：LaTeX

---

## 4. 译本相对链接

从 `translations/<track>/<slug>/paper_zh.md` 引用图（三级回到仓库根）：

```markdown
![fig](../../../arxiv/<track>/<slug>/extracted/...)
```

---

## 5. Tracks

| track                | 含义                                                      |
| -------------------- | --------------------------------------------------------- |
| `foundations`      | CNN / Transformer / 自监督等前置                          |
| `base_models`      | 扩散/DiT 基座与控制插件                                   |
| `geometry_control` | 相机与物体运动控制                                        |
| `multi_view`       | 多视角 / 3D 一致性                                        |
| `streaming`        | 流式视频生成                                              |
| `long_video`       | 长视频生成（分层/免训练/token 流/分镜；Forcing 见 streaming） |
| `vla`              | Vision-Language-Action 具身智能                           |
| `audio_driven`     | 语音驱动视频 / 多人声脸绑定 / VC（HarmonizedVisualAudio） |
| `visual_dubbing`   | 画面译制 / 跨语言说话脸编辑（Faceglot）                   |
| `eeg_visual`       | EEG 驱动图像 / 视频 / 3D 生成                           |
| `remote_sensing`   | 遥感全色锐化 / 多光谱融合（与视频生成主线正交）         |
| `spatial`          | 视觉语言模型的空间理解（与视频生成主线正交）           |

新增 track：先改 `papers.yaml` 的 `tracks:`，再建同名目录。

---

## 6. 状态机（papers.yaml status）

`pending` → `downloading` → `reading` → `notes_done` → `mastered`

`translation:` 指向 `paper_zh.md`；`notes:` 指向 `notes/<track>/<slug>/` 或其中某件（`QA.md` / `critique.md` / `note.md`），不要把译本写进 `notes:`。

---

## 7. 命名

- **slug**：小写 + 下划线，稳定不改（如 `deformable_conv`、`sd3_rectified_flow`）
- **不要**再使用 `01_` 数字前缀或 `bg_` 前缀作为目录名
- 全文中文译本统一叫 `paper_zh.md`
- 单篇笔记只使用这三个文件名：`QA.md`、`critique.md`、`note.md`

---

## 8. 常见操作

| 想做                | 命令 / 位置                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| 下载一批 PDF        | `python scripts/download_catalog.py --track base_models`                                            |
| 图转 PNG            | `python scripts/figures_to_png.py arxiv/<track>/`（只扫 `extracted/`；PyMuPDF+Pillow，EPS 需 gs） |
| 跑 Transformer 玩具 | `cd code/foundations/transformer/toy && python demo.py`                                             |
| 看进度              | 打开`papers.yaml`                                                                                   |
| 写对比              | `comparisons/<track>/*.md`                                                                          |
