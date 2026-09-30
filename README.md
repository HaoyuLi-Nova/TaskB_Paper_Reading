# Paper Reading

长期维护的个人论文精读仓库：**arXiv 原文 / 中文译本 / 阅读笔记 / 学习代码 / 横向对比** 分轨存放。

主题覆盖：模型结构与自监督基础 → 扩散生成基座与可控生成 → 几何运动控制 → 多视角一致性 → 流式视频生成 → 长视频生成。

> 论文 PDF / TeX 来自 [arXiv](https://arxiv.org/)，版权归原作者；本仓库仅整理学习笔记与代码练习。

---

## 目录约定

| 目录 / 文件                       | 放什么                                                   |
| --------------------------------- | -------------------------------------------------------- |
| [`papers.yaml`](papers.yaml)     | 元数据与进度（权威源）                                   |
| [`arxiv/`](arxiv/)               | 从 arXiv 下载的 PDF、TeX、`extracted/`（含转好的 PNG） |
| [`translations/`](translations/) | 人工撰写的全文译本`paper_zh.md`                        |
| [`notes/`](notes/)               | 单篇三件套：`QA.md` / `critique.md` / `note.md`（不是译本） |
| [`code/`](code/)                 | 按论文 slug 的玩具实现                                   |
| [`scripts/`](scripts/)           | 下载、**图转 PNG** 等工具                          |
| [`comparisons/`](comparisons/)   | 跨论文对比                                               |
| [`ROADMAP.md`](ROADMAP.md)       | 读什么、什么顺序                                         |
| [`STRUCTURE.md`](STRUCTURE.md)   | 怎么放、命名规范                                         |

每篇论文的稳定标识是 **`track/slug`**：

```text
arxiv/<track>/<slug>/
translations/<track>/<slug>/paper_zh.md
notes/<track>/<slug>/QA.md        # 考察理解深度
notes/<track>/<slug>/critique.md  # 六维批判
notes/<track>/<slug>/note.md      # 动机 / 创新点 / 技术方案
code/<track>/<slug>/              # 可选
```

**硬性约定**

1. `paper_zh.md` **由 Agent / 人亲自撰写**，禁止用代码批量生成译文正文。
2. 精读笔记写到 `notes/<track>/<slug>/` 的三件套，**不要覆盖** `translations/` 里的译本。
3. 插图先 `figures_to_png.py` 转为 **PNG**，译本中只引用 `.png`。

---

## 项目结构（当前）

```text
.
├── papers.yaml
├── ROADMAP.md / STRUCTURE.md
├── arxiv/<track>/<slug>/          # 原文 + extracted/*.png
├── translations/<track>/<slug>/paper_zh.md
├── notes/
│   ├── <track>/<slug>/QA.md        # 考察理解深度的问答
│   ├── <track>/<slug>/critique.md  # 六维批判笔记
│   └── <track>/<slug>/note.md      # 动机、创新点、技术方案
├── code/foundations/…
├── comparisons/
└── scripts/
    ├── download_catalog.py
    └── figures_to_png.py          # PDF/JPG → PNG
```

---

## 日常工作流

**1. 安装**

```bash
pip install -r requirements.txt
```

**2. 下载论文**

```bash
python scripts/download_catalog.py --track base_models
python scripts/download_catalog.py --slug ldm --with-tex
python scripts/download_catalog.py --all
```

**3. 转图 → 撰写译本**

```bash
# 需要引用的图先转 PNG
python scripts/figures_to_png.py arxiv/base_models/ldm/extracted/img/foo.pdf
python scripts/figures_to_png.py arxiv/base_models/ldm/extracted/img/bar.jpg

# 然后人工撰写：
# translations/base_models/ldm/paper_zh.md
```

**4. 更新进度**

编辑 [`papers.yaml`](papers.yaml) 的 `status`：`pending` → `reading` → `notes_done` → `mastered`。
译本路径写在 `translation:`；阅读笔记写在 `notes:`（指向该 slug 目录或其中某件，如 `critique.md`）。

**5. 跑学习代码 / 写笔记（可选）**

```bash
cd code/foundations/transformer/toy && python demo.py
# 问答 → notes/<track>/<slug>/QA.md
# 批判 → notes/<track>/<slug>/critique.md
# 技术方案 → notes/<track>/<slug>/note.md
```

---

## Tracks

| Track                      | 说明                                             |
| -------------------------- | ------------------------------------------------ |
| **foundations**      | 模型结构 / 自监督前置                            |
| **base_models**      | 生成基座与可控插件                               |
| **geometry_control** | 相机 / 物体 / 角色运动控制                       |
| **multi_view**       | 多视角与 3D 一致性                               |
| **streaming**        | 流式 / Forcing 系视频生成                        |
| **long_video**       | 长视频生成（分层、免训练外推、token 流、LLM 分镜） |
| **vla**              | Vision-Language-Action 具身智能                  |
| **audio_driven**     | 语音驱动视频 / 音色协调（HarmonizedVisualAudio） |
| **visual_dubbing**   | 画面译制 / 跨语言说话脸（Faceglot）              |
| **eeg_visual**       | EEG 驱动图像 / 视频 / 3D                       |
| **remote_sensing**   | 遥感全色锐化 / 多光谱融合（与主线正交）        |
| **spatial**          | 视觉语言模型的空间理解（与主线正交）           |

---

## 许可

- 笔记与脚本：[MIT License](LICENSE)
- 论文 PDF / LaTeX：遵循各论文在 arXiv 上的原始许可
