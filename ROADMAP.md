# 精读路线图（VideoGen / Controllable Generation）

> **形成可持续的论文精读项目**。
> 总篇数：**26 篇新译本 + foundations 已有译本**。建议按依赖关系分 4 个阶段推进。

---

## 0. 怎么用这份路线图

每篇论文建议完成：

| 步骤                | 产出                             | 状态标记（`papers.yaml`）             |
| ------------------- | -------------------------------- | --------------------------------------- |
| 1. 下载 PDF/TeX     | `arxiv/<track>/<slug>/`        | `downloading` → `pending` 有源文件 |
| 2. 全文中文译本     | `translations/.../paper_zh.md` | `reading`                           |
| 3. 笔记三件套       | `notes/<track>/<slug>/{QA,critique,note}.md` | `notes_done`              |
| 4. 横向对比（可选） | `comparisons/`                 | `mastered`（可选）                    |

译本约定：按原文结构**人工翻译**（禁止脚本生成正文）；图先转 PNG 再引用；表格改为 Markdown。

进度总览见 [`papers.yaml`](papers.yaml) 的 `status` / `priority` 字段。

---

## 1. 知识依赖（先看图再开读）

```
LDM ──┬── ControlNet ──┬── OminiControl ── EasyControl
      ├── IP-Adapter ──┘         ↑
      ├── GLIGEN                 │
      └── Zero-1-to-3 ── CAT3D   │
                                 │
SD3 (Rectified Flow / MM-DiT) ───┴── SANA / Qwen-Image / Z-Image / BAGEL
              │
              └── Wan ──┬── MotionCtrl ── CameraCtrl ── Free-Form Motion
                        │         │              │
                        │         ├── MIMO / DragAnything
                        │         └── SynCamMaster ── ReCamMaster
                        │                    │
                        │              GEN3C ─┘
                        ├── FullDiT
                        ├── Thinking with Video
                        └── Diffusion Forcing ── Self Forcing
```

**读法原则**

1. **基座先于插件**：LDM → SD3 → Wan，再读控制/多视角/流式。
2. **UNet 控制先于 DiT 控制**：ControlNet / IP-Adapter → OminiControl / EasyControl。
3. **单相机控制先于多相机**：MotionCtrl / CameraCtrl → SynCamMaster / ReCamMaster / GEN3C。
4. **全序列生成先于流式**：有 Wan 直觉后再读 Forcing 系列。

---

## 2. 分阶段计划

### Phase A — 图像生成与可控生成基线（约 1.5–2 周）

| 顺序 | 论文                                        | 优先级 | 精读时必答的问题                                            |
| ---- | ------------------------------------------- | ------ | ----------------------------------------------------------- |
| A1   | **LDM**                               | P0     | 为何进 latent？UNet 条件怎么进？与 pixel diffusion 差在哪？ |
| A2   | **SD3 / Rectified Flow Transformers** | P0     | Flow vs DDPM？MM-DiT 双流如何对齐文本与图像？               |
| A3   | **ControlNet**                        | P0     | 零卷积为何有效？可训练副本与主骨干如何共享？                |
| A4   | **IP-Adapter**                        | P0     | 图像提示与文本提示如何解耦？与 ControlNet 正交点？          |
| A5   | **GLIGEN**                            | P1     | Grounding token 如何注入？开放集布局控制极限在哪？          |
| A6   | **OminiControl**                      | P0     | DiT 上**极简通用控制**相对 ControlNet 改了什么？      |
| A7   | **EasyControl**                       | P0     | LoRA 条件注入 + KV Cache 如何同时解决多条件与效率？         |

**本阶段对比产出**：[`comparisons/base_models/control_injection.md`](comparisons/base_models/control_injection.md)
（ControlNet vs IP-Adapter vs GLIGEN vs OminiControl vs EasyControl）

### Phase B — 高效 / 统一图像基座 + 视频基座（约 1–1.5 周）

| 顺序 | 论文                          | 优先级 | 精读时必答的问题                                   |
| ---- | ----------------------------- | ------ | -------------------------------------------------- |
| B1   | **SANA**                | P1     | 线性注意力如何换高分辨率？质量-速度权衡？          |
| B2   | **Qwen-Image**          | P1     | 文本渲染与编辑的数据/课程学习设计？                |
| B3   | **Z-Image**             | P1     | 单流 DiT vs MM-DiT 双流：参数效率从哪来？          |
| B4   | **BAGEL**               | P1     | 统一理解-生成的涌现阶段？对齐你视频生成目标的点？  |
| B5   | **Wan**                 | P0     | 视频 VAE、时空 DiT、数据与评测：可复现的关键设计？ |
| B6   | **Thinking with Video** | P2     | **视频当推理媒介**对生成系统的启发（可略读） |

**本阶段对比产出**：[`comparisons/base_models/image_video_foundations.md`](comparisons/base_models/image_video_foundations.md)

### Phase C — 几何 / 运动控制（约 1.5–2 周）

| 顺序 | 论文                               | 优先级 | 精读时必答的问题                                     |
| ---- | ---------------------------------- | ------ | ---------------------------------------------------- |
| C1   | **MotionCtrl**               | P0     | 相机运动 vs 物体运动如何解耦？条件表示是什么？       |
| C2   | **CameraCtrl**               | P0     | Plücker 嵌入相对纯 pose 数值条件强在哪？            |
| C3   | **Free-Form Motion Control** | P0     | 6D 姿态联合控制与合成数据如何补标注缺口？            |
| C4   | **MIMO**                     | P1     | 空间分解对角色可控性的帮助？                         |
| C5   | **DragAnything**             | P1     | Entity representation 如何泛化到**任意物体**？ |
| C6   | **LeviTor**                  | P1     | 相对深度如何消解 2D 拖拽的出平面歧义？           |
| C7   | **3DTrajMaster**             | P1     | 多实体 6DoF 轨迹如何注入且不伤视频先验？         |
| C8   | **MotionCanvas**             | P1     | 镜头设计意图如何译成扩散可吃的时空条件？         |
| C9   | **UCPE**                     | P1     | Relative Ray Encoding 相对 Plücker 强在哪？     |
| C10  | **EchoMotion**               | P1     | 显式人体运动模态如何补像素目标的运动缺陷？       |
| C11  | **MotionCrafter**            | P1     | 4D VAE 如何联合稠密几何与 scene flow？           |
| C12  | **FullDiT**                  | P1     | 多任务 + Full Attention 的统一接口是什么？           |

**本阶段对比产出**：[`comparisons/geometry_control/camera_vs_object.md`](comparisons/geometry_control/camera_vs_object.md)

### Phase D — 多视角一致性 + 流式长视频（约 1.5–2 周）

| 顺序 | 论文                        | 优先级 | 精读时必答的问题                                                   |
| ---- | --------------------------- | ------ | ------------------------------------------------------------------ |
| D1   | **Zero-1-to-3**       | P0     | 相机相对变换如何条件化？零样本新视角极限？                         |
| D2   | **CAT3D**             | P1     | 多视角扩散如何接到 3D 重建/生成管线？                              |
| D3   | **SynCamMaster**      | P0     | 多相机同步的一致性约束如何实现？                                   |
| D4   | **ReCamMaster**       | P0     | 单视频 → 可控相机重拍：几何与外观如何平衡？                       |
| D5   | **GEN3C**             | P0     | **3D-informed + world-consistent**相对纯 2D 视频控制差在哪？ |
| D6   | **Diffusion Forcing** | P0     | Next-token 与 full-sequence diffusion 如何统一？                   |
| D7   | **Self Forcing**      | P0     | 训练-测试 gap 的根因与 Self Forcing 对策？                         |

**本阶段对比产出**：

- [`comparisons/multi_view/consistency_stack.md`](comparisons/multi_view/consistency_stack.md)
- [`comparisons/streaming/forcing_family.md`](comparisons/streaming/forcing_family.md)

---

## 3. Faceglot 课题阅读优先级（画面译制 / SIGGRAPH 2026）

> 与上面 Phase A–D（可控视频生成通识）并行。目标是尽快把 **任务差**（编辑 vs 生成、只改嘴 vs 改说话脸、画面 vs 音色）写进 related work，而不是再刷一遍 talking-head 编年史。
> TeX 落盘：`arxiv/visual_dubbing/`、`arxiv/audio_driven/`、`arxiv/base_models/avcontrol/`。

```
mask inpaint          mask-free 仍只改嘴           全身跟音频重生
Wav2Lip ── LatentSync ── OmniSync ── X-Dub ── Faceglot（改语言相关整张说话脸）
   │                         │              │
   └── MuseTalk / VideoReTalking            └── InfiniteTalk / LongCat-Avatar
                                                    （生成，不锁原片走位）

跨语言表情（生成，非剪辑）：DisentTalk / Polyglot
姿态造数据：LTX-2 + AVControl（去嘴/眼/眉关键点）
可选音色（正交）：Neural Dubber → FunCineForge
```

### 第一档 — 立刻精读（决定课题能不能立住）

| 顺序 | 论文 | track/slug | 精读时必答 |
|------|------|------------|------------|
| F1 | **X-Dub** | `visual_dubbing/xdub` | 生成器 G 如何造对？输入输出除嘴外是否逐帧相同？编辑器如何注入源视频？与本仓库 Wan2.2-TI2V-5B 公开版差在哪？ |
| F2 | **LatentSync** | `visual_dubbing/latentsync` | mask inpaint 的信息泄漏从哪来？TREPA / SyncNet 改进能否证明「表情自然」？ |
| F3 | **OmniSync** | `visual_dubbing/omnisync` | 同团队 mask-free 口型同步：配对数据怎么来的？相对 X-Dub bootstrap 差在哪？ |
| F4 | **InfiniteTalk** | `visual_dubbing/infinitetalk` | 稀疏帧 V2V 改了头/身/表情，还算不算译制剪辑？镜头/遮挡/道具还在吗？ |
| F5 | **LongCat-Avatar 1.5** | `audio_driven/longcat_avatar_15` | Whisper-large 怎么接 DiT？为何不能当 Faceglot 编辑器？人因评测可抄什么？ |
| F6 | **LTX-2 + AVControl** | `base_models/{ltx2,avcontrol}` | 姿态 LoRA 能否在抹掉嘴/眼/眉点后仍锁身体？音频条件会不会把说话脸写死？ |

### 第二档 — 写 related work / 消融设计

| 顺序 | 论文 | track/slug | 精读时必答 |
|------|------|------------|------------|
| F7 | **KeySync** | `visual_dubbing/keysync` | LipLeak 怎么定义？我们要的是「故意改眉眼」，如何把 leakage 指标反过来用？ |
| F8 | **MuseTalk / VideoReTalking** | `musetalk`, `videoretalking` | 实验表里必须报的开源 inpaint 基线；SIGGRAPH Asia 口型编辑写法 |
| F9 | **DisentTalk / Polyglot** | `disenttalk`, `polyglot` | 跨语言/多语说话脸证据链：他们改的是生成 avatar 还是原片？CHDTF 能否当中文评测？ |
| F10 | **FunCineForge / Neural Dubber** | `funcineforge`, `neural_dubber` | 音色模块的任务边界；CineDub-CN 画面能否当中文源？v1 为什么不联合训练？ |
| F11 | **Wan-S2V / OmniHuman-1.5** | `wan_s2v`, `omnihuman15` | 编辑器基座与「语义表情」生成对照；Whisper vs Wav2Vec |

### 第三档 — 背景，不必全文精读

| 论文 | 用法 |
|------|------|
| Wav2Lip / SadTalker / EMO / DiffTalk / Diff2Lip | 引用 + 实验表；SyncNet 评测起源 |
| OpenHumanVid | 中文人体源语料与骨架标注 |
| InstructDubber | FunCineForge 的配音基线，画面课题可略 |
| AnyTalker / InterActHuman | 已有笔记；多人生成，v1 刻意不用 AFCA |

**建议读法**：F1–F3 连着读（同一条「只改嘴」进化链）→ F4–F5 对照生成路线 → F6 再动手造 $V_{\mathrm{syn}}$ → F7–F9 补主张与指标 → F10 仅在做音色附录时展开。

---

## 4. 建议节奏（可按周调整）

| 周次   | 焦点                      | 交付                                |
| ------ | ------------------------- | ----------------------------------- |
| W1     | Phase A1–A4              | LDM/SD3/ControlNet/IP-Adapter 译本  |
| W2     | Phase A5–A7 + 控制对比表 | DiT 控制线打通                      |
| W3     | Phase B（含 Wan）         | 基座横向对比一页                    |
| W4     | Phase C                   | 运动控制对比一页                    |
| W5–W6 | Phase D                   | 多视角 + Forcing 系列；总览脑图更新 |

Task B / foundations（`translations/foundations/`、`arxiv/foundations/`、`notes/foundations/<slug>/`、`code/foundations/`）视为**前置已完成**，主进度以 Phase A–D 为准；需要时回看 Transformer / 动态模块直觉即可。

---

## 5. 译本质量标准（自检）

每篇 `paper_zh.md` 至少做到：

1. **人工撰写**：正文由 Agent / 人直接写，不用代码/API 批量生成。
2. **结构对齐原文**：章节标题与原文对应，内容完整翻译（非摘要式改写）。
3. **图**：先用 `scripts/figures_to_png.py` 转为 PNG，再 Markdown 引用，并保留图注。
4. **表**：改写成 Markdown 表格，数字与原文一致。
5. **公式**：保留 LaTeX，符号与原文一致。
6. **可读**：专有名词可保留英文，首次出现给中文说明。

---

## 6. 下一步（仓库就绪后立刻做）

```bash
# 1) 下载论文 PDF / TeX
python scripts/download_catalog.py --track base_models
python scripts/download_catalog.py --all

# 2) 需要引用的图 → PNG
python scripts/figures_to_png.py arxiv/base_models/ldm/extracted/img/...

# 3) 人工撰写第一篇译本
# translations/base_models/ldm/paper_zh.md
```
