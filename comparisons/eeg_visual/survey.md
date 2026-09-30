# EEG 生成图像 / 视频 / 3D：顶会顶刊调研、创新分类与空白点

> 调研范围：非侵入式 EEG → 图像 / 视频 / 3D（感知重建为主）。fMRI 工作（MinD-Vis、Mind-Video、MindEye）只作对照，不并入主表。
> 时间：2017–2026；优先 CVPR / ICCV / NeurIPS / ICLR / AAAI / ACM MM，并覆盖 NeuroImage、Scientific Data、IEEE TNNLS/TNSRE、ISMAR 上的关键数据与方法文。
> 结论先行：**2023 年后这条线几乎被**EEG 编码器 + CLIP 对齐 + 冻结扩散先验**收编；视觉质量多半来自生成器，不来自 EEG 解码出的细节。真正还没做透的，是想象解码、跨被试基础模型、现代视频 DiT 时序对齐，以及证明 EEG 贡献超过类别标签。**

---

## 0. 一句话地图

| 模态 | 阶段 | 代表工作 | 当前天花板 |
|------|------|----------|------------|
| 图像 | 2017–2022 分类/GAN | Spampinato CVPR'17, Brain2Image MM'17 | 类别可分，像素不可信 |
| 图像 | 2023– 扩散重建 | DreamDiffusion, NICE ICLR'24, ATM NeurIPS'24, CognitionCapturer AAAI'25 | CLIP 空间里语义像样，实例级细节弱 |
| 视频 | 2024– | EEG2Video NeurIPS'24, EEGMirror ICCV'25, NEED NeurIPS'25 | 2 秒 / ~3 fps / 6 帧，仍是 inflated SD 1.4 |
| 3D | 2024– | Neuro-3D CVPR'25；Mind2Matter / 3D-Telepathy arXiv | 单物体点云或**文本中介再 3DGS**，场景级弱 |

EEG 相对 fMRI 的卖点是：**便携、便宜、毫秒级时间分辨率**。短板是：**空间分辨率差、SNR 低、跨被试变异大、公开配对数据少一个数量级**。所以图像线先证明**语义能对齐**，视频线才开始用时间分辨率，3D 线则刚立任务。

---

## 1. 该看哪些顶会顶刊

### 1.1 计算机视觉 / 机器学习（近三年主战场）

| 会议 | 为什么重要 | 本方向代表 |
|------|------------|------------|
| **NeurIPS** | 生成 + 神经解码交叉最密 | EEG2Video (2024), ATM (2024), NEED (2025) |
| **ICLR** | 表征与对比学习 | NICE (2024)；NECOMIMI 曾投 2025 |
| **CVPR** | 3D / 视觉重建 | Neuro-3D (2025), D²-FOSA (2026) |
| **ICCV** | 数据/自监督扩展 | EEGMirror (2025) |
| **AAAI** | 多模态条件 | CognitionCapturer (2025) |
| **ACM MM** | 早期生成 | Brain2Image (2017) |

### 1.2 神经科学 / BCI / 数据

| 出处 | 角色 |
|------|------|
| **NeuroImage** | THINGS-EEG2（Gifford et al., 2022）——当前图像检索/重建的事实基准 |
| **Scientific Data / Nature Sci Data** | THINGS-EEG（Grootswagers et al., 2022） |
| **IEEE TNNLS / TNSRE / TBME / JNE** | 大量 EEG 分类、少通道、跨被试；生成质量通常不如 CV 顶会，但实验更**BCI 味** |
| **ISMAR** | EEG2Gaussian (2025)：VR 场景 + 3DGS |

### 1.3 不要和 fMRI 混读

fMRI 图像/视频已经到 MindEye2、MinD-Vis、Mind-Video，像素与语义都明显强于 EEG。EEG 论文常把 fMRI 当动机（**fMRI 太慢/太贵**），但**评测数字不可直接对比**：刺激范式、时间窗口、被试规模完全不同。EEG 的合理对照是 **THINGS-EEG2 上的 NICE/ATM** 与 **SEED-DV 上的 EEG2Video 家族**，不是 NSD 上的 MindEye。

---

## 2. 数据集（这条线的真正瓶颈）

| 数据集 | 年份 | 被试 | 刺激 | 通道 | 用途 | 隐患 |
|--------|------|------|------|------|------|------|
| **ImageNet-EEG** (Palazzo/Spampinato) | 2017 | 6 | 40 类 × 50 图，0.5 s | 128 | 早期分类/生成 | **组块设计**：同类连放，模型可能学到时间结构而非视觉内容 |
| **ThoughtViz** | 2018 | 23 | 想象数字/字符/物体 | 14 | 想象生成 | 刺激极简，难迁移到自然图像 |
| **THINGS-EEG** | 2022 | 50 | 1854 概念 / 22248 图，RSVP | 64 | 大规模概念空间 | RSVP 很快，试次重叠 |
| **THINGS-EEG2** | 2022 | 10 | 16540 训练图 + 200 测试图，5 Hz RSVP | 63 | **图像 zero-shot 检索/重建主基准** | 非自然观看；被试少 |
| **Alljoined / Alljoined-1.6M** | 2024 | 8 | ~1 万图/人，MS-COCO 类 | 64 | 消费级/跨被试 | 硬件与 THINGS 不同，ATM 一类复杂模型变脆 |
| **SEED-DV** | 2024 | 20 | 1400 段 2 s 视频，40 概念 | 62 @ 200 Hz | **视频主基准** | 概念仍偏粗（40 类），时长极短 |
| **EEG-3D** | 2025 | 12 | 72 类 Objaverse 物体，旋转视频+静图+3D | — | **3D 主基准** | 单物体、渲染刺激，不是真实场景 |

额外观察：

- 图像线从 ImageNet-EEG 迁到 THINGS-EEG2，是质量跃迁，也是对早期**83% 分类准确率**叙事的纠偏。
- 视频几乎只有 SEED-DV；3D 几乎只有 EEG-3D。**跟一篇新方法，数据贡献往往比模块贡献更值钱。**
- RSVP（5 Hz 闪图）和**看 2 秒视频**测的不是同一种神经过程。跨任务迁移（NEED 宣称图像↔视频）因此既重要又容易刷到虚假泛化。

---

## 3. 图像：论文与创新

### 3.1 阶段一：分类流形（2017）

**Deep Learning Human Mind for Automated Visual Classification** — Spampinato et al., **CVPR 2017**

- 问题：用 EEG 学视觉类别流形，再把 CNN 图像投到该流形上做分类。
- 方法：RNN 读 128 通道 EEG；报告 ~83% 40 类准确率。
- 定位：不是生成，但是后续 Brain2Image / DreamDiffusion 的数据与叙事源头。
- 异味：ImageNet-EEG 组块设计被后续社区反复质疑；高准确率可能部分来自时间伪迹。

### 3.2 阶段二：VAE / GAN 直接生成（2017–2022）

**Brain2Image** — Kavasidis et al., **ACM MM 2017**

- 创新：LSTM 压 EEG → VAE/GAN 解码图像。第一次认真做**脑信号出图**。
- 局限：类别可辨、实例不像；GAN 模式崩塌。

后续同类：ThoughtViz（想象刺激 + GAN）、EEG2IMAGE（triplet / 对比 + cGAN）、若干 IEEE 期刊上的条件 GAN / 注意力 GAN。共同公式：

```
EEG encoder (LSTM/CNN) → class-discriminative z → GAN/VAE decoder
```

这一阶段的真问题是 **EEG 有没有视觉信息**；生成器只是把 z 可视化。

### 3.3 阶段三：CLIP 对齐 + 冻结扩散（2023–）

这是当前主范式。骨架几乎统一：

```
EEG  →  encoder  →  对比学习对齐 CLIP  →  diffusion prior  →  冻结 SD / SDXL
```

| 论文 | 出处 | 相对 NICE/DreamDiffusion 的 delta |
|------|------|-------------------------------------|
| **DreamDiffusion** | arXiv 2023 | MAE 式 masked EEG pretrain，再对齐 CLIP、微调 SD 1.5。仍用 ImageNet-EEG、同被试 Subject 4。 |
| **NICE** | **ICLR 2024** | 不做生成；TSConv + 可选 SA/GA，与冻结 CLIP 做对称对比。THINGS-EEG2 上 200-way zero-shot Top-1 **15.6%**。把**EEG 图像检索**立成可复现任务。 |
| **ATM / EEG Embeddings + Guided Diffusion** | **NeurIPS 2024** | Adaptive Thinking Mapper 把不同源 EEG 投到 CLIP 子空间；两阶段：high-level CLIP prior + 模糊低层图 + caption，再条件扩散。检索/重建 SOTA；可迁 MEG。 |
| **BrainDreamer** | arXiv 2024 | mask 三重对比（EEG–text–image）；EEG adapter 注入 SD；允许额外文本控制。语言当推理桥。 |
| **CognitionCapturer** | **AAAI 2025** | 模态专家：图 / 文 / 深度 分别从 EEG 抽特征，diffusion prior → CLIP，**不微调** SDXL-Turbo + 多 IP-Adapter。强调 EEG 里有**超出 RGB**的信息。 |
| **NECOMIMI** | ICLR 2025 投稿 / arXiv | NERV 编码器；提出 **CAT Score** 专评 EEG-to-image。关键发现：生成常是抽象风景而非特定物体——EEG 更像粗语义。 |
| **D²-FOSA** | **CVPR 2026** | FOMamba 显式建模节律；双向 EEG–image dual diffusion。THINGS-EEG 上 FID 相对 MB2C 降 20+。 |
| **ENIGMA** | arXiv 2026 | 攻击 ATM 过重：15 分钟、<1% 参数，在消费级 Alljoined 上更稳。暗示复杂编码器在低 SNR 上过拟合。 |

### 3.4 图像线的创新其实只有三类

1. **更好的 EEG encoder**（NICE 的空间注意/图注意，ATM 的自适应映射，D²-FOSA 的频域 Mamba，DreamDiffusion 的 MAE）。
2. **更丰富的对齐目标**（CLIP 图、CLIP 文、深度、多层次 caption、模糊低层图）。
3. **更省事的生成器接入**（微调 SD → 冻结 SD + adapter/IP-Adapter/ControlNet）。

**没有出现****从 EEG 恢复精确轮廓/纹理**的证据。NECOMIMI 的抽象化现象、以及**换更强 SD 画质就涨、换 EEG 编码器涨得少**的模式，都指向同一判断：当前图像质量上限由先验决定。

---

## 4. 视频：论文与创新

动机成立：fMRI ~0.5 Hz，跟不上视觉动态；EEG ~200–1000 Hz。若时间分辨率是 EEG 的核心优势，视频才是它该打的仗。

### 4.1 奠基：**EEG2Video** — Liu et al., **NeurIPS 2024**

三项贡献（方法 < 数据）：

1. **SEED-DV**：20 人 × 1400 段 2 s 视频 × 40 概念。填了 EEG–video 配对空白。
2. 元信息标注（颜色、是否动态、是否有人）→ EEG-VP 基准。
3. 方法：Seq2Seq 密对齐低层动态 + 语义预测器 + **DANA**（dynamic-aware noise-adding，把快/慢动态写进扩散噪声日程）+ inflated Stable Diffusion（Tune-A-Video 风格）。

结果量级：40 类语义准确率 ~15.9%，2-way ~79.8%，SSIM ~0.256。作者拿来和 fMRI 视频重建的 SSIM 比，数字接近，但刺激与时长不可比。

### 4.2 后续：同一数据集上的模块战

| 论文 | 出处 | delta vs EEG2Video | 仍未动的地方 |
|------|------|---------------------|--------------|
| **EEGMirror** | **ICCV 2025** | 神经量化 + montage-agnostic PE（MAPE），用**野外**不同导联 EEG 做 masked 自监督；再多模态对比 + 微调 SD。语义 82.1% / SSIM 0.261。 | 生成器仍是 SD；收益主要来自预训练数据 |
| **NEED** | **NeurIPS 2025** | 个体适应模块 + 双通路（低层动态 / 高层语义）；宣称 **跨被试、跨任务（视频↔图像）** zero-shot。跨被试保住 ~93% 被试内分类、~92% 重建质量；无微调转图像 SSIM 0.352。 | 需独立复现；跨任务数字对 RSVP vs 视频范式差很敏感 |
| **DynaMind** | arXiv 2025 | 脑区语义映射 RSM + 时序蓝图 TDA + 双引导重建 DGVR。SEED-DV 上视频/帧准确率 +12.5 / +10.3 pp，SSIM +9.4%，FVMD −19.7%。 | 仍 6 帧、512×288、3 fps、SD v1.4 |
| **MindCine** | arXiv 2026 | 大规模 EEG 基础模型微调；CausalSeq 解低层动态；文+深度辅助；T2V 扩散。明确打**数据少**和**超文本模态**。 | 方向对，基座仍非开源视频 DiT 主流 |

### 4.3 视频线的创新分类

1. **数据立任务**（SEED-DV）——目前最大的真贡献。
2. **时序对齐算子**：Seq2Seq、DANA、temporal blueprint、CausalSeq。这是相对图像线唯一不可约的新问题。
3. **用更多无标注/异构 EEG**：MAPE、foundation EEG model。
4. **跨被试 / 跨任务**（NEED）。

**没有出现**：长视频、高帧率、镜头运动、多物体交互、音画同步、现代 Video DiT（Wan / CogVideoX / Hunyuan）条件注入。现在的**视频**更接近**6 张语义一致的幻灯片**。

---

## 5. 3D：论文与创新

3D 比视频更年轻，且路线分裂。

### 5.1 任务立得最干净：**Neuro-3D** — Guo et al., **CVPR 2025**

- **EEG-3D**：12 人，72 类着色 3D 物体；旋转视频 + 静图 + 文本 + 几何/颜色标注。
- 方法：动态–静态 EEG 融合编码器；embedding 解耦为 **geometry / appearance**；点云扩散出形状，再上色。
- 创新类型：**新任务 + 解耦条件**，不是又一个 CLIP-SD 套壳。
- 局限：刺激是转盘上的单物体，不是自然 3D 场景；**3D 感知**有多少来自旋转视频的 2D 运动能量，需要更硬的对照（静止多视角 vs 连续旋转）。

### 5.2 另外三条分叉（多为 arXiv / 非 CV 顶会）

| 论文 | 3D 表示 | 中介 | 判断 |
|------|---------|------|------|
| **EEG-Driven 3D + style** (arXiv 2024) | NeRF | EEG latent 微调 LDM 当 diffusion prior + style loss | 2D 风格一致性前置，3D 是优化结果 |
| **3D-Telepathy** (arXiv 2025) | NeRF + VSD | 双自注意 EEG encoder；Stable Diffusion 作先验 | 标准 Score Distillation，EEG 当条件 |
| **Mind2Matter** (arXiv 2025) | 3DGS 场景 | **EEG → LLM 文本 → layout-constrained 3DGS** | 最像套娃：3D 质量几乎全是文本-3D 模型的；EEG 只负责 caption |
| **EEG2Gaussian** (ISMAR 2025) | 3DGS VR 场景 | 时频编码器解耦高低层语义 | 应用场景（VR）比物体重建更接近落地 |

### 5.3 3D 线该怎么分类

- **直接 3D 解码**（Neuro-3D 点云扩散）：难，但 EEG 信号与几何有直接损失。
- **2D 先验蒸馏 3D**（NeRF + SDS/VSD）：生成观感好，EEG 监督稀疏。
- **语言瓶颈**（Mind2Matter）：最容易出 demo，也最容易让 EEG 退化成 40 类分类器。

若目标是神经科学（脑如何编码 3D），应走 Neuro-3D 这类几何损失。若目标是 BCI demo，语言瓶颈会赢短期，但论文贡献薄。

---

## 6. 创新分类总表（跨模态）

按**改了哪一层**归类，避免被论文标题带跑。

### I. 任务与数据（最高杠杆）

| ID | 创新 | 代表 | 拥挤度 |
|----|------|------|--------|
| T1 | 把分类做成检索/zero-shot | NICE | 拥挤 |
| T2 | 感知图像重建 | ATM, CognitionCapturer | 拥挤 |
| T3 | 感知视频重建 | EEG2Video 家族 | 中等 |
| T4 | 感知 3D 重建 | Neuro-3D | 稀疏 |
| T5 | **想象 / 无刺激生成** | ThoughtViz（旧、简） | **极稀疏** |
| T6 | 新配对数据集 | SEED-DV, EEG-3D, Alljoined, THINGS-EEG2 | 每篇都值钱 |

### II. EEG 表征

| ID | 创新 | 代表 | 备注 |
|----|------|------|------|
| E1 | 时空卷积 + 通道注意/图注意 | NICE | 成为标配 |
| E2 | Masked signal modeling | DreamDiffusion, EEGMirror | 从 MAE/fMRI MSM 搬来 |
| E3 | 被试 embedding / 个体适应 | ATM, NEED | 跨被试刚开始有效 |
| E4 | Montage-agnostic 位置编码 | EEGMirror MAPE | 能吃异构公开 EEG |
| E5 | 显式频域/节律 | D²-FOSA FOMamba | EEG 该有的归纳偏置，来得晚 |
| E6 | 脑区划分 / 区域聚合 | DynaMind RSM | 有神经科学叙事 |
| E7 | 大规模 EEG 基础模型再对齐 | MindCine, EEGMirror | 正确方向，仍早期 |

### III. 跨模态对齐

| ID | 创新 | 代表 | 风险 |
|----|------|------|------|
| A1 | CLIP 图对比 | NICE 及几乎所有后人 | **主路径，边际收益递减** |
| A2 | 文本 / caption 中介 | BrainDreamer, Mind2Matter | 可控，但易让 EEG 只剩类别 |
| A3 | 多模态专家（图+文+深度） | CognitionCapturer, MindCine | 深度/文来自教师模型，不是脑 |
| A4 | 高低层双通路 | ATM, NEED, EEG2Video | 目前最接近**不只是分类** |
| A5 | 几何–外观解耦 | Neuro-3D | 3D 特有，值得迁到视频运动–语义 |

### IV. 生成器与条件注入

| ID | 创新 | 代表 | 风险 |
|----|------|------|------|
| G1 | GAN/VAE | Brain2Image | 过时 |
| G2 | 冻结 LDM + adapter | DreamDiffusion → CognitionCapturer | 画质虚高 |
| G3 | Inflated SD / Tune-A-Video | EEG2Video, DynaMind | 视频线仍停在 2023 基座 |
| G4 | DANA / 时序噪声 / temporal blueprint | EEG2Video, DynaMind | **视频特有真问题** |
| G5 | ControlNet / IP-Adapter / 双扩散 | ATM, D²-FOSA | 工程增量 |
| G6 | 点云扩散 / NeRF+VSD / 3DGS | Neuro-3D 等 | 3D 表示之争未结束 |
| G7 | **现代 Video DiT（Wan 等）条件** | — | **空白** |

---

## 7. 这条线的**异味**与方法学陷阱

1. **CLIP + SD 幻觉**  
   生成图好看，不等于 EEG 解码对。消融必须包含：只用类别标签条件 SD、随机 EEG、时间打乱 EEG。很多论文不报这三项。若三类对照与全文方法接近，贡献就只是分类器。

2. **ImageNet-EEG 组块泄漏**  
   同类刺激连续呈现，时序模型可走捷径。THINGS-EEG2 的 RSVP + 概念打乱更可信，但仍不是自然观看。

3. **SSIM 对扩散输出几乎无意义**  
   扩散样本与 GT 像素不对齐是常态。应用 CLIP similarity、2-way identification、检索 Top-k，以及 NECOMIMI 的 CAT Score；像素指标只作辅。

4. **被试内 vs 跨被试**  
   早期 DreamDiffusion 只报 Subject 4。NICE/ATM 仍以被试内为主。BCI 落地必须跨人、跨会话、跨设备。NEED / EEGMirror / SCORE 刚碰到这个问题。

5. **教师模态不是脑模态**  
   CognitionCapturer 的深度/文本来自 DepthAnything / BLIP2。它们是**用视觉教师给 EEG 加监督**，不是**EEG 里解码出了深度**。论文可以做，但叙事不能说成脑内已有显式深度图。

6. **感知 ≠ 想象**  
   fMRI 上已有证据：视觉重建 SOTA 迁不到 imagery（如 MIRAGE / NSD-Imagery）。EEG 几乎全是**看着刺激重建刺激**。标题里的 mind / thought / telepathy 多数名不副实。

---

## 8. 还能做什么（按值得做的程度）

下面区分**跟进性小改**和**值得开题**。与本仓库（视频生成 / 音频驱动 / 可控 DiT）的接口单独标出。

### 8.1 不建议再做（拥挤且增量薄）

- 再换一个 EEG encoder，接到 SD 1.5/SDXL，在 THINGS-EEG2 上刷 CLIP similarity。
- 在 SEED-DV 上把 inflated UNet 换成另一个 2022–2023 视频微调术，帧数仍是 6。
- EEG → caption → 现成 text-to-3D，没有几何监督或可控实验证明 caption 来自 EEG 而非分类。

### 8.2 高价值、可做的研究方向

**P0 — 证明信息量：EEG 到底贡献了什么**

- 强制对照：class-only SD、shuffled EEG、相位随机、仅枕叶 vs 全脑。
- 把可解码属性拆开：类别、颜色、空间布局、运动方向、物体数。EEG2Video 的元信息标注是对的，但没有做成标准 disentangle benchmark。
- 神经科学可解释：时间窗（P1 / N170 / P300）、频带、电极贡献。NICE 做过一轮，生成论文大多不碰。

**P0 — 想象与意图，而不是刺激重建**

- 看过 vs 想象 vs 记忆 vs 语言描述同一物体，四条件 EEG。
- 这才是 BCI；也是最难的，因为 SNR 更差、无像素 GT。评测应走检索/属性/人类判别，而不是 FID。
- ThoughtViz 之后几乎真空。谁做出自然图像级 imagery，就是 seed 论文级。

**P0 — 把视频基座从 SD 1.4 换到现代 Video DiT，并做密时序条件**

- 本仓库已有 Wan / 音频注入经验。EEG 条件可以走与 `audio_proj` 同类的 **时空 token 注入**（按 VAE 时间格对齐），而不是 CLIP 向量打在 cross-attn 上。
- 必须回答：EEG 的毫秒级动态如何对齐到 latent fps（例如 16 fps 的 2 秒 = 32 latent 帧）？DANA / CausalSeq 是雏形，还没碰到 DiT 的 3D attention。
- 评测加运动：FVMD、光流一致性、动作分类，而不是只报 SSIM。

**P1 — 跨被试 EEG foundation + 极少校准**

- 先在异构无标签 EEG 上 pretrain（montage-agnostic，EEGMirror 已做），再在 SEED-DV / THINGS 上对齐。
- 目标：新被试 5–10 分钟校准达到被试内 90%+。NEED 声称接近，需要更硬的 LOSO 与跨数据集。
- 与语音/视觉基础模型一样，**表征论文可以不生成**，生成只是下游。

**P1 — 统一条件：一个 EEG encoder，图像 / 视频 / 3D / 音频多头**

- NEED 做了图像↔视频；没有人做**同一表征驱动 Wan + 3DGS + vocoder**。
- 避免三个独立项目。解耦头：语义、运动、几何、音色。
- 和本仓库 HarmonizedVisualAudio 的接口：EEG 作**意图/情绪/节奏**条件，语音作内容条件，二者分层注入——比纯 EEG 出完整视频更可发表、也更像产品。

**P1 — 4D / 动态 3D**

- Neuro-3D 是静态物体；SEED-DV 是 2D 视频。中间空白是：**EEG → 随时间变形的 3D**（旋转的物体已有，交互/形变没有）。
- 数据要自己采：EEG + 多相机或 EEG + 已知 3D 资产的 6DoF 轨迹。

**P2 — 评测与反作弊基础设施**

- 统一 THINGS-EEG2 / SEED-DV / EEG-3D 的官方 split、跨被试协议、class-only 下限。
- CAT Score 类语义指标 + 属性解码雷达图。方法论论文可以中 NeuroImage / NeurIPS Datasets & Benchmarks。

**P2 — 低通道、实时、闭环**

- ENIGMA / 8 通道工作说明消费级可行。
- 流式：不是离线 2 秒一块，而是因果窗口 + 低延迟 DiT 蒸馏。
- 闭环：生成结果再给被试看，测是否能用 EEG 做**选这个/改颜色**——从重建变成控制。

### 8.3 和本仓库最贴的三条具体题目

1. **EEG-conditioned Wan**：把 AnyTalker / InteractHuman 里音频 token 的位置，换成或并上 EEG 时空 token；SEED-DV 上对比 CLIP-only vs dense temporal tokens。核心科学问题是时序对齐，不是再训一个 CLIP 头。
2. **音画脑三条件说话人视频**：语音驱动口型与身份，EEG 驱动表情/情绪/意图。比**脑直接出整段电影**诚实，也比纯音频驱动多一个可控轴。
3. **属性级 EEG 控制，而非像素重建**：从 EEG 解颜色、运动方向、是否有人，再作为 ControlNet/相机/物体轨迹条件。绕开像素 GT 不足，直接进入可控生成叙事。

---

## 9. 推荐阅读顺序

1. NICE (ICLR 2024) —— 把任务和数据集读懂，建立对准确率量级的直觉。  
2. ATM (NeurIPS 2024) —— 看生成范式如何从检索长出来。  
3. EEG2Video (NeurIPS 2024) + SEED-DV 主页 —— 视频问题定义。  
4. Neuro-3D (CVPR 2025) —— 3D 问题定义，对照语言瓶颈路线。  
5. EEGMirror / NEED —— 看社区把**数据少、跨被试**当真正问题后怎么走。  
6. 对照一篇 fMRI：Mind-Video 或 MindEye2 —— 知道上限，避免把 EEG demo 当成同等信息量。

综述可作索引（不必精读）：

- *A Survey on Bridging EEG Signals and Generative AI* (arXiv:2502.12048)
- *Comprehensive Review of EEG-to-Output* (arXiv:2412.19999)

---

## 10. 总判

- **图像**：方法学已收敛，顶会还在收**encoder / 频域 / 多模态教师**增量；除非做想象解码或硬核信息量证明，否则难成为 seed。
- **视频**：任务刚立住，生成器落后视频社区两年；**密时序条件 + 现代 DiT** 是最适合本仓库的切入点。
- **3D**：Neuro-3D 把直接几何解码立住了；语言中介 3D 不建议跟。下一步是场景级与 4D，都先缺数据。
- **共同空白**：跨被试基础模型、标准化评测、感知→想象、证明 EEG ≠ 类别标签。

若只做一件事：在 SEED-DV 上用现代视频基座做 **dense EEG–latent 时序对齐**，并把 class-only / shuffle 对照写成主表，而不是附录。
