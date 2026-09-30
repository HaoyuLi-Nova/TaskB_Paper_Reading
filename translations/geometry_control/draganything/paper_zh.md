# DragAnything：用 Entity Representation 对任意物体做运动控制

**作者：** Weijia Wu$^{1,2,3}$、Zhuang Li$^{1}$、Yuchao Gu$^{3}$、Rui Zhao$^{3}$、Yefei He$^{2}$、David Junhao Zhang$^{3}$、Mike Zheng Shou$^{3\dagger}$、Yan Li$^{1}$、Tingting Gao$^{1}$、Di Zhang$^{1}$

**机构：** $^{1}$快手科技；$^{2}$浙江大学；$^{3}$新加坡国立大学 Show Lab

**贡献说明：** $\dagger$通讯作者

**出处：** ECCV 2024（arXiv:2403.07420）

**arXiv：** [2403.07420](https://arxiv.org/abs/2403.07420)

**项目页：** https://weijiawu.github.io/draganything_page/

**代码：** https://github.com/showlab/DragAnything

**原文 TeX：** [`arxiv/geometry_control/draganything/extracted/arxiv.tex`](../../../arxiv/geometry_control/draganything/extracted/arxiv.tex)

---

## 摘要

我们提出 **DragAnything**：用 entity representation（实体表示）在可控视频生成中实现对任意物体的运动控制。

相对已有运动控制方法，DragAnything 有几点优势。第一，基于轨迹的交互更友好：获取 mask、depth map 等其他引导信号通常费力，用户只需画一条线（轨迹）。第二，entity representation 是一种开放域 embedding，能表示任意物体，从而控制包括背景在内的多样实体运动。第三，该表示支持同时、分别控制多个物体的运动。

大量实验表明，DragAnything 在 FVD、FID 与 User Study 上达到 state-of-the-art；尤其在物体运动控制上，人工投票超出先前方法（如 DragNUWA）$26\%$。

项目页：https://weijiawu.github.io/draganything_page/

**关键词：** Motion Control；Controllable Video Generation；Diffusion Model

---

![图 1. 与先前工作的对比。(a) 先前工作（MotionCtrl、DragNUWA）通过拖拽像素点或像素区域实现运动控制。(b) DragAnything 通过操控对应的 entity representation，实现更精确的实体级运动控制。](../../../arxiv/geometry_control/draganything/extracted/images/motivation.png)

---

## 1. 引言

视频生成近期进展显著，Imagen Video、Gen-2、PikaLab、SVD、SORA 等引起社区广泛关注。然而，可控视频生成尽管至关重要，进展相对更慢。与可控静态图像生成不同，可控视频生成不仅要操控空间内容，还要精确控制时间上的运动，因此更复杂。

基于轨迹的运动控制已被证明是可控视频生成中用户友好且高效的方案。相对 mask 或 depth map 等引导信号，画一条轨迹更简单、更灵活。早期基于轨迹的工作用光流或循环神经网络来控制物体运动。代表性工作 DragNUWA 把稀疏笔画编码到稠密 flow 空间，再以此作为引导信号控制物体运动。类似地，MotionCtrl 把每个物体的轨迹坐标直接编码成向量图，用该向量图作为条件控制物体运动。这些工作对可控视频生成贡献很大。但一个重要问题被忽略了：*目标上的单个点真的能代表该目标吗？*

显然，单个像素点无法代表整个物体，如图 2(a)–(b) 所示。因此，拖拽单个像素点未必能精确控制它所对应的物体。如图 1：给定星空中某颗星上一个像素的轨迹，模型可能分不清要控制的是这颗星还是整片星空，它只是在拖与该像素相关的区域。

要解决这一问题，需要澄清两个概念：1）**拖什么（What entity）**：识别要拖拽的具体区域或实体。2）**怎么拖（How to drag）**：如何只拖选中区域，即把需要拖拽的前景与背景分开。对第一点，交互式分割是高效方案：例如在首帧用 SAM，可方便选出要控制的区域。相对而言，第二个技术问题更具挑战。为此，本文提出一种新的 Entity Representation，以对视频中任意实体做精确运动控制。

已有工作表明，用 latent feature 表示对应物体是有效的。AnyDoor 用 DINO v2 的特征做物体定制；VideoSwap 与 DIFT 用扩散模型的特征做视频编辑。受此启发，我们提出 DragAnything：用扩散模型的 latent feature 表示每个实体。如图 2(d)，根据实体 mask 的坐标索引，可从首帧的 diffusion feature 中提取对应语义特征，再用这些特征表示该实体；通过操控对应 latent feature 的空间位置，实现实体级运动控制。

本文以 SVD 为基座模型。训练 DragAnything 需要视频数据，以及运动轨迹点与首帧的实体 mask。为获得所需数据与标注，我们用视频分割基准训练 DragAnything：用首帧每个实体的 mask 提取该实体中心坐标，再用 Co-Tracker 预测该点的运动轨迹，作为实体运动轨迹。

主要贡献如下：

- 为基于轨迹的可控生成提供新认识：揭示像素级运动与实体级运动的差异。
- 不同于拖像素范式，我们提出 DragAnything，用 entity representation 实现真正的实体级运动控制。
- DragAnything 在 FVD、FID 与 User Study 上达到 SOTA，运动控制人工投票超出先前方法 $26\%$。DragAnything 支持对上下文中任意内容（含背景，如 `sky`）做交互式运动控制，见图 6 与图 9。

![图 2. 不同表示建模的对比。(a) 点表示：用坐标点 $(x,y)$ 表示实体。(b) Trajectory Map：用轨迹向量图表示实体轨迹。(c) 2D Gaussian：用 2D Gaussian 图表示实体。(c) Box representation：用 bounding box 表示实体。(d) Entity representation：提取实体的 latent diffusion feature 来刻画它。](../../../arxiv/geometry_control/draganything/extracted/images/pipeline1.png)

---

## 2. 相关工作

### 2.1 图像与视频生成

图像生成近期受到广泛关注。代表性工作包括 Stability AI 的 Stable Diffusion、OpenAI 的 DALL-E 2、Google 的 Imagen、商汤的 RAPHAEL、Meta 的 Emu 等，在图像生成任务上影响显著。可控图像生成同样进展很快，以 ControlNet 为代表：利用 Canny 边缘、Hough 线、用户涂鸦、人体关键点、分割图等引导信息，可实现精确的图像生成。

相比之下，视频生成仍相对早期。Video Diffusion Models 首先用 3D U-Net 扩散架构预测并生成视频序列。Imagen Video 提出级联扩散视频模型以生成高清视频，并尝试把 text-to-image 设定迁移到视频生成。Show-1 直接在像素空间实现时间扩散模型，并用 inpainting 与超分辨率做高分辨率合成。Video LDM 首次把 LDM 范式用于高分辨率视频生成，在潜空间扩散模型中引入时间维。I2VGen-XL 引入级联网络，通过分离相关因素提升性能，并以静图作为必要引导保证数据对齐。产业界也有许多工作，包括 Gen-2、PikaLab、SORA。然而，相对通用视频生成，可控视频生成仍有提升空间。本文旨在推进基于轨迹的视频生成。

### 2.2 可控视频生成

已有若干工作关注可控视频生成，例如 AnimateDiff、Control-A-Video、Emu Video、MotionDirector。Control-A-Video 尝试以边缘或深度图等控制信号序列为条件生成视频，并采用两种运动自适应噪声初始化策略。Follow Your Pose 提出两阶段训练方案，可利用图像–pose 对以及无 pose 视频，得到 pose 可控的人物视频。ControlVideo 设计无训练框架，实现结构一致的可控 text-to-video 生成。这些工作都聚焦由稠密引导信号（mask、人体 pose、depth 等）引导的视频生成。但在真实应用中获取稠密引导信号既困难也不友好。相比之下，用基于轨迹的拖拽更为可行。

早期基于轨迹的工作常用光流或循环神经网络实现运动控制。TrailBlazer 用 bounding box 引导主体运动，以增强视频合成的可控性。DragNUWA 把稀疏笔画编码到稠密 flow 空间，再作为引导信号控制物体运动。类似地，MotionCtrl 把每个物体的轨迹坐标直接编码成向量图，作为条件控制物体运动。这些工作可归为两类范式：Trajectory Map（点）与 box representation。Box representation（如 TrailBlazer）只能处理 instance 级物体，无法容纳星空这类背景。现有 Trajectory Map Representation（如 DragNUWA、MotionCtrl）较为粗放，未考虑实体的语义：换言之，单个点不足以代表一个实体。本文提出 DragAnything，用所提出的 entity representation 实现真正的实体级运动控制。

---

## 3. 方法

### 3.1 任务形式化与动机

#### 任务形式化

基于轨迹的视频生成要求模型根据给定运动轨迹合成视频。给定点轨迹 $\{(x_{1}, y_{1}), (x_{2}, y_{2}), \dots, (x_{L}, y_{L})\}$，其中 $L$ 为视频长度，用条件去噪自编码器 $\epsilon_{\theta}(z, c)$ 生成与该运动轨迹对应的视频。本文中引导信号 $c$ 包含三类信息：轨迹点、视频首帧，以及首帧的实体 mask。

#### 动机

近期若干基于轨迹的工作（如 DragNUWA、MotionCtrl）探索用轨迹点控制视频生成中的物体运动。它们通常用给定的轨迹坐标或其派生量，直接操控对应像素或像素区域。但它们忽略了一个关键问题：如图 1 与图 2 所示，*所给轨迹点未必能充分代表我们想控制的实体*。因此，拖这些点未必能正确控制物体运动。

为验证该假设——即简单拖拽像素或像素区域无法有效控制物体运动——我们设计了一项 toy experiment。如图 3，我们用经典点跟踪器 Co-Tracker 跟踪合成视频中的每个像素，观察其轨迹变化。从像素运动变化中，我们得到两条新认识：

**洞见 1：物体上的轨迹点无法代表该实体**（图 3(a)）。从 DragNUWA 的像素运动轨迹可见，拖拽 `cloud` 上的一个像素点并不会让云移动，反而导致相机上移。这说明模型感知不到我们要控制云的意图，即单个点无法代表云。因此我们思考：是否存在更直接、更有效的表示，能精确控制我们想操控的区域（所选区域）。

**洞见 2：对轨迹点表示范式（图 2(a)–(c)）而言，越靠近拖拽点的像素受影响越大，运动也越大**（图 3(b)）。对比可见，DragNUWA 合成视频中，越靠近拖拽点的像素运动越大。而我们期望的是物体按所给轨迹整体运动，而不是个别像素各自运动。

基于上述两点认识与观察，我们提出一种新的 Entity Representation：提取我们想控制的物体的 latent feature 作为其表示。如图 3，对应运动轨迹的可视化表明，本方法能实现更精确的实体级运动控制。例如图 3(b) 显示，本方法能精确控制 `seagulls` 与 `fish` 的运动；而 DragNUWA 只是拖对应像素区域，导致外观异常形变。

![图 3. 关于 Entity Representation 动机的 toy experiment。已有方法（DragNUWA、MotionCtrl）直接拖像素，无法精确控制物体目标；本方法用 entity representation 实现精确控制。](../../../arxiv/geometry_control/draganything/extracted/images/motivation_method.png)

![图 4. DragAnything 框架。架构含两部分：1）Entity Semantic Representation Extraction：根据实体 mask 索引从 Diffusion Model 提取 latent feature，作为对应实体表示。2）DragAnything 主体框架：用对应的 entity representation 与 2D Gaussian representation 控制实体运动。](../../../arxiv/geometry_control/draganything/extracted/images/pipeline.png)

### 3.2 架构

遵循 SVD，基座架构主要由三部分组成：去噪扩散模型（3D U-Net），用于在时空上高效学习去噪过程；编码器与解码器，分别把视频编入潜空间，并把去噪后的 latent feature 重建为视频。受 ControlNet 启发，我们用一个 3D U-Net 编码引导信号，再将其作用于 SVD 去噪 3D U-Net 的 decoder blocks，如图 4。与先前工作不同，我们设计了 entity representation 提取机制，并与 2D Gaussian representation 结合，形成最终有效表示，从而用该表示做实体级可控生成。

### 3.3 Entity Semantic Representation Extraction

本方法的条件信号需要 Gaussian representation（§3.3.2）与对应的 entity representation（本节）。下面说明如何从首帧图像提取这些表示。

#### Entity Representation Extraction

给定首帧图像 $\boldsymbol{\mathrm{I}} \in \mathbb{R}^{H \times W \times 3}$ 及对应实体 mask $\boldsymbol{\mathrm{M}}$，我们先通过 diffusion inversion（扩散前向过程）得到该图像的 latent noise $\boldsymbol{x}$；该过程不可训练，基于固定马尔可夫链，逐步向图像添加高斯噪声。然后用去噪 U-Net $\epsilon_{\theta}$ 提取对应的 latent diffusion features $\mathcal{F} \in \mathbb{R}^{H \times W \times C}$：

$$
\mathcal{F} = \epsilon_{\theta}(\boldsymbol{x}_{t}, t),
$$

其中 $t$ 表示第 $t$ 个时间步。已有工作表明，单次前向即可有效提取表示；只从一步提取特征有两个优点：推理更快、性能更好。有了 diffusion features $\mathcal{F}$，通过按实体 mask 的对应坐标索引，即可得到对应的 entity embeddings。为简便，用 average pooling 处理这些 embedding，得到最终 embedding $\{e_{1}, e_{2}, \ldots, e_{k}\}$，其中 $k$ 为实体个数，每个 embedding 通道数为 $C$。

为把这些 entity embeddings 与对应轨迹点关联，我们直接初始化零矩阵 $\boldsymbol{\mathrm{E}} \in \mathbb{R}^{H \times W \times C}$，再按轨迹序列点插入 entity embeddings，如图 5。训练时，用首帧实体 mask 提取各实体中心坐标 $\{(x^{1}, y^{1}), (x^{2}, y^{2}), \ldots, (x^{k}, y^{k})\}$，作为每条轨迹序列的起点。有了这些中心坐标索引，把 entity embeddings 插入对应零矩阵 $\boldsymbol{\mathrm{E}}$，即可得到最终 entity representation $\boldsymbol{\hat{\mathrm{E}}}$（细节见 §3.4）。

有了首帧实体中心坐标 $\{(x^{1}, y^{1}), (x^{2}, y^{2}), \ldots, (x^{k}, y^{k})\}$，我们用 Co-Tracker 跟踪这些点，得到对应运动轨迹 $\bigl\{\{(x^{1}_{i}, y^{1}_{i})\}_{i=1}^{L}, \{(x^{2}_{i}, y^{2}_{i})\}_{i=1}^{L}, \ldots, \{(x^{k}_{i}, y^{k}_{i})\}_{i=1}^{L}\bigr\}$，其中 $L$ 为视频长度。由此可得到每帧对应的 entity representation $\{\boldsymbol{\hat{\mathrm{E}}}_{i}\}_{i=1}^{L}$。

#### 2D Gaussian Representation Extraction

越靠近实体中心的像素通常越重要。我们希望所提出的 entity representation 更关注中心区域，同时降低边缘像素权重。2D Gaussian Representation 能有效增强这一点：越靠近中心的像素权重越大，如图 2(c)。给定点轨迹 $\bigl\{\{(x^{1}_{i}, y^{1}_{i})\}_{i=1}^{L}, \{(x^{2}_{i}, y^{2}_{i})\}_{i=1}^{L}, \ldots, \{(x^{k}_{i}, y^{k}_{i})\}_{i=1}^{L}\bigr\}$ 与 $\{r^{1}, \ldots, r^{k}\}$，可得到对应的 2D Gaussian Distribution Representation 轨迹序列 $\{\boldsymbol{\mathrm{G}}_{i}\}_{i=1}^{L}$，如图 5。再经编码器 $\mathcal{E}$（见下一小节）处理后，与 entity representation 融合，以增强对中心区域的关注，如图 4。

#### Entity Representation 与 2D Gaussian Map 的编码器

如图 4，编码器记为 $\mathcal{E}$，用于把 entity representation 与 2D Gaussian map 编码到 latent feature 空间。该编码器由四个卷积块处理输入引导信号，每块含两层卷积与一个 SiLU 激活。每块把输入特征分辨率下采样 2 倍，最终输出分辨率为 $1/8$。处理 entity 与 Gaussian representation 的编码器结构相同，唯一差别是第一块的通道数：当两种表示通道数不同时该通道数不同。经编码器后，我们遵循 ControlNet，把 Entity Representation 与 2D Gaussian Map Representation 的 latent features 与视频对应的 latent noise 相加：

$$
\{\boldsymbol{\mathrm{R}}_{i}\}_{i=1}^{L} = \mathcal{E}\bigl(\{\boldsymbol{\hat{\mathrm{E}}}_{i}\}_{i=1}^{L}\bigr) + \mathcal{E}\bigl(\{\boldsymbol{\mathrm{G}}_{i}\}_{i=1}^{L}\bigr) + \{\boldsymbol{\mathrm{Z}}_{i}\}_{i=1}^{L},
\tag{2}
$$

其中 $\boldsymbol{\mathrm{Z}}_{i}$ 表示第 $i$ 帧的 latent noise。随后将特征 $\{\boldsymbol{\mathrm{R}}_{i}\}_{i=1}^{L}$ 输入去噪 3D U-Net 的编码器，得到四个不同分辨率的特征，作为 latent 条件信号。这四个特征再加到基座模型去噪 3D U-Net 的特征上。

### 3.4 训练与推理

![图 5. Ground truth 生成流程示意。训练时，我们从带实体级标注的视频分割数据集生成 ground truth 标签。](../../../arxiv/geometry_control/draganything/extracted/images/GT.png)

#### Ground Truth 标签生成

训练时需要生成对应的 Entity Representation 轨迹与 2D Gaussian，如图 5。首先，对每个实体，用对应 mask 计算其内切圆，得到中心坐标 $(x, y)$ 与半径 $r$。然后用 Co-Tracker 得到该中心的对应轨迹 $\{(x_{i}, y_{i})\}_{i=1}^{L}$，作为该实体的代表运动轨迹。有了这些轨迹点与半径，可计算每帧对应的 Gaussian 分布值。对 entity representation，我们把对应 entity embedding 插入以 $(x, y)$ 为圆心、半径为 $r$ 的圆内。最终得到用于训练模型的 Entity Representation 轨迹与 2D Gaussian。

#### 损失函数

视频生成任务中常用 Mean Squared Error（MSE）优化模型。给定对应的 entity representation $\boldsymbol{\hat{\mathrm{E}}}$ 与 2D Gaussian representation $\boldsymbol{\mathrm{G}}$，目标可简化为：

$$
\mathcal{L}_{\theta} = \sum_{i=1}^{L} \boldsymbol{\mathrm{M}} \Bigl\| \epsilon - \epsilon_{\theta}\bigl(\boldsymbol{x}_{t,i}, \mathcal{E}_{\theta}(\boldsymbol{\hat{\mathrm{E}}}_{i}), \mathcal{E}_{\theta}(\boldsymbol{\mathrm{G}}_{i})\bigr) \Bigr\|_{2}^{2},
$$

其中 $\mathcal{E}_{\theta}$ 表示 entity 与 2D Gaussian representation 的编码器，$\boldsymbol{\mathrm{M}}$ 为每帧图像上实体的 mask。模型优化目标是控制目标物体的运动。对其它物体或背景，我们不希望影响生成质量。因此用 mask $\boldsymbol{\mathrm{M}}$ 约束 MSE loss，使其只在我们想优化的区域上回传。

#### 用户轨迹交互的推理

DragAnything 对用户友好。推理时，用户只需用 SAM 点击选出要控制的区域，再拖该区域内任意像素，形成合理轨迹。随后 DragAnything 即可生成与期望运动对应的视频。

---

## 4. 实验

### 4.1 实验设置

**实现细节。** DragAnything 基于 Stable Video Diffusion（SVD）的架构与权重，该模型训练用于生成 $25$ 帧、分辨率 $320 \times 576$ 的视频。全部实验在 PyTorch 与 Tesla A100 GPU 上进行。优化器为 AdamW，共 $100\mathrm{k}$ 训练步，学习率 $1\times 10^{-5}$。

**评测指标。** 为全面评估本方法，我们从人工评估与自动脚本指标两方面评测。遵循 MotionCtrl，采用两类自动指标：1）*视频质量*：用 Frechet Inception Distance（FID）与 Frechet Video Distance（FVD）评估视觉质量与时间一致性。2）*物体运动控制性能*：用预测轨迹与 ground truth 物体轨迹之间的欧氏距离（ObjMC）评估物体运动控制。此外，考虑到视频美观度，User Study 从 Google Image 收集并标注了 $30$ 张图像及其对应点轨迹与 mask。三位专业评估者需从视频质量与运动匹配两方面对合成视频投票。图 6 与图 9 的视频采样自这 $30$ 个案例。

**数据集。** 评测轨迹引导的视频生成需要测试集中每个视频的运动轨迹作为输入。为获得此类标注，我们采用 VIPSeg 验证集作为测试集：用视频首帧每个物体的 instance mask 提取其中心坐标，再用 Co-Tracker 跟踪该点得到对应运动轨迹，作为指标评测的 ground truth。由于 FVD 要求视频分辨率与长度一致，我们将 VIPSeg `val` 数据集缩放到分辨率 $256 \times 256$、长度 $14$ 帧再评测。相应地，我们也用 VIPSeg 训练集作为训练数据，并用 Co-Tracker 获取对应运动轨迹作为标注。

![图 6. DragAnything 可视化。所提出的 DragAnything 能在实体级精确控制物体运动，并生成高质量视频。第 $20$ 帧的像素运动可视化由 Co-Tracker 得到。](../../../arxiv/geometry_control/draganything/extracted/images/vis.png)

### 4.2 与 State-of-the-Art 方法对比

从四方面比较生成视频：1）用 FID 评估视频质量；2）用 FVD 评估时间一致性；3）用 ObjMC 评估物体运动；4）用人工投票做 User Study。

**VIPSeg `val` 上的视频质量。** 表 1 给出 VIPSeg `val` 上用 FID 比较的视频质量。我们控制其它条件相同（基座架构），比较本方法与 DragNUWA。DragAnything 的 FID 达到 $33.5$，显著优于当前 SOTA 模型 DragNUWA（$33.5$ vs. $39.8$，低 $6.3$）。图 6 与图 9 也表明 DragAnything 合成的视频质量很高。

**VIPSeg `val` 上的时间一致性。** FVD 通过比较生成视频与 ground truth 视频中的特征分布，评估生成视频的时间一致性。FVD 对比见表 1。相对 DragNUWA 的 $519.3$ FVD，DragAnything 达到更优的时间一致性，即 $494.8$，提升 $24.5$。

**VIPSeg `val` 上的物体运动。** 遵循 MotionCtrl，ObjMC 通过计算预测轨迹与 ground truth 轨迹之间的欧氏距离来评估运动控制性能。表 1 给出 VIPSeg `val` 上的 ObjMC 对比。相对 DragNUWA，DragAnything 达到新的 state-of-the-art：$305.7$，提升 $18.9$。图 7 给出两种方法的可视化对比。

**运动控制与视频质量的 User Study。** 图 8 给出运动控制与视频质量的 User Study 对比。我们的模型在运动控制与视频质量的人工投票上分别超出 DragNUWA $26\%$ 与 $12\%$。图 7 提供视觉对比，更多可视化见图 6。本算法对运动控制有更准确的理解与实现。

**表 1. VIPSeg `val` $256\times 256$ 上的性能对比。** 我们只与 DragNUWA 比较，因为其它相关工作（如 MotionCtrl）未基于 SVD 发布源代码。

| Method              | Base Arch. | ObjMC           | FVD             | FID            | Venue/Date       |
| ------------------- | ---------- | --------------- | --------------- | -------------- | ---------------- |
| DragNUWA            | SVD        | 324.6           | 519.3           | 39.8           | arXiv, Aug. 2023 |
| DragAnything (Ours) | SVD        | **305.7** | **494.8** | **33.5** | —               |

![图 7. 与 DragNUWA 的可视化对比。DragNUWA 导致外观畸变（第一行）、`sky` 与 `ship` 失控（第三行）、错误的相机运动（第五行）；而 DragAnything 能精确控制运动。](../../../arxiv/geometry_control/draganything/extracted/images/comparison1.png)

![图 8. 运动控制与视频质量的 User Study。DragAnything 在运动控制与视频质量上均更优。](../../../arxiv/geometry_control/draganything/extracted/images/UserStudy.png)

### 4.3 Ablation Studies

Entity representation 与 2D Gaussian representation 都是本文的核心组件。我们保持其它条件不变，只修改对应的条件 embedding 特征。表 2 给出两种表示的 ablation。

**Entity Representation $\boldsymbol{\hat{\mathrm{E}}}$ 的作用。** 为考察 Entity Representation $\boldsymbol{\hat{\mathrm{E}}}$ 的影响，我们观察最终 embedding（式 (2)）中是否包含该表示时性能如何变化。条件信息 $\boldsymbol{\hat{\mathrm{E}}}$ 主要影响生成视频中的物体运动，因此只需比较 ObjMC；FVD 与 FID 关注时间一致性与整体视频质量。加入 Entity Representation $\boldsymbol{\hat{\mathrm{E}}}$ 后，模型的 ObjMC 显著提升（$92.3$），达到 $318.4$。

**表 2. Entity 与 2D Gaussian Representation 的 ablation。** 二者结合收益最大。

| Entity Rep. | Gaussian Rep. | ObjMC           | FVD   | FID  |
| :---------: | :-----------: | --------------- | ----- | ---- |
|            |              | 410.7           | 496.3 | 34.2 |
|     ✓     |              | 318.4           | 494.5 | 34.1 |
|            |      ✓      | 339.3           | 495.3 | 34.0 |
|     ✓     |      ✓      | **305.7** | 494.8 | 33.5 |

**表 3. Loss Mask $\boldsymbol{\mathrm{M}}$ 的 ablation。** Loss mask 能带来一定收益，尤其对 ObjMC。

| Loss Mask $\boldsymbol{\mathrm{M}}$ | ObjMC           | FVD   | FID  |
| :--------------------------: | --------------- | ----- | ---- |
|                              | 311.1           | 500.2 | 34.3 |
|              ✓              | **305.7** | 494.8 | 33.5 |

**2D Gaussian Representation 的作用。** 与 Entity Representation 类似，我们观察最终 embedding 中是否包含 2D Gaussian Representation 时 ObjMC 如何变化。2D Gaussian Representation 带来 $71.4$ 的提升，达到 $339.3$。总体而言，同时使用 Entity 与 2D Gaussian Representations 时性能最高，达到 $305.7$。这一现象表明两种表示具有相互增强的效果。

**Loss Mask $\boldsymbol{\mathrm{M}}$ 的作用。** 表 3 给出 Loss Mask $\boldsymbol{\mathrm{M}}$ 的 ablation。不使用 loss mask $\boldsymbol{\mathrm{M}}$ 时，我们直接对整图每个像素优化 MSE loss。Loss mask 能带来一定收益，ObjMC 约提升 $5.4$。

![图 9. DragAnything 的多种运动控制。DragAnything 可实现多样运动控制，例如前景、背景与相机。](../../../arxiv/geometry_control/draganything/extracted/images/vis2.png)

### 4.4 多种运动控制的讨论

DragAnything 非常灵活且用户友好，支持对视频中出现的任意实体做多样运动控制。本节按四类讨论相应运动控制。

**前景运动控制。** 如图 9(a)，前景运动控制是最基本、最常用的操作。`sun` 与 `horse` 都属于前景。我们用 SAM 选出需要控制的对应区域，再拖该区域内任意点，即可实现对该物体的运动控制。可见 DragAnything 能精确控制太阳与马的运动。

**背景运动控制。** 相对前景，背景通常更难控制，因为背景元素（如 `clouds`、`starry skies`）形状不可预期、难以刻画。图 9(b) 展示两种场景下视频生成的背景运动控制。通过拖云上的一个点，DragAnything 可控制整层云向右移动或更远离。

**前景与背景同时运动控制。** DragAnything 也能同时控制前景与背景，如图 9(c)。例如，通过拖三个像素，可同时实现 `cloud layer` 右移、`sun` 上升、`horse` 右移。

**相机运动控制。** 除视频中实体的运动控制外，DragAnything 也支持一些基本的相机运动控制，例如 zoom in 与 zoom out，如图 9(d)。用户只需选中整幅图像，再拖四个点，即可实现相应的放大或缩小。此外，用户也可通过拖任意点控制整台相机上、下、左、右移动。

---

## 5. 结论

本文重新审视视频生成任务中当前基于轨迹的运动控制方法，并提出两点新认识：1）物体上的轨迹点不足以代表该实体。2）对轨迹点表示范式而言，越靠近拖拽点的像素影响越强，运动越大。针对这两项技术挑战，我们提出 DragAnything：用扩散模型的 latent features 表示每个实体。所提出的 entity representation 作为一种开放域 embedding，能表示任意物体，从而控制包括背景在内的多样实体运动。大量实验表明，DragAnything 在 User Study 上达到 SOTA，人工投票超出先前 state of the art（DragNUWA）$26\%$。

![图 10. DragAnything 的失败案例。DragAnything 仍有一些坏例，尤其在控制较大运动时。](../../../arxiv/geometry_control/draganything/extracted/images/badcase.png)

![图 11. DragAnything 的更多可视化。](../../../arxiv/geometry_control/draganything/extracted/images/vis3.png)

---

## 附录

### 潜在负面影响讨论

一个潜在负面影响是可能强化训练数据中的偏见：模型从可能含社会偏见的已有数据集中学习。此外，生成内容存在被滥用的风险，可能产生误导或不适宜的视觉材料。隐私问题也可能出现，尤其是在未获当事人明确同意的情况下生成涉及个人的视频。与其它视频生成技术一样，需要保持警惕并负责任地落地，以缓解这些潜在负面影响并确保伦理使用。

### 局限性与失败案例分析

尽管 DragAnything 已展现有前景的性能，仍有若干可改进之处；这些也是当前其它基于轨迹的视频生成模型所共有的：1）当前基于轨迹的运动控制限于 2D 维，无法处理 3D 场景中的运动，例如控制某人转身或更精确的身体旋转。2）当前模型受基座模型 Stable Video Diffusion 性能约束，无法生成运动非常大的场景，如图 10。显然，第一列视频帧中恐龙的腿不符合真实世界约束：有几帧出现五条腿以及一些奇怪运动。第二行鹰翅膀模糊也有类似情况。这可能是由于运动过大，超出基座模型的生成能力，导致视频质量崩溃。

应对这两项挑战有一些潜在方案。对第一项，可行做法是把深度信息并入 2D 轨迹，扩展为 3D 轨迹信息，从而控制物体在 3D 空间中的运动。对第二项，需要更强的基座模型以支持更大、更稳健的运动生成能力。例如，利用 OpenAI 最新的 text-to-video 基座 SORA，无疑有潜力显著提升生成视频质量。此外，我们在补充材料中提供了更多精美视频案例供参考，如图 11。GIF 格式的更多可视化请见同目录下的 `DragAnything.html`，直接点击打开即可。
