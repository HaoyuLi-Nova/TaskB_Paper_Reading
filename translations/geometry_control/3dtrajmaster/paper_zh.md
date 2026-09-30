# 3DTrajMaster：掌握视频生成中多实体运动的 3D 轨迹

**作者：** Xiao Fu$^{1}$、Xian Liu$^{1}$、Xintao Wang$^{2\Letter}$、Sida Peng$^{3}$、Menghan Xia$^{2}$、Xiaoyu Shi$^{2}$、Ziyang Yuan$^{2}$、Pengfei Wan$^{2}$、Di Zhang$^{2}$、Dahua Lin$^{1\Letter}$

**机构：** $^{1}$香港中文大学；$^{2}$快手科技；$^{3}$浙江大学

**贡献说明：** $\Letter$通讯作者

**出处：** ICLR 2025（arXiv:2412.07759）

**arXiv：** [2412.07759](https://arxiv.org/abs/2412.07759)

**项目页：** http://fuxiao0719.github.io/projects/3dtrajmaster

**原文 TeX：** [`arxiv/geometry_control/3dtrajmaster/extracted/main.tex`](../../../arxiv/geometry_control/3dtrajmaster/extracted/main.tex)

---

## 摘要

本文旨在操控视频生成中的多实体 3D 运动。先前可控视频生成方法主要用 2D 控制信号操控物体运动，并已取得出色的合成效果。然而，2D 控制信号在表达物体运动的 3D 本质时存在固有局限。为克服这一问题，我们提出 **3DTrajMaster**：一个稳健的控制器，根据用户指定的实体 6DoF 姿态（位置与旋转）序列，在 *3D 空间* 中调节多实体动态。方法的核心是一个即插即用的 3D-motion grounded object injector：通过 gated self-attention 把多个输入实体与各自的 3D 轨迹融合。此外，我们利用 injector 架构来保留视频扩散先验，这对泛化能力至关重要。

为缓解视频质量下降，训练时引入 domain adaptor，推理时采用 annealed sampling。针对合适训练数据的缺失，我们构建 **360°-Motion Dataset**：先把收集的 3D 人物与动物资产与 GPT 生成的轨迹对应，再在多样化的 3D UE 平台上用 12 个均匀环绕相机捕捉其运动。大量实验表明，3DTrajMaster 在控制多实体 3D 运动的精度与泛化上均达到新的 state-of-the-art。项目页：http://fuxiao0719.github.io/projects/3dtrajmaster 。

---

![图 1. 3DTrajMaster 在 text-to-video（T2V）生成中，用输入的实体专属 3D 轨迹控制一个或多个实体在 3D 空间中的运动。它支持多样实体类别（人、动物、汽车、机器人、自然力等），并可灵活编辑实体描述（更多见附录图 S4）。文本提示为 “{Entity 1},..., and {Entity N} is/are moving in the {Location}”。（请读者到项目网站查看更多泛化结果，数量 ≥200。）](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/teaser_demo.png)

---

## 1. 引言

可控视频生成（Sora、AnimateDiff、VideoCrafter）旨在根据用户输入（文本提示、草图或 bounding box 等）合成高保真视频。其中一项关键目标是精确操控视频中的物体运动：这对模拟动态世界至关重要，并有助于视频生成模型理解世界背后的物理。此外，它还能释放视频生成模型的多种应用，例如电影工业中的虚拟摄影、交互式游戏，以及为具身智能系统提供 world model。

近期已有方法通过引入 2D 控制信号来操控视频生成中的物体运动，例如 2D 草图（VideoComposer、SparseCtrl）、bounding box（Direct-a-Video、Boximator）以及点（MotionCtrl、Tora）。这些方法提供便捷的用户交互，并已给出令人印象深刻的生成结果。然而我们认为，2D 控制信号无法充分表达运动固有的 3D 本质，从而限制了对物体运动的控制能力。真实世界物体在 3D 空间中运动，部分运动属性只能通过 3D 表示来描述。例如，物体旋转在 3D 中可用三个参数简洁描述，物体间遮挡可用 z-buffering 简单表示。相比之下，2D 控制信号很难表达这些概念。

本文关注在视频生成模型中控制多物体 3D 运动，以模拟物体在 3D 空间中的真实动态。该设定更贴近下游应用需求，例如在电影中仿真写实人体运动，或在游戏中探索 3D 虚拟场景。但该问题极具挑战。我们需要回答三个核心问题：

1. 如何精确表示物体的 3D 运动；
2. 如何在视频生成模型中把多个物体描述与各自的运动序列对应起来；
3. 注入 3D 运动信息后，如何保持视频模型的泛化能力。

为此，我们提出 **3DTrajMaster**：以实体专属 6DoF 姿态序列作为额外输入，在视频生成中操控多实体在 3D 空间中的运动。模型核心是一个即插即用的 3D-motion grounded object injector：把每个实体与其对应姿态序列关联，再把这些条件注入基础模型以控制实体运动。具体地，实体与轨迹分别经冻结的 text encoder 与可学习的 pose encoder 投影为潜嵌入；两种模态嵌入再按实体相加以形成对应关系，并送入 gated self-attention 层做运动融合。这种即插即用架构保留视频模型先验，并能在更多样的实体与 3D 轨迹上泛化。

训练的另一挑战在于数据可得性。现有视频数据集有两项关键局限：

1. *实体多样性低*：带配对实体与 3D 轨迹的数据集大多限于人与自动驾驶车辆，空间分布不一致，实体可能过于拥挤。
2. *姿态估计不准/失败*：当前 6D 姿态估计方法聚焦刚体；非刚体（如动物）代表性不足，人体姿态则主要靠 SMPL 研究。

因此我们选择自建数据集，称为 **360°-Motion Dataset**，用先进 UE 渲染技术得到统一的轨迹分布。我们先收集人物与动物的 3D 资产，并缩放到统一立方空间；再用 GPT 为这些资产生成 3D 轨迹模板；将各类实体与轨迹模板排列组合，形成多样运动。这些全局动画资产在收集到的 3D 场景中用 12 个均匀布置的相机捕捉，场景包括城市（MatrixCity）、沙漠、森林以及投影到 3D 空间的 HDRI（Poly Haven：https://polyhaven.com/ ）。为防止自建数据集带来的视频域偏移，我们引入两个关键组件：

1. **Video domain adaptor**：训练以拟合数据分布，推理时略微降低其影响。
2. **Annealed sampling**：早期步注入轨迹以引导总体运动，后期 dropout，转回标准 T2V。

我们在 GPT 生成实体提示、经筛选的新姿态序列上评估 3DTrajMaster，相对当前 SOTA 取得显著领先。贡献总结如下：

1. 据我们所知，这是首次在可控视频生成中为 3D 空间的多实体运动定制 6 自由度（DoF），为细粒度运动控制建立新基准。
2. 提出 3D-motion grounded 视频扩散模型，用姿态序列作为运动表示来控制多实体运动。灵活的 object injector 强制物体与其运动之间的实体级对应，并保留视频扩散先验。
3. 引入可扩展的 4D 运动数据构建机制，以及 video domain adaptor、annealed sampling 等技术，在保持运动精度的同时提升视频质量。
4. 3DTrajMaster 在控制 3D 实体运动上达到 state-of-the-art 精度，并允许细粒度定制实体输入，例如改变人物发型、服装、性别与体型。

---

## 2. 相关工作

**用 2D 引导定制视频运动。** 先前方法主要在 2D 空间做运动控制，因为这更容易与输入视频格式对齐。一条直接路径是按参考视频中的运动模式来引导视频（MotionDirector、VMC、MotionClone）。但它们要求用户提供参考视频模板。免训练范式（Direct-a-Video 等）用 attention 编辑时空布局，可缓解该问题，但在真实场景中泛化差，且高度依赖试错。后续工作采用更高层表示，例如草图与深度（稠密或稀疏；VideoComposer、SparseCtrl）、pose skeleton（DreamMoving、MagicAnimate、DreamCinema）、bounding box（Boximator）以及 2D 轨迹（MotionCtrl、Tora、DragNUWA、Direct-a-Video），以实现更灵活的运动生成。这些方法虽能建模相机、物体或关节运动，但缺少 3D 感知，限制了精确的 3D 运动控制。

**学习 3D-aware 运动合成。** 视频是从 3D 世界投影得到的图像序列，因此在 3D 空间操控视频既更关键也更有影响力。其中一方面是相机运动。MotionCtrl 首次用 3D 空间中的相机姿态（旋转与平移）调节视频；CameraCtrl 与 VD3D 进一步用 Plücker embedding 增强相机表示。SynCamMaster 把单相机控制扩展到多相机同步。GameGen-X 能根据新颖的 “WASD” 键盘输入生成游戏视频。另有工作探索免训练范式（MotionMaster 等）。然而，它们都未解决 3D 空间中物体运动的定制。在 2D 图上操控（MotionCtrl、Tora）在多物体场景中常常失败，尤其是：(1) 对齐每个实体与其对应运动；(2) 处理 *3D 遮挡*。相比之下，3DTrajMaster 首次克服这些问题，并仿真合理的 3D 运动。

---

## 3. 3DTrajMaster

我们的目标是：以实体专属 3D 轨迹作为额外输入，掌握 text-to-video（T2V）生成中实体在 3D 空间的运动。为此我们提出 *3DTrajMaster*（见图 2），一个分两阶段训练的 3D-motion grounded 视频扩散模型。首先描述视频扩散模型与任务形式化（第 3.1 节）；然后给出所提模型：核心是训练一个即插即用的 3D grounded object injector，以整合多个细粒度实体描述及其各自的姿态序列（第 3.2 节）；再引入 domain adaptor，缓解自建训练数据带来的视频域偏移（第 3.3 节）；最后详述推理过程，用 annealed sampling 提升视频质量（第 3.4 节）。

![图 2. 3DTrajMaster 框架。给定由 N 个实体 $\{\mathbf{e}_n\}_{n=1}^{N}$ 组成的文本提示，3DTrajMaster（a）能生成实体运动符合输入实体级姿态序列 $\{\mathbf{P}_n\}_{n=1}^{N}$ 的目标视频。训练分两阶段。首先用 domain adaptor 缓解训练视频的负面影响；然后在 2D spatial self-attention 层之后插入 object injector 模块，以整合成对的实体提示与 3D 轨迹。（b）物体注入过程细节。实体经 text encoder 投影为潜嵌入；成对姿态序列经可学习 pose encoder 投影，再与实体嵌入融合，形成实体–轨迹对应。该条件嵌入与视频潜变量拼接，送入 gated self-attention 做运动融合。最后，修改后的潜变量回到 DiT block 的其余层。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/video_generator.png)

### 3.1 3D 实体感知视频分布的预备知识

**视频扩散模型。** 潜空间 text-to-video 扩散模型（Imagen Video、Video Diffusion Models、Sora、VideoCrafter、Stable Video Diffusion）在潜空间学习编码视频数据 $\mathbf{x}$（$\mathbf{x}=\mathcal{E}(X)$，$\mathcal{E}(\cdot)$ 为 VAE encoder）在文本描述 $\mathbf{c}$ 条件下的条件分布 $p(\mathbf{x}|\mathbf{c})$。前向过程在 Markov 链中把干净数据 $\mathbf{x}_{0}$ 逐步转到目标高斯分布：

$$
\left\{\mathbf{x}_t, t \in(1, T) \mid \mathbf{x}_t=\alpha_t \mathbf{x}_0+\sigma_t \boldsymbol{\epsilon}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \right\}.
$$

为从噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_{t}^2 \mathbf{I})$ 迭代恢复数据 $\hat{\mathbf{x}}_{0}$，它学习去噪模型 $\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}$，目标为 $\boldsymbol{\epsilon} \approx \hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{x}_t;t,\mathbf{c})$。采用 preconditioning 策略（EDM、progressive distillation）时，通过把 $\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}$ 参数化为

$$
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}} = c_{\mathrm{out}}(\sigma_t) \hat{F}_{\boldsymbol{\theta}}\bigl(c_{\mathrm{in}}(\sigma_t) \mathbf{x}_t ; \boldsymbol{c}, \sigma_t\bigr)+c_{\mathrm{skip }}(\sigma_t) \mathbf{x}_t
$$

来优化网络 $\hat{F}_{\boldsymbol{\theta}}$。

**任务形式化。** 给定由 N 个实体 $\{\mathbf{e}_n\}_{n=1}^{N}$ 组成的输入文本提示 $\boldsymbol{c}$，以及成对的 3D 轨迹 $\{\mathbf{P}_n\}_{n=1}^{N}$，其中第 $f$ 帧的 $\mathbf{P}_n^f=[\mathbf{R}; \mathbf{T}] \in \mathbb{R}^{3 \times 4}$，物体朝向与平移分别由 $\mathbf{R} \in \mathbb{R}^{3\times3}$ 与 $\mathbf{T} \in \mathbb{R}^{3}$ 表示。目标是生成合理视频 $\mathbf{X} \in \mathbb{R}^{F \times H \times W}$，使其符合每个实体描述 $\mathbf{e}$ 及其对应轨迹 $\mathbf{P}$。总体生成映射 $f(\cdot)$ 为

$$
f(\cdot): \mathbf{c} \in \mathcal{Y}^L,\ (\mathbf{e}_n \in \mathcal{Y}^{L_n},\mathbf{P}_n \in \mathbb{R}^{3 \times 4})_{n=1}^N \rightarrow \mathbf{X} \in \mathbb{R}^{F \times H \times W},
$$

其中 $\mathbf{X} \approx \mathcal{D}(\hat{\mathbf{x}}_0)$（$\mathcal{D}(\cdot)$ 为 VAE decoder），

$$
\hat{\mathbf{x}} = p(\hat{\mathbf{x}}_T) \prod_{t=1}^T p_{\boldsymbol{\theta}}\bigl(\hat{\mathbf{x}}_{t-1} \mid \hat{\mathbf{x}}_t, \mathbf{c}, (\mathbf{e}_n,\mathbf{P}_n)_{n=1}^N\bigr),
$$

$\mathcal{Y}$ 为字母表，$L$ 为 token 长度。主要挑战在于建模分布 $p_{\boldsymbol{\theta}}$，或具体地建模 $\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}$，以生成准确对应给定多个 3D 实体条件的写实视频。这里把 $\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{x}; \boldsymbol{c}, \sigma_t, (\mathbf{e}_n,\mathbf{P}_n)_{n=1}^N)$ 做成 transformer 架构（DiT），因其相对 U-Net 具有更优的可扩展性与性能。

### 3.2 即插即用的 3D-Motion Grounded Object Injector

**匹配实体–轨迹对。** 实体提示 $\{\mathbf{e}_n\}_{n=1}^{N}$ 经冻结 text encoder $\mathcal{E}_{\mathbf{T}}(\cdot): \mathbf{e}_n \in \mathcal{Y}^{L_n} \rightarrow \mathbf{Z}^{\mathbf{e}}_n \in \mathbb{R}^{L_{\mathrm{max}}\times D}$ 投影为潜嵌入 $\{\mathbf{Z}^{\mathbf{e}}_n\}_{n=1}^{N}$，每个嵌入 $\mathbf{Z}^{\mathbf{e}}_n$ 零填充到最大 token 长度 $L_{\mathrm{max}}$。相应地，姿态序列 $\{\mathbf{P}_n\}_{n=1}^{N}$ 经可训练 pose encoder $\mathcal{E}_{\mathbf{P}}(\cdot)$ 投影为潜嵌入 $\{\mathbf{Z}^{\mathbf{P}}_n\}_{n=1}^{N}$：$\mathbf{P}_n \in \mathbb{R}^{F \times 12} \rightarrow \mathbf{Z}^{\mathbf{P}}_n \in \mathbb{R}^{\tilde{F} \times D}$。Pose encoder $\mathcal{E}_{\mathbf{P}}$ 由一层 linear 与沿时间维的 downsampler 组成，类似于 3D VAE 对视频输入 $\mathbf{x}$ 的因果编码，映射为 $\mathcal{E}_{\mathbf{X}}(\cdot)$：$\mathbf{X} \in \mathbb{R}^{F \times H \times W} \rightarrow \mathbf{x}\in \mathbb{R}^{\tilde{F} \times \tilde{H} \times \tilde{W}}$。此处 downsampler 指张量的间隔采样；我们也尝试过若干顺序一维卷积层，效果相近。随后，成对的实体与轨迹嵌入经扩展并以实体级相加结合，形成绑定的实体–运动对应 $\mathbf{Z}^{\mathbf{Pe}}\in \mathbb{R}^{\tilde{F} \times N \times L_{\mathrm{max}} \times D}$。

**用于运动融合的 Gated Self-Attention。** 受 GLIGEN 启发，我们用 gated self-attention 层处理多个实体–轨迹对 $\mathbf{Z}^{\mathbf{Pe}}$（嵌入维数可变）作为输入，并进一步精炼相关特征。具体地，复制每个 DiT block 中 2D spatial self-attention 层的权重作为初始化，以实现 grounding。输入视频 token $\mathbf{x}_t$ 与 $\mathbf{Z}^{\mathbf{Pe}}$ 经该可训练副本做截断 self-attention。输出可写成残差连接形式：

$$
\begin{aligned}
\mathbf{x}_t &= \mathbf{x}_t + \beta \cdot \mathbf{Tc}\bigl(\mathbf{Att}(\mathbf{q}, \mathbf{k}, \mathbf{v})\bigr), \\
\mathbf{q} &= \mathbf{Q} \cdot \mathbf{T},\quad
\mathbf{k} = \mathbf{K} \cdot \mathbf{T},\quad
\mathbf{v} = \mathbf{V} \cdot \mathbf{T},\quad
\mathbf{T} = \mathbf{x}_t \oplus \mathbf{Z}^{\mathbf{Pe}},
\end{aligned}
$$

其中 $\beta$ 为可训练尺度，$\mathbf{Tc}(\cdot)$ 为截断操作以保留 $\mathbf{x}_t$ 的 token，$\mathbf{Att}(\cdot)$ 为 softmax attention，$\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 为 query、key、value 嵌入矩阵，$\oplus$ 表示拼接。本阶段训练 $\boldsymbol{\theta}_1$，包含 pose encoder 与 gated self-attention 参数：

$$
\mathcal{L}(\boldsymbol{\theta}_1)=\mathbb{E}_{\mathbf{x}, \mathbf{c}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_{t}^2 \mathbf{I}), \mathbf{e}, \mathbf{P}, t, \beta}\Bigl[\bigl\|\boldsymbol{\epsilon}-\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}_1}\bigl(\mathbf{x}_t, \mathbf{c}, (\mathbf{e}_n,\mathbf{P}_n)_{n=1}^N, t, \beta \bigr)\bigr\|_2^2\Bigr].
$$

### 3.3 缓解自建训练数据带来的视频域偏移

![图 3. 数据集构建示意。我们把（a）收集的 3D 资产与（b）GPT 生成的 3D 轨迹在（c）多样化 3D UE 平台上对应，并布置（d）12 个均匀分布的环绕相机，以视频形式捕捉物体运动。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/dataset.png)

**360°-Motion Dataset。** 高质量训练数据对学习可泛化的 3D 运动控制至关重要。直接做法是从常见视频数据集提取成对的实体描述与 6DoF 姿态。但这因两点而困难：

1. *实体多样性/质量低*：带配对实体与 3D 轨迹的数据集大多限于人（Scaling Up Dynamic Human Avatars、CIRCLE）与自动驾驶车辆（KITTI、Waymo），数据集之间空间分布不同，实体可能过于拥挤。在 Artgrid、Pixabay、Pexels 等视频数据集中（Artgrid：https://artgrid.io/ ，Pixabay：https://www.videvo.net/ ，Pexels：https://www.pexels.com/ ），人类类别在 3D/4D 资产目标中占比较大（见附录第 A.5.2 节），限制模型向动物、车辆等其他类别泛化。WebVid 中的水印等问题进一步增加过滤成本。
2. *姿态估计精度低/失败*：多数 6D 姿态估计方法只关注刚体，并依赖 CAD 模型（MegaPose、FoundationPose）或带姿态的多视图图像（Gen6D、OnePose）。对非刚体动画物体，仅人体姿态经 SMPL 等方法被广泛研究，限制了对动物等一般 4D 物体的估计。更简单的替代是仅用深度模型（DepthCrafter、Video Depth Anything、GeoWizard）表示 3D 位置。但从前背景分割前景实体存在误差，且无法生成一致的视频度量深度。

为规避上述挑战，我们选择通过 Unreal Engine（UE）与先进渲染技术构建合成数据集，名为 *360°-Motion*（见图 3）。首先收集 70 个动画 3D 资产，覆盖人类与动物两类。人类按性别、服装、体型、发型等属性区分。随后用 GPT-4V 为每张渲染资产图像生成文本描述 $\mathbf{e}_n \in \mathcal{Y}^{L_n}$（$L_n \leq 20$）（图 3(a)）。对带姿态的物体轨迹模板（图 3(b)），我们遵循 TC4D：用 GPT 生成 3D spline（位置 $\mathbf{T}$），并通过对 spline 求梯度得到额外朝向 $\mathbf{R}$。该过程在规范空间得到约 96 个模板，每个模板关联 1 到 3 个资产。另外把动物尺寸缩小为 0.6 倍，以防与其他资产碰撞。成对资产及其运动模板被放置在某一 3D 平台中 $5\times 5$ 平方米范围内，平台包括城市（MatrixCity）、沙漠、森林以及投影到 3D 的 HDRI。我们在场景周围均匀布置 12 组相机以捕捉 360 度视角，每个相机产出 100 帧、分辨率 $384\times 672$ 的视频片段。通过对各类物体与轨迹进行排列组合，该过程共产生 54,000 段视频。（见附录第 A.5.1 节及补充视频样例。）

**Video Domain Adaptor。** 在这批相对较小的自建视频片段上训练视频扩散模型，容易学到不希望出现的 UE 风格，从而限制泛化。为避免学习这种质量变化并保留基座 T2V 的知识，我们训练 LoRA 模块作为 video domain adaptor。具体地，把 LoRA 接入基座 T2V 模型的 self-attention、cross-attention 与 linear 层，如图 2 所示。Attention / linear 投影矩阵 $\{\mathbf{W}_n\}_{n=1}^{K}$ 配以额外可训练低秩矩阵 $\{\Delta\mathbf{W}_n=\alpha \mathbf{A}_n \mathbf{B}_n^T\}_{n=1}^{K}$，其中 $\alpha$ 为可调尺度，用于控制 adaptor 影响。推理时把 $\alpha$ 设为较小值，以减轻合成视频数据的负面影响。我们用如下目标优化 $\boldsymbol{\theta}_2=\{\Delta\mathbf{W}_n\}_{n=1}^{K}$：

$$
\mathcal{L}(\boldsymbol{\theta}_2)=\mathbb{E}_{\mathbf{x}, \mathbf{c}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_{t}^2 \mathbf{I}), t}\Bigl[\bigl\|\boldsymbol{\epsilon}-\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}_1}(\mathbf{x}_t, \mathbf{c}, t, \alpha )\bigr\|_2^2\Bigr].
$$

注意：训练 object injector $\boldsymbol{\theta}_1$ 时，domain adaptor $\boldsymbol{\theta}_2$ 被冻结。

### 3.4 推理过程

我们把视频潜变量 $\hat{\mathbf{x}}_T$ 初始化为标准高斯噪声，并在目标实体–轨迹对 $(\mathbf{e}_n,\mathbf{P}_n)_{n=1}^{N}$ 的引导下逐步去噪，调度与前两个训练阶段相同。我们使用 classifier-free guidance（CFG），并用 DDIM 做重间隔采样以加速。为进一步提升视频质量，采用 annealed sampling 策略（算法 1）：推理前期把轨迹插入模型以定义总体物体运动；后期将其 dropout，转入标准 T2V 生成过程。我们还观察到：把负向 3D 轨迹设为静态运动 $\{(\hat{\mathbf{P}}_n)_{n=1}^{N}\mid\hat{\mathbf{P}}_n=\mathbf{P}_0,\forall n\}$ 可进一步提升姿态精度。该现象反映出模型学习 3D 运动表示的能力：由于训练时不像文本那样随机 dropout 运动序列，模型从实体主要处于运动状态的视频中隐式学到了静态运动建模。因此把静态运动当作 “negative motion prompt” 时，可放大实体位移幅度，从而在评估中提升姿态精度。但我们并未采用该做法，因为它有时导致视频质量下降（见附录第 A.6.3 节）。

**算法 1.** 带 classifier-free guidance（CFG）的 annealed 条件采样。

**输入：** $w$：guidance 强度；$T_c$：annealed timestep；$\alpha$：LoRA 调制系数；$\tilde{\boldsymbol{\theta}}$：冻结的基座 T2V 模型；$\boldsymbol{\theta}_1$：object injector；$\boldsymbol{\theta}_2$：domain adaptor；$\mathbf{c}$：文本条件；$(\mathbf{e},\mathbf{P})$：实体–轨迹对。

1. $\hat{\mathbf{x}}_1 \sim \mathcal{N}(\mathbf{0}, \sigma_{t}^2 \mathbf{I})$
2. **for** $t=1,\ldots,T$ **do**
3. $\quad$ **if** $t \leq T_c$ **then**
4. $\qquad$ $\tilde{\epsilon}_t=(1+w) \hat{\boldsymbol{\epsilon}}_{\tilde{\boldsymbol{\theta}},\boldsymbol{\theta}_1,\boldsymbol{\theta}_2}\bigl(\hat{\mathbf{x}}_t, \mathbf{c}, (\mathbf{e}_n,\mathbf{P}_n)_{n=1}^N, \alpha\bigr)-w \hat{\boldsymbol{\epsilon}}_{\tilde{\boldsymbol{\theta}},\boldsymbol{\theta}_1,\boldsymbol{\theta}_2}(\hat{\mathbf{x}}_t, \alpha)$
5. $\quad$ **else**
6. $\qquad$ $\hat{\epsilon}_t=(1+w) \boldsymbol{\epsilon}_{\tilde{\boldsymbol{\theta}}}(\hat{\mathbf{x}}_t, \mathbf{c})-w\boldsymbol{\epsilon}_{\tilde{\boldsymbol{\theta}}}(\hat{\mathbf{x}}_t)$
7. $\quad$ **end if**
8. $\quad$ $\hat{\boldsymbol{z}}_t = (\hat{\mathbf{x}}_t-\sigma_t \tilde{\boldsymbol{\epsilon}}_t) / \alpha_t$
9. $\quad$ 若 $t < T$，则 $\hat{\mathbf{x}}_{t+1} \sim \mathcal{N}\bigl(\hat{\mathbf{x}}_{t+1} ; \tilde{\boldsymbol{\mu}}_{t+1 \mid t}(\hat{\boldsymbol{z}}_t, \hat{\mathbf{x}}_t), \sigma_{t+1 \mid t}^2 \mathbf{I}\bigr)$；否则 $\hat{\mathbf{x}}_{t+1}=\hat{\boldsymbol{z}}_t$
10. **end for**
11. **return** $\hat{\mathbf{x}}_{t+1}$

---

## 4. 实验

### 4.1 实现细节

输入文本提示使用统一模板：“*{Entity 1},..., and {Entity N} are moving in the {Location}.*” 其中 “{Location}” 按对应 3D UE 平台设定。我们基于内部视频扩散模型训练 3DTrajMaster（研究用途，详见附录第 A.1 节），参数量约 1B。裁剪后的训练视频与推理视频分辨率设为 $384\times 672$。每段视频 5 秒。使用 Adam 优化器，在 8 张 NVIDIA H800 GPU 上训练，学习率 $5 \times 10^{-5}$，batch size 为 8。Domain adaptor 训练 50,000 步，object injector 再训练 36,000 步。推理时 DDIM 步数为 50，CFG 为 12.5。

### 4.2 基线

我们把 3DTrajMaster 与现有、能定制物体运动的 SOTA 方法比较：MotionCtrl、Direct-a-Video 与 Tora。基线按其官方开源代码库中的最佳性能设置配置。

### 4.3 评估指标

1. *轨迹精度*：由于缺少开放世界 4D 物体的姿态估计器，评估仅限于人体目标。具体地，用 GVHMR 估计人体姿态 $\{(\mathbf{R}^{\mathrm{est}}_n,\mathbf{T}^{\mathrm{est}}_n)\}_{n=1}^{F}$，并与输入姿态序列 $\{(\mathbf{R}^{\mathrm{gt}}_n,\mathbf{T}^{\mathrm{gt}}_n)\}_{n=1}^{F}$ 比较。两条轨迹在首帧位置对齐。我们遵循 CameraCtrl 估计旋转角误差 **RotErr** 与平移尺度误差 **TransErr**，但取平均而非求和。
2. *视频质量*：用标准指标评估视频外观，包括 Fréchet Video Distance（**FVD**）、Fréchet Image Distance（**FID**）以及 CLIP Similarity（**CLIPSIM**）。

### 4.4 评估数据集

1. *姿态序列*：收集 44 个新颖姿态模板，每个包含一个或多个物体运动。
2. *实体描述*：用 GPT 生成 20 条新颖人类描述、52 条新颖非人类描述以及 32 个新颖地点（见附录第 A.5.3 节），随机分配到姿态上形成 100 对（12 个单实体、72 个双实体、16 个三实体；每对含一个人类实体）。

### 4.5 对比

**粒度层级。** 如表 1 所示，3DTrajMaster 能在 3D 空间定制物体位置与朝向。相比之下，点（MotionCtrl / Tora）与 bounding box（Direct-a-Video）等 2D 运动表示缺少对 z 维的感知。处理 3D 遮挡时，这种歧义更成问题。此外，MotionCtrl 与 Tora 把多个实体融入单一 2D 特征，无法把各个实体与各自轨迹对应起来（见图 6 失败案例）。在多实体输入上测试时，Direct-a-Video（免训练范式）结果尤其弱。此外，3DTrajMaster 支持多样实体与背景（见图 4），以及对实体输入的细粒度控制（见图 5）。

**表 1. 多实体输入下的细粒度控制对比。**

| | Location | Orientation | Entity-Traj. Corresp. | Learning-based? |
| --- | --- | --- | --- | --- |
| Direct-a-Video | ✓（2D） | ✗ | ✓ | ✗ |
| MotionCtrl / Tora | ✓（2D） | ✗ | ✗ | ✓（未解耦） |
| 3DTrajMaster（本文） | **✓（3D）** | ✓ | ✓ | ✓（解耦） |

![图 4. 实体与背景的多样性。3DTrajMaster 能控制多样实体（人、动物、汽车、机器人，甚至抽象自然力），同时生成多样地点。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/diverse_entity_bg.png)

![图 5. 对人类实体输入的细粒度编辑。3DTrajMaster 支持修改发型、服装、体型等属性。（更多见附录图 S4。）](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/diffhuman.png)

![图 6. 单实体 / 多实体运动的定性对比。3DTrajMaster 通过建模 6DoF 实体运动优于所有 2D 基线，更能表达运动固有的 3D 本质。最后一图中，Tora 误把背景实体当成女孩实体。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/main_comparison.png)

**表 2. 单实体 / 多实体运动的定量对比。** 3DTrajMaster 在多实体输入上表现更好，因为单实体轨迹更复杂。

| Methods | Single Entity TransErr (m)  | Single Entity RotErr (deg)  | Multiple Entities TransErr (m)  | Multiple Entities RotErr (deg)  | All Entities TransErr (m)  | All Entities RotErr (deg)  |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base T2V | 1.946 | 1.799 | 1.586 | 1.208 | 1.629 | 1.279 |
| MotionCtrl | 1.752 | 2.134 | 1.682 | 1.613 | 1.690 | 1.675 |
| Tora | 1.707 | 1.158 | 1.867 | 1.514 | 1.848 | 1.471 |
| Direct-a-Video | 1.632 | 1.902 | 1.391 | 0.942 | 1.420 | 1.057 |
| 3DTrajMaster | **0.456** | **0.319** | **0.390** | **0.272** | **0.398** | **0.277** |

**定量与定性结果。** 为对齐 MotionCtrl 与 Direct-a-Video 的输入要求，我们把 3D 姿态轨迹投影到 2D 空间。对基线，我们简化实体描述，例如把 “a man with messy black hair, tall frame, a red shirt” 改成 “a man” 或 “a man in red”；否则它们可能无法用细粒度描述生成视频。如图 6 所示，在单实体设定中，3DTrajMaster 生成精确的实体运动，例如 180° 回头与连续向内 90° 转身。相比之下，Tora 与 Direct-a-Video 产生更简单的运动，仅把物体从左移到右或移到右上。在多实体基准上，3DTrajMaster 成功处理 3D 遮挡，例如男人走在斑马前方；Direct-a-Video 则在重叠区域失败，人与斑马混在一起。指标结果见表 2。本文显著优于所有基线，并不令人意外。

### 4.6 消融实验

**表 3. 完整测试集上的消融，并以 Base T2V 视频作为参考视频。**

| Ablation Setting | FVD  | FID  | CLIPSIM $\uparrow$ | TransErr (m)  | RotErr (deg)  |
| --- | ---: | ---: | ---: | ---: | ---: |
| w/ Cross-Attn. Fusion | 1673.24 | 102.13 | 32.87 | 0.453 | 0.341 |
| w/ 3D Self-Attn. | 1597.51 | 98.74 | 33.15 | 0.427 | 0.296 |
| w/o Domain Adaptor | 2379.89 | 157.51 | 30.50 | 0.415 | 0.301 |
| w/o Annealed Sampl. | 1841.64 | 112.57 | 32.26 | 0.407 | **0.265** |
| Full Model | **1546.15** | **96.75** | **33.77** | **0.398** | 0.277 |

![图 7. Domain adaptor（上）与 annealed sampling（下）的消融结果。附录第 A.6.2 节提供更多实验，用于选择合适的 $\alpha$ 与 $T_c$ 以提升视频质量。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/ablation_video_quality.png)

**提升视频质量。** 如图 7 与表 3 所示，去掉 video domain adaptor 后，视频质量显著恶化，外观退回到与训练集类似的纯 UE 风格。同样，省略 annealed sampling 也会导致视频质量下降（见狮子胡须与整体场景风格）。旋转精度略有下降（$0.277\rightarrow 0.265$），这可以接受，因为开放世界人体姿态评估本身存在误差。

**运动融合设计。** 如表 3 所示，用 cross-attention fusion 替换 gated self-attention（w/ Cross-Attn. Fusion；此处以实体–运动绑定特征 $\mathbf{Z}^{\mathbf{Pe}}$ 作为 query），或把 object injector 放在 3D self-attention 层之后（w/ 3D Self-Attn.），都会使视频质量与姿态序列精度略有下降。

---

## 5. 结论

本文提出 3DTrajMaster：一个在 3D 空间控制多实体运动的统一框架，运动表示为 6DoF 位置与旋转序列。灵活的 object injector 建立实体级对应，并允许灵活编辑实体描述。

**局限。** 可泛化实体（如动物）无法达到与人类同等粒度的编辑。该局限可通过构建同一类别中更多样、更细致的 3D 资产来缓解。当前模型受限于全局运动模式；但细粒度局部运动（例如人跳舞或挥手）以及不同实体间的交互（例如人抱起狗）也可以用与我们的 6DoF 运动类似的结构化运动模式来建模。目前模型一次只能生成有限数量的实体（$\leq 3$），但可随更强的视频基础模型与配对数据而改进。

---

## 致谢

感谢快手科技的 Jinwen Cao、Yisong Guo、Haowen Ji、Jichao Wang 与 Yi Wang 在构建 360°-Motion Dataset 上的帮助。讨论方面，感谢 Yuzhou Huang、Qinghe Wang、Runsen Xu、Zeqi Xiao 与 Zhouxia Wang。

---

## 附录

### A.1 研究用内部视频扩散模型

我们的模型是基于 transformer 的潜空间扩散模型，如图 S1 所示。首先用 3D VAE 把视频从像素级变换到潜空间，再在其上构建基于 transformer 的视频扩散模型（DiT）。先前依赖 UNet（Stable Video Diffusion、VideoCrafter、AnimateDiff）或 transformer（Latte）的模型通常额外加入 1D temporal attention 以生成视频；这种时空分离设计并不能得到最优结果。我们改为用 3D self-attention 替换 1D temporal attention，使模型更有效地感知与处理时空 token，从而得到高质量、连贯的视频生成模型。具体地，我们把 timestep 映射为一个尺度，从而在每个 attention 或 feed-forward network（FFN）模块之前对时空 token 施加 RMSNorm。

![图 S1. 本文的视频潜扩散模型骨干。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/DiT_backbone.png)

### A.2 补充相关工作

**把控制注入视频基础模型。**

*(1) 基于学习：* 控制信号通常经额外 encoder（例如可学习卷积 / linear / attention / LoRA 层，或冻结的预训练特征 encoder）投影为潜嵌入，再通过拼接、相加或插入整合进基座模型架构。VideoComposer 用统一的 STC-encoder 与 CLIP 模型，把多模态输入条件（文本、空间、时间）送入基座 T2V 模型。MotionCtrl 通过微调基座 U-Net 的特定层引入相机运动，并通过额外卷积层引入物体运动。CameraCtrl 进一步引入 ControlNet 的思路：用基于 attention 的 pose encoder，以 Plücker embedding 形式融合相机信号，同时冻结基座模型。类似地，SparseCtrl 学习附加 encoder，把控制信号（RGB、草图、深度）接入基座模型。Tora 用 trajectory encoder 与即插即用 motion fuser，把 2D 轨迹与基座视频模型融合。MotionDirector 利用空间与时间 LoRA 层，从参考视频学习期望运动模式。

*(2) 免训练：* 这些方法修改 attention 层或视频潜变量，以计算高效的方式调节控制信号。但免训练方法往往泛化差，且需要大量试错。Direct-a-Video 通过放大或抑制 spatial cross-attention 中的 attention 来注入 box 引导；FreeTraj 把目标轨迹嵌入低频分量，并重新设计各 attention 层的重加权策略。MOFT 通过去除内容相关并做运动通道滤波来提取运动先验，再用参考 MOFT 改变采样过程。

### A.3 补充应用

我们概述方法在若干方向上的潜在应用如下。

1. **电影：** 复现角色的经典动作。可从给定视频提取人体姿态，再利用本模型能力将其应用到不同实体与背景。
2. **自动驾驶：** 仿真危险安全事故，例如两车相撞、车撞人。
3. **具身智能：** 用多样实体与轨迹输入生成大量视频，以训练通用 4D 姿态估计器，尤其针对非刚体。
4. **游戏：** 通过 LoRA 训练角色 ID（例如《黑神话：悟空》），再用不同轨迹驱动角色运动。

### A.4 关于有限实体数量（$\leq 3$）的说明

如正文“局限”所述，当前方法最多生成 3 个实体。该约束主要来自视频基础模型的能力，而非训练数据。在视频中生成同类的 $\gg 2$ 个实体（例如 “a group of people/cars/animals”）相对容易；但要通过文本输入生成彼此差异很大的 $\gg 2$ 个实体则困难得多，因为 T5 text encoder 倾向于把不同实体的文本特征混在一起，从而难以把特定轨迹与对应文本实体关联。基于对视频基础模型的经验研究，我们在工作中把实体数量限制为 3。就数据构建而言，在 UE 平台流程中纳入更多带配对轨迹的实体并不困难。关键瓶颈是视频基础模型难以同时生成如此多样的实体集合。此外，Tora、MotionCtrl、Direct-a-Video 等先前工作也聚焦于有限数量的实体。

### A.5 数据集说明

#### A.5.1 360°-Motion Dataset 数据

图 S2 给出用 12 个均匀环绕相机捕捉的一个样本。每个相机拍摄 100 帧、分辨率 $384\times 672$ 的片段。训练时丢弃前 10 帧，以消除 UE 平台中 3D 模型初始化可能带来的模糊与噪声。

![图 S2. 360°-Motion 中由 12 个均匀环绕相机捕捉的一个样本。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/supp_dataset.png)

#### A.5.2 常见视频数据集中的实体分布不平衡

在 Artgrid、Pixabay、Pexels 等高质量视频数据集中（Artgrid：https://artgrid.io/ ，Pixabay：https://www.videvo.net/ ，Pexels：https://www.pexels.com/ ），类别不平衡非常显著，并带来重大挑战。我们先用 QWen-VL 为上述三个数据集的视频做 caption，再用 spaCy（https://spacy.io/ ）从视频 caption 中提取名词块作为实体词。我们预定义 60 余类作为实体过滤关键词。如图 S3 所示，某些类别（例如人类）在实体对象中占比过大，从而限制模型向出现频率较低的其他类别泛化。

![图 S3. Artgrid、Pixabay 与 Pexels 中超过 60 类的实体分布。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/dataset_distribution.png)

#### A.5.3 GPT 生成的评估提示

评估用的人类提示、非人类（动物、汽车、机器人）提示以及地点提示分别见表 R1、表 R2 与表 R3、以及表 R4。提示原文为英文评估输入，此处保留原句。

**表 R1. 评估用人类提示。** 由 GPT 提示生成：“Generate more human samples similar to {Train Human Sample}, no more than 25 words.”

| # | Prompt |
| ---: | --- |
| 1 | a man with short spiky brown hair, athletic build, a navy blue jacket, beige cargo pants, and black sneakers |
| 2 | a woman with long wavy blonde hair, petite figure, a red floral dress, white sandals, and a yellow shoulder bag |
| 3 | a man with a shaved head, broad shoulders, a gray graphic t-shirt, dark jeans, and brown leather boots |
| 4 | a woman with shoulder-length straight auburn hair, a slender figure, a green button-up blouse, black leggings, and white sneakers |
| 5 | a man with messy black hair, tall frame, a plaid red and black shirt, faded blue jeans, and tan hiking boots |
| 6 | a man with medium-length straight brown hair, tall and slender, a gray crew-neck t-shirt, beige trousers, and dark green sneakers |
| 7 | a woman with short curly black hair, slender build, a pink hoodie, light gray joggers, and blue sneakers |
| 8 | a man with short black wavy hair, lean figure, a green and yellow plaid shirt, dark brown pants, and black suede shoes |
| 9 | a man with curly black hair, muscular build, a dark green hoodie, gray joggers, and white running shoes |
| 10 | a woman with short blonde hair, slim athletic build, a red leather jacket, dark blue jeans, and white sneakers |
| 11 | a man with medium-length wavy brown hair, lean build, a black bomber jacket, olive green cargo pants, and brown hiking boots |
| 12 | a man with buzz-cut blonde hair, stocky build, a gray zip-up sweater, black shorts, and red basketball shoes |
| 13 | a woman with long straight black hair, toned build, a blue denim jacket, light gray leggings, and black slip-on shoes |
| 14 | a man with short curly red hair, average build, a black leather jacket, dark blue cargo pants, and white sneakers |
| 15 | a woman with shoulder-length wavy brown hair, slim build, a green parka, black leggings, and gray hiking boots |
| 16 | a man with short straight black hair, tall and lean build, a navy blue sweater, khaki shorts, and brown sandals |
| 17 | a woman with pixie-cut blonde hair, athletic build, a red windbreaker, blue ripped jeans, and black combat boots |
| 18 | a man with medium-length wavy gray hair, muscular build, a maroon t-shirt, beige chinos, and brown loafers |
| 19 | a woman with long curly black hair, average build, a purple hoodie, black athletic shorts, and white running shoes |
| 20 | a man with short spiky blonde hair, slim build, a black trench coat, blue jeans, and brown hiking shoes |

**表 R2. 评估用非人类提示（1/2）。** 由 GPT 提示生成：“Generate more animal/car/robot samples similar to {Train Sample}, no more than 25 words.”

| # | Prompt |
| ---: | --- |
| 1 | a dog with a fluffy coat, wagging tail, and warm golden-brown fur, exuding a gentle and friendly charm |
| 2 | a tiger with vibrant orange and black stripes, piercing yellow eyes, and a powerful stance, exuding strength and grace |
| 3 | a giraffe with golden-yellow fur, long legs, a tall slender neck, and patches of brown spots, exuding elegance and calm |
| 4 | an alpaca with soft white wool, short legs, a thick neck, and a fluffy head of fur, radiating gentle charm |
| 5 | a zebra with black and white stripes, sturdy legs, a short neck, and a sleek mane running down its back |
| 6 | a deer with sleek tan fur, long slender legs, a graceful neck, and tiny antlers atop its head |
| 7 | a gazelle with light golden fur, long slender legs, a thin neck, and short, sharp horns, embodying elegance and agility |
| 8 | a horse with chestnut brown fur, muscular legs, a slim neck, and a flowing mane, exuding strength and grace |
| 9 | a sleek black panther with a smooth, glossy coat, emerald green eyes, and a powerful stance |
| 10 | a cheetah with golden fur covered in black spots, intense amber eyes, and a slender, agile body |
| 11 | a regal lion with a thick, flowing golden mane, sharp brown eyes, and a powerful muscular frame |
| 12 | a snow leopard with pale gray fur adorned with dark rosettes, icy blue eyes, and a stealthy, poised posture |
| 13 | a jaguar with a golden-yellow coat dotted with intricate black rosettes, deep green eyes, and a muscular build |
| 14 | a wolf with thick silver-gray fur, alert golden eyes, and a lean yet strong body, exuding confidence and boldness |
| 15 | a tiger with a pristine white coat marked by bold black stripes, bright blue eyes, and a graceful, poised form |
| 16 | a lynx with tufted ears, soft reddish-brown fur with faint spots, and intense yellow-green eyes |
| 17 | a bear with dark brown fur, small but fierce black eyes, and a broad and muscular build, radiating power |
| 18 | a swift fox with reddish-orange fur, a bushy tail tipped with white, and sharp, intelligent amber eyes |
| 19 | a falcon with blue-gray feathers, sharp talons, and keen yellow eyes fixed on its prey below |
| 20 | a fox with sleek russet fur, a bushy tail tipped with black, and bright green and cunning eyes |
| 21 | a kangaroo with brown fur, powerful hind legs, and a muscular tail, showcasing its strength and agility |
| 22 | a polar bear with thick white fur, strong paws, and a black nose, embodying the essence of the Arctic |
| 23 | a cheetah with a slender build, spotted golden fur, and sharp eyes, epitomizing speed and agility |
| 24 | a dolphin with sleek grey skin, a curved dorsal fin, and intelligent, playful eyes, reflecting its nature |
| 25 | a wolf with a body covered in thick silver fur, sharp ears, and piercing yellow eyes, showcasing its alertness |
| 26 | a leopard with a body covered in golden fur, dark rosettes, and a long muscular tail, emphasizing its strength |
| 27 | a penguin with a body covered in smooth black-and-white feathers, short wings, and webbed feet |
| 28 | a gazelle with a body covered in sleek tan fur, long legs, and elegant curved horns, showcasing its grace |

**表 R3. 评估用非人类提示（2/2）。** 由 GPT 提示生成：“Generate more animal/car/robot samples similar to {Train Sample}, no more than 25 words.”

| # | Prompt |
| ---: | --- |
| 29 | a rabbit with a body covered in soft fur, quick hops, and a playful demeanor, showcasing its energy |
| 30 | a koala with a body covered in soft grey fur, large round ears, and a black nose, radiating cuteness |
| 31 | a rhinoceros with a body covered in thick grey skin, a massive horn on its snout, and sturdy legs |
| 32 | a flamingo with a body covered in pink feathers, long slender legs, and a gracefully curved neck |
| 33 | a parrot with bright red, blue, and yellow feathers, a curved beak, and sharp eyes |
| 34 | a hippopotamus with a body covered in thick grey-brown skin, massive jaws, and a large body |
| 35 | a crocodile with a body covered in scaly green skin, a powerful tail, and sharp teeth |
| 36 | a moose with a body covered in thick brown fur, massive antlers, and a bulky frame |
| 37 | a fluttering butterfly with intricate wing patterns, vivid colors, and graceful flight |
| 38 | a chameleon with a body covered in vibrant green scales, bulging eyes, and a curled tail, showcasing its unique charm |
| 39 | a lemur with a body covered in soft grey fur, a ringed tail, and wide yellow eyes, and curious expression |
| 40 | a squirrel with a body covered in bushy red fur, large eyes, and a fluffy tail |
| 41 | a panda with a body covered in fluffy black-and-white fur, a round face, and gentle eyes, radiating warmth |
| 42 | a porcupine with a body covered in spiky brown quills, a small nose, and curious eyes |
| 43 | a sedan with a sleek metallic silver body, long wheelbase, a low-profile hood, and a small rear spoiler |
| 44 | an SUV with a matte black exterior, elevated suspension, a tall roofline, and a compact rear roof rack |
| 45 | a pickup truck with rugged dark green paint, extended cab, raised suspension, and a modest cargo bed cover |
| 46 | a vintage convertible with a body covered in shiny red paint, chrome bumpers, and a stylish design |
| 47 | a futuristic electric car with a minimalist silver design, slim LED lights, and smooth curves |
| 48 | a compact electric vehicle with a silver finish, aerodynamic profile, and efficient battery |
| 49 | a firefighting robot with a water cannon arm, heat sensors, and durable red-and-silver exterior |
| 50 | an industrial welding robot with articulated arms, a laser precision welder, and heat-resistant shields |
| 51 | a disaster rescue robot with reinforced limbs, advanced AI, and a rugged body designed to navigate |
| 52 | an exploration rover robot with solar panels, durable wheels, and advanced sensors for planetary exploration |

**表 R4. 评估用地点提示。**

| # | Location | # | Location | # | Location | # | Location |
| ---: | --- | ---: | --- | ---: | --- | ---: | --- |
| 1 | fjord | 2 | sunset beach | 3 | cave | 4 | snowy tundra |
| 5 | prairie | 6 | asian town | 7 | rainforest | 8 | canyon |
| 9 | savanna | 10 | urban rooftop garden | 11 | swamp | 12 | riverbank |
| 13 | coral reef | 14 | volcanic landscape | 15 | wind farm | 16 | town street |
| 17 | night city square | 18 | mall lobby | 19 | glacier | 20 | seaside street |
| 21 | gymnastics room | 22 | abandoned factory | 23 | autumn forest | 24 | mountain village |
| 25 | coastal harbor | 26 | ancient ruins | 27 | modern metropolis | 28 | dessert |
| 29 | forest | 30 | city | 31 | snowy street | 32 | park |

### A.6 更多实验

#### A.6.1 细粒度实体提示输入

图 S4 给出更多样本，表明 3DTrajMaster 支持细粒度实体定制。男人的描述可通过调整发型、性别、体型、服装与配饰等属性灵活修改。

![图 S4. 输入文本提示中的灵活实体编辑。另一实体 “a swift falcon with blue-gray feathers, sharp talons, and keen yellow eyes focused on its prey below” 保持固定，同时变化人类实体描述。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/supp_diffhuman.png)

#### A.6.2 最优超参数

正文提出 video domain adaptor 与 annealed sampling，以缓解自建 UE 数据集带来的视频域偏移。然而，完全去掉 LoRA adaptor（学到的运动与域偏差在一定程度上耦合）或去掉插入的运动引导，都会导致 3D 轨迹精度下降。因此，在适当 dropout 下应用视频增强技术至关重要。我们从随机初始化的参数组开始：$T_{c}=10$，$\alpha=0.2$，$TS=72{,}000$。在评估子集上做消融。如表 R5、表 R6 与表 R7 所示，这些超参数增大时，视频质量呈单调下降趋势；相比之下，3D 轨迹精度起初急剧下降，后期趋于稳定。为在视觉质量下降与保持姿态精度之间取得平衡，我们选择最优参数组 $T_{c}=25$，$\alpha=0.4$，$TS=36{,}000$ 作为默认推理设置。

**表 R5. Annealed timestep $T_c$ 的消融。**

| Annealed Timestep $T_c$ | FVD  | FID  | CLIPSIM $\uparrow$ | TransErr (m)  | RotErr (deg)  |
| --- | ---: | ---: | ---: | ---: | ---: |
| $T_{c}=5$ | **1492.79** | **76.95** | **0.3469** | 0.844 | 1.099 |
| $T_{c}=10$ | 1976.01 | 106.45 | 0.3429 | 0.546 | 0.493 |
| $T_{c}=15$ | 2179.15 | 122.55 | 0.3405 | 0.437 | 0.422 |
| $T_{c}=20$ | 2236.05 | 128.89 | 0.3374 | 0.391 | 0.284 |
| $T_{c}=25$ | 2240.40 | 132.90 | 0.3337 | **0.344** | 0.274 |
| $T_{c}=30$ | 2295.13 | 137.52 | 0.3314 | 0.360 | **0.261** |
| $T_{c}=35$ | 2323.20 | 142.71 | 0.3276 | 0.352 | 0.264 |
| $T_{c}=40$ | 2338.47 | 148.27 | 0.3240 | 0.351 | 0.266 |
| $T_{c}=45$ | 2363.49 | 156.39 | 0.3207 | 0.350 | 0.268 |
| $T_{c}=50$ | 2347.64 | 166.71 | 0.3185 | 0.348 | 0.281 |

**表 R6. LoRA 尺度 $\alpha$ 的消融。**

| LoRA Scalar $\alpha$ | FVD  | FID  | CLIPSIM $\uparrow$ | TransErr (m)  | RotErr (deg)  |
| --- | ---: | ---: | ---: | ---: | ---: |
| $\alpha=0$ | **1495.38** | **80.56** | **0.3467** | 0.646 | 0.900 |
| $\alpha=0.2$ | 1976.01 | 106.45 | 0.3429 | 0.546 | 0.493 |
| $\alpha=0.4$ | 2150.42 | 133.76 | 0.3367 | 0.444 | 0.428 |
| $\alpha=0.6$ | 2330.56 | 152.12 | 0.3277 | 0.394 | **0.393** |
| $\alpha=0.8$ | 2318.78 | 195.93 | 0.3125 | 0.378 | 0.450 |
| $\alpha=1.0$ | 2481.33 | 224.81 | 0.3087 | **0.358** | 0.432 |

**表 R7. 训练步数 $TS$ 的消融。**

| Train. Steps $TS$ | FVD  | FID  | CLIPSIM $\uparrow$ | TransErr (m)  | RotErr (deg)  |
| --- | ---: | ---: | ---: | ---: | ---: |
| $TS=12{,}000$ | **1493.68** | **72.03** | 0.3427 | 0.561 | 0.713 |
| $TS=36{,}000$ | 1883.15 | 99.98 | 0.3408 | 0.523 | 0.631 |
| $TS=72{,}000$ | 1976.01 | 106.45 | **0.3429** | 0.546 | 0.493 |
| $TS=108{,}000$ | 2068.43 | 111.01 | 0.3388 | 0.446 | **0.480** |
| $TS=144{,}000$ | 2102.28 | 114.84 | 0.3367 | **0.411** | 0.482 |

#### A.6.3 负向姿态条件设为静态运动

我们发现：把负向姿态序列设为静态运动 $\{(\hat{\mathbf{P}}_n)_{n=1}^{N}\mid\hat{\mathbf{P}}_n=\mathbf{P}_0,\forall n\}$，而非正向运动序列 $\{(\mathbf{P}_n)_{n=1}^{N}\}$，可进一步提升姿态精度，见表 R8。我们推断模型从随机生成的 3D 轨迹中捕捉到了底层 3D 运动表示。但由于视频质量下降，我们未采用该做法。

**表 R8. 负向姿态序列的消融。**

| Negative Condition | FVD  | FID  | CLIPSIM $\uparrow$ | TransErr (m)  | RotErr (deg)  |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neg. Pose = Static Motions | 2141.39 | 118.22 | 0.3360 | **0.371** | **0.448** |
| Neg. Pose = Pos. Pose | **1976.01** | **106.45** | **0.3429** | 0.546 | 0.493 |

#### A.6.4 来自人类用户的定性反馈

我们进行问卷调查，收集 53 份样本以形成用户偏好对比。每位参与者获得 0.80 美元报酬，约用 5 分钟完成问卷。问卷评估四个维度：(1) 视频质量；(2) 轨迹精度；(3) 实体多样性；(4) 背景多样性。表 R9 报告相对基线更偏好本文模型的用户比例。

**表 R9. 用户偏好对比。**

| Method | MotionCtrl | Direct-a-Video | Tora |
| --- | ---: | ---: | ---: |
| 3DTrajMaster | 47.2% | 56.6% | 81.1% |

#### A.6.5 可泛化的实体提示与 3D 轨迹

我们给出更多由 GPT 生成的新颖实体提示与 3D 轨迹上的泛化结果，见图 S5 至图 S24。每条文本提示包含 1 到 3 个实体。（请读者到项目网站查看可视化结果。）

![图 S5. 新颖 3D 轨迹与实体提示上的泛化结果（1/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/single_entity_1.png)

![图 S6. 新颖 3D 轨迹与实体提示上的泛化结果（2/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/single_entity_2.png)

![图 S7. 新颖 3D 轨迹与实体提示上的泛化结果（3/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/single_entity_3.png)

![图 S8. 新颖 3D 轨迹与实体提示上的泛化结果（4/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/single_entity_4.png)

![图 S9. 新颖 3D 轨迹与实体提示上的泛化结果（5/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_1.png)

![图 S10. 新颖 3D 轨迹与实体提示上的泛化结果（6/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_2.png)

![图 S11. 新颖 3D 轨迹与实体提示上的泛化结果（7/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_3.png)

![图 S12. 新颖 3D 轨迹与实体提示上的泛化结果（8/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_4.png)

![图 S13. 新颖 3D 轨迹与实体提示上的泛化结果（9/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_5.png)

![图 S14. 新颖 3D 轨迹与实体提示上的泛化结果（10/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_6.png)

![图 S15. 新颖 3D 轨迹与实体提示上的泛化结果（11/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_7.png)

![图 S16. 新颖 3D 轨迹与实体提示上的泛化结果（12/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_8.png)

![图 S17. 新颖 3D 轨迹与实体提示上的泛化结果（13/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_9.png)

![图 S18. 新颖 3D 轨迹与实体提示上的泛化结果（14/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_10.png)

![图 S19. 新颖 3D 轨迹与实体提示上的泛化结果（15/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/two_entity_11.png)

![图 S20. 新颖 3D 轨迹与实体提示上的泛化结果（16/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/three_entity_1.png)

![图 S21. 新颖 3D 轨迹与实体提示上的泛化结果（17/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/three_entity_2.png)

![图 S22. 新颖 3D 轨迹与实体提示上的泛化结果（18/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/three_entity_3.png)

![图 S23. 新颖 3D 轨迹与实体提示上的泛化结果（19/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/three_entity_4.png)

![图 S24. 新颖 3D 轨迹与实体提示上的泛化结果（20/20）。](../../../arxiv/geometry_control/3dtrajmaster/extracted/figs/supp/three_entity_5.png)
