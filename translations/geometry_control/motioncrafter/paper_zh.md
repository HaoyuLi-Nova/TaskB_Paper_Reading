# MotionCrafter：用 4D VAE 做稠密几何与运动重建

**作者：** Ruijie Zhu$^{1,2}$、Jiahao Lu$^{3}$、Wenbo Hu$^{2\dagger}$、Xiaoguang Han$^{4}$、Jianfei Cai$^{5}$、Ying Shan$^{2}$、Chuanxia Zheng$^{1}$

**机构：** $^{1}$南洋理工大学（NTU）；$^{2}$腾讯 PCG ARC Lab；$^{3}$香港科技大学（HKUST）；$^{4}$香港中文大学（深圳）（CUHK(SZ)）；$^{5}$莫纳什大学（Monash University）

**贡献说明：** $\dagger$通讯作者

**出处：** CVPR 2026（arXiv:2602.08961）

**arXiv：** [2602.08961](https://arxiv.org/abs/2602.08961)

**项目页：** https://ruijiezhu94.github.io/MotionCrafter_Page/

**原文 TeX：** [`arxiv/geometry_control/motioncrafter/extracted/main.tex`](../../../arxiv/geometry_control/motioncrafter/extracted/main.tex)

---

## 摘要

我们提出 **MotionCrafter**：一个利用 video generator，从单目视频联合重建 4D 几何并估计稠密运动的框架。关键想法是：在共享坐标系中联合表示稠密 3D point map 与 3D scene flow，并设计一个专门学习该表示的 4D VAE。先前工作把 3D 数值与 latent 严格对齐到 RGB VAE 的 latent——尽管二者分布本质不同——我们表明这种对齐并非必要，甚至会损害性能。我们提出新的数据归一化与 VAE 训练策略，能更好迁移 diffusion prior，并显著提升重建质量。在多个数据集上的大量实验表明，MotionCrafter 在几何重建与稠密 scene flow 估计上均达到 state-of-the-art，几何与运动重建分别提升 38.64% 与 25.0%，且无需任何 post-optimization。项目页：https://ruijiezhu94.github.io/MotionCrafter_Page/ 。

---

![图 1. MotionCrafter 是基于 video diffusion 的联合稠密几何与运动重建框架。给定单目视频，它在共享世界坐标系中同时预测每一帧的稠密 point map 与 scene flow；效果优于基于优化的方案，且无需任何 post-optimization。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/teaser_figure.png)

---

## 1. 引言

我们考虑如下问题：从动态场景的单目 RGB 视频中，以 feed-forward 方式同时重建 *4D 场景几何* 并估计 *稠密点运动*。该形式化对应物理世界的运作方式：物体由其在 3D 空间中的几何结构，以及随时间的运动所共同刻画。实现这一目标极具挑战：单目 4D 重建本质是 ill-posed 的，稠密时间对应仍然困难，尤其在遮挡与大幅运动下。但一旦成功，应用面很广，从视频理解到机器人（DeepVO、VINS），再到 world model。

传统方法先找跨时间的像素对应，再迭代优化 3D mesh 以拟合 RGB(D) 观测（DynamicFusion、VolumeDeform 等）。它们往往受传感器噪声限制，且需要逐场景优化，泛化较差。深度学习时代，该问题通常被拆成两个子任务：动态几何重建（MonST3R、Continuous 4D、Geo4D）与对应估计（RAFT、CoTracker）。二者本就相关，都依赖多视图几何中的像素对应。

近期 feed-forward 方法，如 St4RTrack、Dynamic Point Maps 与 Stereo4D，把 DUSt3R、MASt3R 等 *静态* 3D 重建网络扩展到动态场景，通过预测目标时刻的 point map 来处理该问题。即便如此，它们一次只处理 *成对* 帧，并依赖 post-optimization 对齐结果，削弱了对长程运动一致性的刻画。

本文提出 **MotionCrafter**：利用 video generator，对较长的单目视频序列以 feed-forward 方式联合重建 4D 几何并估计稠密运动，*无需任何 post-optimization*。为此我们提出一种 *world-centric* 4D 表示：用世界坐标系中定义的一系列 point map（DUSt3R、VGGT、Geo4D）以及对应的 scene flow 来描述动态场景。该表示直观且有效：去掉相机引起的运动分量后，静态背景点在该坐标系中理想情况下 flow 为零，从而更易学习动态物体的运动模式。相比之下，先前工作（St4RTrack、Dynamic Point Maps、Stereo4D）只预测与参考帧配对的目标时刻 point map，并不显式建模整段视频中的稠密运动。因此我们认为：要完整理解动态 3D 场景，必须在 *整段* 视频上、在共享坐标系中联合建模稠密几何与运动。

该任务的另一挑战是缺少带稠密几何与运动标注的大规模 in-the-wild 数据。沿用近期把预训练生成模型用于 3D 的趋势（Marigold、Matrix3D、Geo4D），我们不从零训练，而以预训练 video generator（SVD）为起点。该策略显著缓解数据稀缺：生成器在数十亿视觉数据上训练过。此外，video generator 本身就建模多帧时空一致性，适合捕捉长期运动对应。Geo4D 已探索用 video generator（SVD、GeometryCrafter）做 4D 重建，但它只为每帧输出 *相互独立* 的 point map，并不建模这些点之间的稠密运动。

本文更进一步，联合建模稠密几何与运动。我们把融合了 point map 与 scene flow 的统一 4D 表示编码到紧凑 latent 空间。无需在像素空间构建 cost volume（RAFT-3D）或建立稠密对应（Dynamic Point Maps），这一整合表示能高效地把 video generator 的强 prior 迁移到稠密 4D 几何与运动重建。

此外我们表明：不必把 4D 数据的取值范围严格对齐到扩散模型原始 VAE。人们通常认为这种对齐对有效利用预训练 prior 至关重要，即便对分布与自然图像差异很大的 3D 几何也是如此（Marigold、World-from-Pixels、Matrix3D、Geo4D）。我们的结果则相反。具体地，我们对 point map 采用 canonical normalization，即把 3D 坐标居中，并按场景的平均尺度缩放。尽管这与 VAE 原始 RGB 分布不对齐，MotionCrafter 仍能强泛化，并给出准确的 4D 重建与运动估计。该发现挑战了既有信念，为把 diffusion model 用于几何任务打开新可能。

总结本文主要贡献：

1. 提出 MotionCrafter：利用 video generator，从单目视频联合重建 4D 场景几何并估计稠密运动。
2. 提出一种新的 4D latent 表示，统一建模几何与运动，使模型简单有效、易于扩展。
3. 表明无需把 4D 表示严格对齐到 video diffusion 的 latent 空间也能获得强泛化，挑战了基于 diffusion 的 3D 学习中的常规看法。

![图 2. MotionCrafter 总览。我们先训练一个新的 4D VAE（右下），由 Geometry VAE 与 Motion VAE 组成；二者把 point map 与 scene flow 联合编码为统一的 4D latent。在 Diffusion U-Net 中，我们用 SVD（Stable Video Diffusion）的预训练 VAE 编码视频 latent 作为条件，再与 4D latent 沿通道拼接以引导去噪。Diffusion 版本训练时只对 4D latent 加噪。注意我们并不强制 4D latent 分布与原始 SVD VAE latent 严格对齐；该更宽松的训练策略持续提升 VAE 与 Diffusion U-Net 的泛化。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/pipeline_figure.png)

---

## 2. 相关工作

**4D Scene Reconstruction。** 早期 4D 重建主要是基于优化的方法（DynamicFusion、D-NeRF、NSFF、Nerfies、HyperNeRF、HexPlane、DynIBaR、4D-GS、Deformable 3DGS、Real-time 4DGS、MotionGS、FreeTimeGS 等）：迭代地把 4D 表示拟合到单目或多视图视频。随着 neural radiance field（NeRF）的发展，许多随时间变化的 NeRF 把可变形 3D 表示拟合到动态场景。但它们依赖昂贵的体渲染，实际应用不便。3D Gaussian Splatting（3D-GS）用基于光栅化的渲染管线避免昂贵采样。若干工作把它扩展到动态场景重建，显著缩短渲染时间。此外，一些方法借助深度 prior（DepthCrafter、Marigold、UniDepth、Depth Anything、Depth Anything V2）实现准确稳健的 4D 重建，但仍然需要逐场景优化。

近期工作探索从单目视频做 feed-forward 4D 场景重建（MonST3R、Stereo4D、Continuous 4D、Dynamic Point Maps、St4RTrack、Geo4D、4DGT、AETHER、BTimer 等）。其中 MonST3R 把著名的 *静态* 3D 重建器 DUSt3R 适配到动态场景。后续工作走类似路线，显式预测点对应。受 DUSt3R 限制，它们一次处理一对帧。为处理长单目视频，$\pi^3$ 在 VGGT 之上构建 permutation-equivariant 架构，用于静态与动态 3D 重建。Geo4D 利用 video generator（DynamiCrafter）从单目视频直接推断 4D point map。4DGT 与 BTimer 用基于 transformer 的架构预测动态 3D Gaussian 表示。但它们并不显式建模随时间的稠密点对应。

**Scene Flow Estimation。** 早期工作把点对应定义为像素空间中的 optical flow（Horn–Schunck、High Accuracy Optical Flow、Lucas–Kanade、FlowNet2、RAFT）。一条流行管线是以 coarse-to-fine 方式直接估计稠密逐像素对应（FlowNet、PWC-Net、RAFT）。但该策略在大运动或遮挡时可能失败（EpicFlow）。更近期，GMFlow 把 optical flow 估计改写为全局匹配而非局部回归。后续工作（FlowFormer、FlowFormer++）也用基于 transformer 的网络建模全局相关。不过这些方法仍处理图像空间中的 2D 点对应。本文则在世界空间中估计 3D scene flow。也有工作从图像对直接估计 3D *scene flow*，包括 RAFT-3D、SpatialTracker、SceneTracker 与 TAPVid-3D。与本文更接近的是若干近期工作（Dynamic Point Maps、St4RTrack、Stereo4D）：重建动态 3D 几何，并在世界空间估计 3D scene flow。但它们一次只处理两张图，并需要后处理精炼结果。

**Geometric Diffusion Model。** 与本文类似，许多近期工作借助现成预训练扩散模型（LDM、Video Diffusion Models、ModelScope、Make-A-Video、Align your Latents、Preserve your Own、VideoComposer、AnimateDiff、SVD、Show-1、Blue、DynamiCrafter、HunyuanVideo 等）处理 3D 任务（CAT3D、ZeroNVS、MVSplat360、ViewCrafter、Bolt3D、DSO、Amodal3R、Marigold、StableNormal、Lotus、DepthMaster、Zero-1-to-3、MVDream、Free3D、SV3D），得益于从大规模图像或视频数据中学到的丰富 prior。对 4D 重建，一条直接路线是生成多视图视频，再经逐场景优化拟合 4D 表示（SV4D、STAG4D、CAT4D）。受 score distillation sampling（SDS）启发，另一条路线从预训练 video generator 直接蒸馏 4D prior（Consistent4D、4Diffusion、DreamScene4D、DreamGaussian4D、DreamMesh4D）。但这些方法仍依赖迭代的逐场景优化，处理 in-the-wild 视频时代价高。最相关的工作是 Geo4D：微调预训练 video diffusion model，从单目视频直接推断动态 3D point map、深度与相机位姿。本文与 Geo4D 有两点不同：（1）我们在统一的 4D VAE 框架中 *同时* 重建 *动态 3D 几何* 并估计 *稠密点对应*；（2）我们表明微调扩散模型时 *不必* 对齐数据空间与 latent 空间。

---

## 3. 方法

给定含动态物体的单目视频序列，目标是学习神经网络 $f_\theta$，使其 *同时* 输出几何的 4D 表示以及稠密逐点对应：

$$
f_\theta: \{\bm{I}_i\}_{i=1}^N \rightarrow
\{\bm{X}_i, \bm{V}_{i\rightarrow i+1}\}_{i=1}^N.
\tag{1}
$$

$\mathcal{I}=\{\bm{I}_i\}_{i=1}^N$ 是含 $N$ 帧的输入单目视频，其中每帧 $\bm{I}_i\in\mathbb{R}^{H\times W\times 3}$ 为 RGB 图像。网络 $f_\theta$ 为每一帧 $i$ 预测视点不变的 point map $\bm{X}_i\in\mathbb{R}^{H\times W\times 3}$，以及相邻帧 $i$ 与 $i+1$ 之间的 3D scene flow $\bm{V}_{i\rightarrow i+1}\in\mathbb{R}^{H\times W\times 3}$。（除非另行说明，下文把 $\bm{V}_{i\rightarrow i+1}$ 简写为 $\bm{V}_{i}$。）point map 与 scene flow 都表示在共享的 *世界坐标系* 中。注意我们只预测前向 scene flow，因此最后一帧 $N$ 没有对应的 flow 预测，即不监督 $\bm{V}_{N\rightarrow N+1}$。

为平滑建模长期运动并泛化到多样场景，我们在预训练 video generator 上构建 $f_\theta$，其中 $\boldsymbol{\theta}$ 为可学习参数。框架见图 2。我们先介绍统一 4D 表示（第 3.1 节），再给出把几何与运动联合编码到统一 latent 空间的专用 4D VAE（此处 4D VAE 指融合后的几何与运动 VAE）（第 3.2 节），最后描述整体训练与推理策略（第 3.3 节）。

### 3.1 统一几何与运动表示

与 DUSt3R 类似，我们把 point map 与 scene flow 定义在第一帧坐标系中，并以该坐标系作为世界坐标系。具体地，point map $\bm{X}_i \in \mathbb{R}^{H\times W\times 3}$ 存储帧 $i$ 中每个像素在世界坐标系下的 3D 坐标 $(x, y, z)$；scene flow $\bm{V}_{i} \in \mathbb{R}^{H\times W\times 3}$ 表示每个像素从帧 $i$ 到 $i+1$ 的 3D 运动向量 $(\Delta x, \Delta y, \Delta z)$。理想情况下，*变形后的 point map*

$$
\bm{X}_i^{d} = \bm{X}_i + \bm{V}_i
\tag{2}
$$

应与下一帧的 point map $\bm{X}_{i+1}$ 在空间上对齐。但由于视点变化，$\bm{X}_i^{d}$ 与 $\bm{X}_{i+1}$ 在像素空间并非一一对应：它们表示不同帧的内容，如图 3 所示。注意，我们的 scene flow 也直接定义在 *（世界）坐标系* 中，即每个 $\bm{V}_i$ 表示世界空间中的运动向量 $(\Delta x, \Delta y, \Delta z)$，从而自然消除相机引起的运动分量。

这种统一的几何–运动表示有若干优点：

1. *无相机建模（Camera-free modeling）。* 与 DUSt3R 类似，在选定的世界坐标系中定义几何与运动，无需额外估计相机位姿。
2. *时间一致性（Temporal consistency）。* 在连续视频中，几何与运动本就时间连贯；在同一坐标系中联合建模更易学习。
3. *更丰富的运动建模。* 与现有方法（Dynamic Point Maps、St4RTrack）不同，我们定义视频中每一对相邻帧之间的 scene flow，而非仅第一帧与其余帧之间。因此该表示对视点变化引起的遮挡更不敏感，仍能捕捉后续帧中新出现动态物体的运动信息。

![图 3. 几何与运动表示。对帧 $\bm{I}_i$ 中的像素 $p_i$，$\bm{X}_i$ 是其对应 3D 点。该 3D 点移动时，用 $\bm{X}_i^d$ 表示移动后的点，用 $\bm{V}_{i} = (\Delta x, \Delta y, \Delta z)$ 表示运动。理想情况下 $\bm{X}_i^d$ 应与下一帧 $\bm{I}_{i+1}$ 中的匹配点 $\bm{X}_{i+1}$ 对齐。但它们的像素索引完全不同（$p_i$ vs. $p_{i+1}$），且由于相机/物体运动，$p_{i+1}$ 甚至可能出画，从而无法在 $\bm{X}_i^d$ 与 $\bm{X}_{i+1}$ 之间建立一一对应。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/data_representation_figure.png)

### 3.2 统一 4D 几何–运动 VAE

下面说明如何把上述 4D 表示有效编码到 latent 空间，作为 video generator 的学习目标。近期几何扩散模型（Marigold、World-from-Pixels、Geo4D）只编码 3D 几何属性，忽略动态场景的显式运动建模。我们则设计新的 4D VAE 架构，把几何与运动联合编码到统一的 4D latent 中，如图 2 所示。

为利用 *预训练* video generator 的 prior，一种广泛看法是：*VAE 的输入应与预训练扩散模型的原始数据分布严格对齐*（Marigold、World-from-Pixels、Geo4D）。朴素做法是用 max normalization 把 disparity、point map 等 3D 属性直接缩放到 $[-1, 1]$，再用冻结的 VAE 权重编码。然而世界坐标下的 3D 属性通常无界，坐标跨越 $(-\infty, +\infty)$，与像素范围有界 $[0, 255]$ 的图像形成对照。此外，3D 属性的分布与自然 RGB 图像本质不同。因此本文探究一个根本问题：*微调扩散模型时，与扩散模型输入空间的严格对齐是否必要？*

![图 4. 不同归一化与 VAE 训练策略的结果。对深度变化很大的室外场景（第二行），原始 VAE 无法恢复场景结构；即便微调 decoder，重建质量仍然很差。本文提出的 mean 归一化与 VAE 训练策略显著提升重建质量。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/rescale_figure.png)

**采用修订归一化的 Geometry VAE。** 为回答上述问题，本模型的一个关键 insight 是：对 Geometry VAE 略微调整 point map 归一化策略。与现有几何扩散模型常用的缩放到 $[-1, 1]$ 的 max normalization 不同，我们对每一段世界坐标 point map 序列采用 *canonical normalization*：

$$
\hat{\bm{X}}_i = \frac{\bm{X}_i-\mu}{S},
\tag{3}
$$

其中 $\mu=\frac{1}{|\mathcal{D}|}\sum_{d\in\mathcal{D}} \bm{X}_d$ 是 point map 序列 $\cup_{i=1}^N\{\bm{X}_i\}$ 中全部有效点（记为 $\mathcal{D}$）的均值；$S=\frac{1}{|\mathcal{D}|}\sum_{d\in\mathcal{D}} \big\| \bm{X}_d - \mu \big\|_2 + \varepsilon$ 是用于尺度归一化的平均距离，$\varepsilon$ 为数值稳定用的小常数。该归一化保持 point map 的尺度不变性，同时相对 max normalization 显著提升 Geometry VAE 的重建质量，并更好保留细结构（DUSt3R、VGGT），尤其在处理大规模室外场景时，见图 4。

我们用新的归一化策略微调整个 encoder–decoder，从而允许更灵活的输入分布。训练目标定义为：

$$
\mathcal{L}_G = \mathcal{L}_{\text{point}} + \lambda_{d}\mathcal{L}_{\text{depth}} + \lambda_{n}\mathcal{L}_{\text{normal}},
\tag{4}
$$

其中 $\mathcal{L}_{\text{point}}$ 是 point map 重建的 MSE loss，$\mathcal{L}_{\text{depth}}$ 是在投影深度图上计算的多尺度损失，$\mathcal{L}_{\text{normal}}$ 约束表面法向一致性（GeometryCrafter、MoGe）。不同之处在于我们编码的是世界坐标系下的点云。因此我们把 ground-truth 相机位姿与点云一起归一化，从而能用尺度对齐后的相机参数把点云投影为深度图。实验表明，这种监督类似于 Geo4D 中的多模态融合，能提升点云重建质量。

我们也尝试用 Kullback–Leibler（KL）散度把 latent 分布约束为标准高斯，但发现这会导致 VAE 性能显著下降。

所提出的归一化与 VAE 训练策略在表 3 中持续提升 VAE 以及下游 diffusion U-Net 的性能，表明 *与扩散模型输入及 latent 空间的严格对齐并非总是必要*，对 3D 属性尤其如此。

**Motion VAE。** 建模 scene flow 的一种简单方式是：用与 Geometry VAE 相同的架构单独训练一个 Motion VAE。但运动与几何本就相关，独立学习运动可能次优。因此我们探索几何与运动之间的若干融合策略：（1）*no fusion*：几何与运动分别编码、互不交互；（2）*offset fusion*：受 LayerDiffuse 启发，把 motion latent 作为 offset 加到 geometry latent 上；（3）*unified fusion*：把 geometry 与 motion latent 拼成统一 4D latent，再送入 Motion VAE decoder 重建 scene flow。如表 4 所示，尽管 unified concatenation 在 VAE 阶段并非重建质量最优，它在后续 diffusion U-Net 上表现更好。训练 Motion VAE 时我们冻结 Geometry VAE 参数，以保留其学到的几何 prior。训练目标为：

$$
\mathcal{L}_{\text{M}} =
\underbrace{
\frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \, \| \hat{\bm{V}}_d - \bm{V}_d \|_2^2
}_{\text{Scene flow reconstruction loss}}
+ \lambda_{\text{reg}} \underbrace{
\frac{1}{|\mathcal{N}|} \sum_{n \in \mathcal{N}} \, \| \hat{\bm{V}}_n \|_2^2
}_{\text{Zero-flow regularization}},
\tag{5}
$$

其中 $\hat{\bm{V}}_d$ 为预测 scene flow，$\bm{V}_d$ 为 ground truth，$\mathcal{D}$ 表示有效像素，$\mathcal{N}$ 表示全部像素。第一项是有效 scene flow 上的 MSE loss；第二项是正则，鼓励 scene flow 趋向零，对应 as-static-as-possible 假设。

把 Geometry VAE 与 Motion VAE 组合成统一 4D VAE 后，我们在单一 latent 空间中得到几何与运动的整合表示，从而高效编码与解码 4D 场景。

### 3.3 模型训练

**训练数据。** 带标注 3D 几何与稠密 scene flow 的动态数据在真实世界中难以采集。因此 scene flow 估计任务依赖合成数据训练。我们把训练数据分成两类：

1. *Geometry Datasets：* Dynamic Replica、GTA-SFM、MatrixCity、MVS-Synth、Point Odyssey、TartanAir、ScanNet++、BlinkVision、OmniWorld 与 Synthia；
2. *Geometry-and-Motion Datasets：* Kubric、Spring 与 Virtual KITTI 2。

第一类只提供几何数据，包括逐帧深度图、相机内参与外参。遵循 DUSt3R，我们把 ground-truth 点云表示到共享的第一帧坐标系。第二类额外提供稠密 scene flow 标注。*几何重建* 训练使用 (1)+(2) 两组；*运动重建* 训练只用第 (2) 组。

**训练策略。** VAE 组件采用两阶段训练。先独立训练 Geometry VAE 以捕捉场景几何；再冻结 Geometry VAE，训练 Motion VAE，从而保留其几何 prior。收敛后把二者组合成统一 4D VAE，其参数在训练 diffusion U-Net 时保持冻结。U-Net 训练时，用第 (1)+(2) 组数据提供几何监督，只用第 (2) 组提供运动监督。沿用 DepthCrafter、GeometryCrafter 等采用 EDM pre-conditioning 的做法，本框架同时支持 *deterministic* 与 *denoising* 两种范式。deterministic 范式的训练目标为：

$$
\mathcal{L}_{\text{deterministic}} = \mathcal{L}_{\text{latent}} + \lambda_G \mathcal{L}_G + \lambda_M \mathcal{L}_{\text{M}},
\tag{6}
$$

其中 $\mathcal{L}_G$ 为式 (4) 的几何重建损失，$\mathcal{L}_{M}$ 为式 (5) 的运动重建损失，$\mathcal{L}_{\text{latent}}$ 为 latent 空间扩散损失：

$$
\mathcal{L}_{\text{latent}} =
\underbrace{
\frac{1}{N} \sum_{N}
\|
\hat{\mathbf{z}}^{\text{G}}_i - \mathbf{z}^{\text{G}}_i
\|_2^2
}_{\text{geometry latent supervision}}
+
\underbrace{
\frac{1}{N-1} \sum_{N-1}
\|
\hat{\mathbf{z}}^{\text{M}}_i - \mathbf{z}^{\text{M}}_i
\|_2^2
}_{\text{motion latent supervision}},
\tag{7}
$$

其中 $N$ 为帧数，$\hat{\mathbf{z}}^{\text{G}}_i,\mathbf{z}^{\text{G}}_i,\hat{\mathbf{z}}^{\text{M}}_i, \mathbf{z}^{\text{M}}_i$ 分别为几何与运动的去噪 latent 及原始 latent。我们只做前向 scene flow 估计，因此丢弃最后一帧的 motion latent。对 denoising 范式，目标简化为 latent 监督：$\mathcal{L}_{\text{denoise}} = \mathcal{L}_{\text{latent}}$。实验发现 deterministic 范式通常更好，故默认使用；消融见补充材料。

这一渐进、模块化的训练管线让模型先获得强几何与运动 prior，再整合时间推理，最终实现稳健、连贯的稠密 4D 重建。

**实现细节。** 为继承 video generator 的强 prior，MotionCrafter 的 VAE 与 U-Net 均用 SVD 预训练权重初始化，并以 AdamW 优化，学习率 $1\times 10^{-4}$。先训练 Geometry VAE 40{,}000 次迭代，再训练 Motion VAE 20{,}000 次迭代。随后把 Geometry VAE 与 Motion VAE 合并为统一 4D VAE，再用编码得到的 4D latent 训练 U-Net 另外 40{,}000 次迭代。VAE 训练 batch size 为 8，U-Net 为 25。全部实验在 8 张 40 GB 显存的 GPU 上进行，约 3 天。更多实现细节见补充材料。

---

## 4. 实验

### 4.1 评估设定

**数据集。** *几何* 评估在三个未见过的动态场景数据集上做 zero-shot 测试：DDAD、Monkaa 与 Sintel。它们覆盖真实与合成场景，含室内与室外。*运动* 评估因带稠密 scene flow 标注的数据有限，使用三个 in-domain 数据集（Kubric、Spring、VKITTI2）与两个 out-of-domain 数据集（Dynamic Replica、Point Odyssey）。由于 Dynamic Replica 与 Point Odyssey 只提供稀疏 scene flow 标注，我们仅在已标注点上计算指标。

**指标。** 与先前方法不同，我们在 *世界坐标系* 中评估几何与运动。预测的世界空间点云通过优化每段序列的尺度与平移参数与 ground truth 对齐。报告 *relative point error*（$\text{Rel}^p$）与 *percentage of inlier*（$\delta^p$，阈值 0.25）。预测的 scene flow 按 point map 尺度对齐。计算 *End Point Error*（EPE）与 *Average Percent of Points within Delta*（APD），APD 下标为度量尺度下的 inlier 阈值。细节见补充材料。

### 4.2 与 State-of-the-art 方法比较

**联合几何与运动重建评估。** 表 1 将 MotionCrafter 与近期有代表性的联合几何–运动估计方法比较。指标均在世界坐标空间中评估。本模型直接输出世界坐标下的预测序列。相比之下，现有方法大多沿用基于 DUSt3R 的成对设计，需要 post-optimization 或相机位姿才能与 ground truth 对齐。为公平起见，我们用 VGGT 预测的相机位姿把它们的预测变换到世界坐标系。

由表 1 与图 5 的定量、定性比较可见：这些成对方法扩展到视频序列时性能通常下降；本方法平均在几何上优于 state-of-the-art 38.64%，在运动上优于 25.0%。注意，与 Zero-MSF 不同，我们并未用 Dynamic Replica 与 Point Odyssey 的运动标注训练，却仍取得更好表现（仅一项指标相当）。图 6 表明本方法在世界坐标系中估计时间一致的 scene flow，对相机运动稳健，能更准确、高效地描述 4D 场景动态。

**表 1. 世界坐标系下联合几何与运动重建评估。** 为便于阅读，所有指标均不带百分号。$*$ 表示非 zero-shot 的 scene flow 评估。-S 与 -P 分别表示 ST4RTrack 的 Sequence 模式与 Pair 模式。由于 ST4RTrack 的运动总是相对第一帧比较，为公平起见，我们在每一对相邻帧上运行它，再用 VGGT 位姿把结果变换到世界坐标系。另外给出 Zero-MSF + GT pose 作为参考（表中灰色行）。

*几何（Geometry）。*

| Method | Kubric $\text{Rel}^{p}\downarrow$ | Kubric $\delta^{p}\uparrow$ | Spring $\text{Rel}^{p}\downarrow$ | Spring $\delta^{p}\uparrow$ | VKITTI2 $\text{Rel}^{p}\downarrow$ | VKITTI2 $\delta^{p}\uparrow$ | Dynamic Replica $\text{Rel}^{p}\downarrow$ | Dynamic Replica $\delta^{p}\uparrow$ | Point Odyssey $\text{Rel}^{p}\downarrow$ | Point Odyssey $\delta^{p}\uparrow$ | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| POMATO + VGGT | 25.56 | 77.85 | 98.13 | 61.71 | 32.16 | 54.89 | 7.26 | 94.64 | 19.88 | 80.08 | 5.0 |
| ST4RTrack-S + VGGT | 6.61 | 95.59 | 123.27 | 42.26 | 67.68 | 21.87 | 5.65 | 96.29 | 31.00 | 68.17 | 4.0 |
| ST4RTrack-P + VGGT | 17.81 | 80.76 | 157.05 | 38.00 | 84.77 | 14.46 | 4.87 | 97.13 | 29.13 | 71.66 | 3.4 |
| DELTA + VGGT | 14.09 | 85.73 | 106.88 | 50.81 | 57.03 | 42.76 | 23.21 | 74.35 | 51.47 | 50.92 | 6.0 |
| Zero-MSF + VGGT | 8.79 | 94.73 | 142.44 | 40.66 | 15.76 | 80.04 | 7.11 | 97.03 | 22.55 | 78.27 | 4.6 |
| Zero-MSF + GT（参考） | 8.78 | 94.73 | 142.45 | 40.69 | 11.22 | 89.92 | 7.11 | 97.01 | 22.52 | 78.27 | — |
| MotionCrafter (ours) | **3.40** | **98.73** | **29.20** | **77.27** | **14.60** | **84.58** | **4.04** | **99.00** | **9.94** | **94.90** | **1.0** |

*运动（Motion）。*

| Method | Kubric EPE | Kubric APD$_{0.05}\uparrow$ | Spring EPE | Spring APD$_{0.1}\uparrow$ | VKITTI2 EPE | VKITTI2 APD$_{0.3}\uparrow$ | Dynamic Replica EPE | Dynamic Replica APD$_{0.05}\uparrow$ | Point Odyssey EPE | Point Odyssey APD$_{0.05}\uparrow$ | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| POMATO + VGGT | 79.58 | 5.23 | 180.13 | 20.16 | 368.78 | 15.66 | 14.56* | 36.34* | 31.94* | 30.32* | 5.0 |
| ST4RTrack-S + VGGT | 59.34* | 1.68* | 105.02 | 33.47 | 156.85 | 22.11 | 1.12* | 97.13* | 10.37* | 53.39* | 4.0 |
| ST4RTrack-P + VGGT | 217.20* | 5.65* | 441.84 | 9.07 | 874.94 | 13.16 | 0.99* | 97.55* | 58.62* | 51.19* | 4.8 |
| DELTA + VGGT | 8.29* | 52.95* | 8.59 | 81.55 | 156.64 | 17.7 | 0.75 | 99.57 | 6.09 | 62.11 | 3.2 |
| Zero-MSF + VGGT | 8.59* | 50.13* | 7.78* | 85.59* | 112.99* | 21.69* | 0.59* | **99.80*** | 3.58* | 80.12* | 2.4 |
| Zero-MSF + GT（参考） | 5.74* | 72.37* | 5.50* | 87.59* | 73.81* | 25.15* | 0.40* | 99.80* | 2.30* | 91.04* | — |
| MotionCrafter (ours) | **4.60*** | **68.01*** | **5.61*** | **90.17*** | **71.75*** | **25.90*** | **0.51** | 99.72 | **3.49** | **80.66** | **1.0** |

![图 5. 与 Zero-MSF 的定性比较。建议放大查看细节。相对 Zero-MSF，我们有更合理的场景结构与更好的几何细节；更重要的是，预测的 3D scene flow 运动方向更准确。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/indomain_result_figure.png)

**几何重建评估。** 表 2 进一步与若干有代表性的几何重建方法比较。按几何表示分组：（1）Camera-centric 方法在相机坐标系中预测深度或 point map。为公平比较，我们用 VGGT 位姿把其输出变换到世界坐标。（2）World-centric 方法直接在参考帧坐标系中预测几何，我们用仿射变换把结果对齐到 ground truth。

本方法在 Monkaa 上达到 *state-of-the-art*，体现所提架构的优势。在 Sintel 与 DDAD 上弱于 VGGT，我们归因于单模态设计（无 camera ray 与深度图）以及室外训练数据规模有限。注意我们不像 Geo4D 那样做 post-optimization。视觉比较（见补充材料）进一步验证：本方法在动态环境中给出更连贯、一致的重建。

**表 2. 世界坐标系下的几何重建评估。** $^\dagger$ 表示使用 post-optimization。本方法结果均未做任何 post-optimization。

| Method | Monkaa $\text{Rel}^{p}\downarrow$ | Monkaa $\delta^{p}\uparrow$ | Sintel $\text{Rel}^{p}\downarrow$ | Sintel $\delta^{p}\uparrow$ | DDAD $\text{Rel}^{p}\downarrow$ | DDAD $\delta^{p}\uparrow$ | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *Camera-centric* |  |  |  |  |  |  |  |
| DepthPro | 36.96 | 63.65 | 43.30 | 42.30 | 35.38 | 45.49 | 7.00 |
| MoGe | 35.21 | 65.95 | 35.28 | 60.97 | 18.63 | 77.07 | 4.33 |
| GeoCrafter | 33.44 | 65.79 | 30.61 | 68.07 | 19.17 | 76.39 | 3.33 |
| *World-centric* |  |  |  |  |  |  |  |
| MonST3R$^\dagger$ | 41.41 | 31.46 | 37.65 | 51.41 | 31.56 | 55.30 | 6.33 |
| VGGT | 34.54 | 56.65 | **26.83** | **67.91** | 15.98 | 84.06 | **2.33** |
| Geo4D$^\dagger$ | 28.04 | 69.52 | 34.61 | 59.54 | **14.58** | **83.68** | **2.33** |
| St4RTrack | 47.04 | 45.46 | 40.59 | 51.54 | 39.59 | 32.47 | 7.67 |
| **Ours** | **25.88** | **74.01** | 32.46 | 63.14 | 21.27 | 72.82 | 2.67 |

![图 6. 与 ST4RTrack 的定性比较。第一例中，像素轨迹显示我们得到更干净的 scene flow，而 ST4RTrack 出现噪声漂移。第二例中，变形后的 point map（颜色更深）显示本方法预测的几何与运动在时间上更一致。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/effectiveness_figure.png)

### 4.3 消融实验

我们对 MotionCrafter 做充分消融，结果见表 3、表 4，围绕以下三个关键问题。

**是否必须把输入分布与 Video Diffusion 严格对齐？** 答案是 **否**。现有几何扩散模型大多把深度、point map 等 3D 属性严格归一化到 $[-1, 1]$，以便继承 diffusion prior。但我们发现这种 *max-rescale* 归一化使预训练 VAE 的重建精度次优（表 3 的 VAE-1 与 VAE-2）。Geo4D 通过冻结 VAE encoder、只微调 decoder 缓解该问题，但消融表明该策略仍然较差（表 3 的 VAE-3）。相比之下，我们提出的 *mean rescale* 策略——再配合微调全部 VAE 组件——在不严格遵循原始 VAE 分布的情况下取得最佳性能（表 3 的 VAE-4）。我们在 Diffusion U-Net 阶段进一步验证该发现（Unet-I vs. Unet-II），几何上平均提升 16.6%。这说明即便 *不* 与 video diffusion 的分布对齐，MotionCrafter 仍保持更好的泛化能力。

**表 3. Geometry VAE 消融。** 同时报告 VAE 与 U-Net 上的几何重建结果，因为更好的 VAE 未必带来更好的最终结果。模型仅在几何重建子集上训练。

| Model | Training Type | Rescale | Sintel $\text{Rel}^{p}\downarrow$ | Sintel $\delta^{p}\uparrow$ | Monkaa $\text{Rel}^{p}\downarrow$ | Monkaa $\delta^{p}\uparrow$ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| VAE-1 | Original | Max | 39.96 | 62.63 | 23.78 | 67.33 |
| VAE-2 | From scratch | Max | 18.80 | 79.68 | 11.48 | 90.55 |
| VAE-3 | Finetune decoder | Max | 20.76 | 77.77 | 14.44 | 85.91 |
| VAE-4 | Finetune all | Mean | **5.68** | **98.77** | **5.03** | **99.13** |
| Unet-I | VAE-3 + Unet | Max | 40.47 | 49.42 | 33.66 | 56.42 |
| Unet-II | VAE-4 + Unet | Mean | **35.17** | **58.08** | **27.36** | **66.21** |

**融合几何与运动 latent 的哪种策略最有效？** 我们按第 3.2 节讨论的 *Offset*、*Separate* 与 *Unify* 策略，在表 4 中联合编码几何与运动。实验表明：尽管分离的 VAE 在 VAE 重建上最优，统一 VAE 最终在 U-Net 预测上更好。该现象说明：要把 4D 建模做得连贯，几何与运动表示需要紧密耦合。

**表 4. Motion VAE 消融。** 在三个动态 scene flow 数据集上比较不同设计；同样同时报告 VAE 与 U-Net 结果。

| Model | Fusion Type | Spring EPE | Spring APD$_{0.03}\uparrow$ | Point Odyssey EPE | Point Odyssey APD$_{0.03}\uparrow$ |
| --- | --- | ---: | ---: | ---: | ---: |
| VAE-5 | Original | 6.43 | 50.49 | 1.55 | 91.68 |
| VAE-6 | Offset | 1.83 | 83.51 | 2.00 | 86.66 |
| VAE-7 | Separate | **0.66** | **96.75** | **0.77** | **94.74** |
| VAE-8 | Unify | 0.88 | 94.78 | **0.77** | 94.19 |
| Unet-III | Separate | 6.37 | 65.94 | 3.65 | 63.19 |
| Unet-IV | Unify | **5.16** | **72.81** | **3.49** | **66.36** |

**Video generator 提供了什么 prior？** 用原始预训练 VAE 编码几何与运动时，模型在室内场景已表现出合理的重建能力，见表 3、表 4 的首行。但由于 point map、scene flow 与图像空间分布之间存在显著尺度差异，原始 VAE 难以处理大幅尺度变化，尤其在室外场景，见图 4。我们认为：要充分利用扩散模型中的 prior，合适的归一化与微调策略至关重要。如表 3 的 VAE-2 所示，从零训练导致次优结果，证实预训练 Video Diffusion 模型确实包含有益于稠密 4D 重建的丰富 prior。通过显式建模这些 prior，我们有效释放了 video diffusion model 的 4D 表示能力。

---

## 5. 结论

我们提出 MotionCrafter：能从单目视频联合重建稠密几何与运动的框架。通过把二者定义在统一世界坐标系中，并设计把它们编码到共享 latent 空间的新型 4D VAE，我们即使不做任何后处理也达到 state-of-the-art。值得注意的是，我们表明不必把 4D latent 分布与原始 SVD latent 分布严格对齐。事实上，更宽松的对齐与 VAE 再训练策略不仅保留、还提升了扩散模型的泛化，为把 diffusion prior 适配到新模态提供更广泛的启示。

**局限性。** 目前我们只关注稠密几何与运动重建；先前工作表明，引入多种几何模态能显著改善 3D 属性预测，包括相机参数、point map、深度图、point track 与新视角。因此探索多模态整合是有前景的未来方向。

**致谢：** Xiaoguang Han 受广东省杰出青年基金（No. 2023B1515020055）资助。Chuanxia Zheng 受 NTU SUG-NAP 以及新加坡国家研究基金会 NRF Fellowship（NRF-NRFF17-2025-0009）资助。

---

## 附录

**补充视频**中提供更多可视化结果。**本文档**给出正文的进一步细节与分析。

### A. 数据处理

对每个视频序列，我们把对应的 point map 与 scene flow 预处理到由第一帧相机位姿参照的统一世界坐标系。处理管线分三步：（1）相机位姿归一化；（2）把 point map 与 scene flow 变换到世界坐标；（3）对世界空间几何与运动做全局归一化。下面分述各步。

#### A.1 相机位姿归一化

单目重建系统给出的相机位姿常含任意的全局旋转与平移。为消除该歧义，遵循 DUSt3R，我们把所有位姿对齐到由第一帧相机定义的规范坐标系。具体地，给定相机位姿序列 $\{\bm{P}_i\}_{i=1}^N$，其中每个 $\bm{P}_i \in \mathbb{R}^{4 \times 4}$，把第一帧位姿分解为

$$
\bm{R}_0 = \bm{P}_0[:3,:3], \qquad \bm{t}_0 = \bm{P}_0[:3,3].
\tag{8}
$$

每个位姿再归一化为

$$
\tilde{\bm{R}}_i = \bm{R}_0^\top \bm{R}_i, \qquad
\tilde{\bm{t}}_i = \bm{R}_0^\top (\bm{t}_i - \bm{t}_0),
\tag{9}
$$

从而在去掉全局旋转与平移的同时保留序列内的相对运动。

#### A.2 Point Map 变换

给定表示在第 $i$ 帧相机坐标系中的 point map $\bm{X}_i^C \in \mathbb{R}^3$，用归一化后的相机位姿把它变到第一帧坐标系：

$$
\bm{X}_i = \tilde{\bm{R}}_i \bm{X}_i^C + \tilde{\bm{t}}_i .
\tag{10}
$$

该变换作用于全部有效像素；无效像素（由 validity mask 指示）置零。对这些无效点，我们用 pyramid padding（LayerDiffuse）填充。注意我们并不监督这些无效点；填充只是为了避免 VAE 在特征提取时受缺失值影响。

#### A.3 Scene Flow 变换

原始 scene flow $\bm{V}_i^C$ 定义在第 $i$ 帧相机坐标中。为得到第一帧（世界）空间中的 scene flow，先计算变形后的点：

$$
\bm{X}_{i \rightarrow i+1}^{C} = \bm{X}_i^C + \bm{V}_i^C,
\tag{11}
$$

再用下一帧的相机位姿变换它们：

$$
\bm{X}_{i \rightarrow i+1} =
\tilde{\bm{R}}_{i+1} \bm{X}_{i \rightarrow i+1}^{C} + \tilde{\bm{t}}_{i+1}.
\tag{12}
$$

世界空间 scene flow 计算为

$$
\bm{V}_i = \bm{X}_{i \rightarrow i+1} - \bm{X}_i .
\tag{13}
$$

若有 deformability mask，则用它把非动态区域的 scene flow 置零。

#### A.4 世界坐标全局归一化

为在不同尺度的场景上保持训练一致，全局归一化作用于世界空间几何。

**Centering。** 先计算全部有效点的质心：

$$
\bm{\mu} = \frac{1}{| \mathcal{D}|} \sum_{d \in \mathcal{D}} \bm{X}_d.
\tag{14}
$$

**Isotropic Rescaling。** 不以最大半径缩放，而以有效点到质心的平均尺度：

$$
S = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \lVert \bm{X}_d -\bm{\mu} \rVert_2.
\tag{15}
$$

然后用仿射变换统一归一化 point map、相机位姿与 scene flow：

$$
\bm{X}_i \leftarrow \frac{\bm{X}_i - \bm{\mu}}{S}, \qquad
\tilde{\bm{t}}_i \leftarrow \frac{\tilde{\bm{t}}_i - \bm{\mu}}{S}, \qquad
\bm{V}_i \leftarrow \frac{\bm{V}_i}{S}.
\tag{16}
$$

这种各向同性缩放在跨数据集归一化绝对尺度的同时保留几何结构。归一化参数 $(\bm{\mu}, S)$ 被保存，以便可选地恢复原始度量尺度。

### B. 额外消融

#### B.1 多模态监督消融

**动机。** Geo4D 等方法依赖多模态输出（如深度、point map 与法向）以及 post-optimization 融合阶段得到最终重建。我们的主要目标是实现 *完全 feed-forward* 的 4D 几何与运动重建。因此推理时有意不引入任何辅助输出或后精炼。有趣的是，尽管测试时不用多模态输出，我们发现 VAE 训练时的多模态 *监督* 仍能改善 4D latent 的重建质量。具体地，我们从世界坐标 point map 导出深度作为额外监督信号。

**深度监督。** 给定重建的世界坐标 point map $\hat{\mathbf{X}}$ 与 ground-truth point map $\mathbf{X}$，用归一化相机位姿 $\tilde{\mathbf{P}}$ 把二者投影到深度域：

$$
\hat{\bm{D}} = \Pi(\hat{\bm{X}}, \tilde{\bm{P}}), \qquad
\bm{D} = \Pi(\bm{X}, \tilde{\bm{P}}),
\tag{17}
$$

其中 $\Pi(\cdot)$ 表示标准投影到深度图。我们施加两项互补损失：

- **逐像素 L1 depth loss。** 该损失鼓励准确的深度预测，并由深度有效性 mask 掩蔽：

$$
\mathcal{L}_{\text{L1-D}} =
\left\|
\, (\hat{\mathbf{D}} - \mathbf{D}) \odot \mathbf{W}
\right\|_{1},
\tag{18}
$$

其中 $\mathbf{W}$ 为二值 valid-mask。

- **多尺度 patch depth loss。** 为提升不同空间尺度上的几何一致性，我们在尺度因子 $\{4,16,64\}$ 定义的 patch 上计算 L1 loss。对每个尺度，深度图被划分为不重叠 patch；在每个 patch 内，用带 mask 的平均减去均值深度以去除全局偏差：

$$
\mathcal{L}_{\text{Patch-D}}
=
\sum_{s \in \{4,16,64\}}
\left\|
\, \left( \hat{\mathbf{D}}^{(s)} - \mathbf{D}^{(s)} \right)
\odot \mathbf{W}^{(s)}
\right\|_{1}.
\tag{19}
$$

该项鼓励局部几何结构一致，并抑制 depth-shift 伪影。

多模态深度监督总计为：

$$
\mathcal{L}_{\text{depth}}
=
\lambda_{\text{L1-D}} \mathcal{L}_{\text{L1-D}}
+
\lambda_{\text{Patch-D}} \mathcal{L}_{\text{Patch-D}}.
\tag{20}
$$

**结果。** 如表 5 所示，引入基于深度的多模态监督显著提升世界坐标 point map 的重建质量（point map 提升 13.55%，depth map 提升 16.41%）。值得注意的是，该提升无需修改推理管线，测试时也不引入额外模态。该消融表明：多模态监督是增强 VAE 所学 4D latent 表示的有效策略。

**表 5. Geometry VAE 组件消融。** 在 ScanNet、Sintel 与 Monkaa 上报告点精度（$\text{Rel}^{p}\downarrow$，$\delta^{p}\uparrow$）与深度精度（$\text{Rel}^{d}\downarrow$，$\delta^{d}\uparrow$）。

*ScanNet。*

| Model | Training | Rescale | Depth Loss | $\text{Rel}^{p}\downarrow$ | $\delta^{p}\uparrow$ | $\text{Rel}^{d}\downarrow$ | $\delta^{d}\uparrow$ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Original | Max | $\times$ | 14.96 | 86.95 | 1.98 | 99.90 |
| 2 | From scratch | Mean | $\times$ | 7.01 | 99.29 | 1.88 | 99.92 |
| 3 | From scratch | Max | $\times$ | 11.88 | 91.08 | 1.82 | 99.59 |
| 4 | Finetune decoder | Max | $\times$ | 10.45 | 92.30 | 2.73 | 98.83 |
| 5 | Finetune decoder | Max | $\checkmark$ | 4.46 | 98.20 | **0.54** | 99.97 |
| 6 | Finetune all | Mean | $\times$ | 4.46 | 99.69 | 1.11 | 99.97 |
| 7 | Finetune all | Mean | $\checkmark$ | **3.03** | **99.88** | 0.76 | **99.99** |

*Sintel。*

| Model | Training | Rescale | Depth Loss | $\text{Rel}^{p}\downarrow$ | $\delta^{p}\uparrow$ | $\text{Rel}^{d}\downarrow$ | $\delta^{d}\uparrow$ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Original | Max | $\times$ | 39.96 | 62.63 | 12.62 | 92.65 |
| 2 | From scratch | Mean | $\times$ | 10.40 | 93.89 | 42.44 | 92.38 |
| 3 | From scratch | Max | $\times$ | 18.80 | 79.68 | 10.21 | 92.26 |
| 4 | Finetune decoder | Max | $\times$ | 20.76 | 77.77 | 12.67 | 89.08 |
| 5 | Finetune decoder | Max | $\checkmark$ | 12.04 | 88.28 | **4.30** | **98.64** |
| 6 | Finetune all | Mean | $\times$ | 5.68 | 98.77 | 9.73 | 94.60 |
| 7 | Finetune all | Mean | $\checkmark$ | **4.39** | **99.14** | 8.04 | 95.31 |

*Monkaa。*

| Model | Training | Rescale | Depth Loss | $\text{Rel}^{p}\downarrow$ | $\delta^{p}\uparrow$ | $\text{Rel}^{d}\downarrow$ | $\delta^{d}\uparrow$ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Original | Max | $\times$ | 23.78 | 67.33 | 4.30 | 98.91 |
| 2 | From scratch | Mean | $\times$ | 8.02 | 96.91 | 4.10 | 98.47 |
| 3 | From scratch | Max | $\times$ | 11.48 | 90.55 | 5.93 | 93.80 |
| 4 | Finetune decoder | Max | $\times$ | 14.44 | 85.91 | 7.10 | 91.66 |
| 5 | Finetune decoder | Max | $\checkmark$ | 8.50 | 93.36 | **1.37** | 99.64 |
| 6 | Finetune all | Mean | $\times$ | 5.03 | 99.13 | 3.54 | 99.27 |
| 7 | Finetune all | Mean | $\checkmark$ | **3.74** | **99.47** | 1.83 | **99.77** |

#### B.2 Decoder Loss 消融

**动机。** 在 deterministic 设定下，常规扩散模型的去噪过程可视为从多步坍缩为单步。因此，除监督去噪后的 latent 表示外，我们引入直接监督 VAE decoder 输出的 *decoder loss*。相对 latent regression，该监督更直接，给 U-Net 的训练信号更强。

**实现。** 训练时不更新 VAE decoder 的权重。但仍计算其梯度，使损失能经 decoder 反传到 U-Net 参数。为降低显存，该过程对 VAE decoder 使用 gradient checkpointing。

**结果。** 如表 6 所示，引入 decoder loss 持续改善 U-Net 训练。在四个未见数据集上平均提升 15.01%，在室外数据集 DDAD 上尤为显著，提升 36.80%。

**表 6. 面向几何重建的 U-Net 组件消融。** 比较不同训练策略、rescale 方法与 decoder loss。

| Model | Training Strategy | Normalization | Decoder Loss | GMU Kitchen $\text{Rel}^{p}\downarrow$ | GMU Kitchen $\delta^{p}\uparrow$ | Monkaa $\text{Rel}^{p}\downarrow$ | Monkaa $\delta^{p}\uparrow$ | Sintel $\text{Rel}^{p}\downarrow$ | Sintel $\delta^{p}\uparrow$ | DDAD $\text{Rel}^{p}\downarrow$ | DDAD $\delta^{p}\uparrow$ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| I | Finetuning VAE Decoder | Max | $\times$ | 17.49 | 80.65 | 33.66 | 56.42 | 40.47 | 49.42 | 41.66 | 39.45 |
| II | Finetuning the whole VAE | Mean | $\times$ | 17.72 | 82.90 | 27.36 | 66.21 | 35.17 | **58.08** | 33.78 | 50.34 |
| III | Finetuning the whole VAE | Mean | $\checkmark$ | **14.47** | **90.54** | **25.33** | **71.77** | **33.39** | 56.94 | **22.39** | **70.42** |

#### B.3 训练范式消融

**动机。** 沿用 DepthCrafter、GeometryCrafter 等采用 EDM pre-conditioning 的做法，本框架在预训练 SVD 之上同时支持 *deterministic* 与 *denoising* 扩散范式。由于 4D 重建是确定性任务，我们默认使用 deterministic 范式；该范式已在先前稠密预测框架（Lotus、DepthMaster、GeometryCrafter）中被广泛探索并证明有效。我们也想确切知道这两种训练范式在本框架中差异有多大，尤其对同时包含几何与运动信息的 4D latent。因此做消融验证。

**实现。** 第 3.3 节已介绍两种训练范式的损失函数。为公平比较，deterministic 范式不使用 decoder loss。具体地，我们把视频 latent 直接送入 U-Net 以预测 4D latent。对 denoising 范式，先对 4D latent 加噪，再与视频 latent 沿通道拼接，然后用 U-Net 的多步去噪预测 4D latent。全部 U-Net 权重从原始 SVD 初始化，仅在第一层调整通道维以适配不同训练范式。

**结果。** 如表 7 所示，相对 diffusion 范式，deterministic 范式在各数据集平均把 $\text{Rel}^{p}$ 降低约 12.4%，把 $\delta^{p}$ 提升约 12.7%。该结果有力说明 deterministic 范式在稠密预测任务中的有效性，也表明 SVD 的先验知识可以在不依赖去噪机制的情况下被模型继承。

**表 7. 不同训练范式的消融。**

| Training Type | Monkaa $\text{Rel}^{p}\downarrow$ | Monkaa $\delta^{p}\uparrow$ | Sintel $\text{Rel}^{p}\downarrow$ | Sintel $\delta^{p}\uparrow$ | DDAD $\text{Rel}^{p}\downarrow$ | DDAD $\delta^{p}\uparrow$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Diffusion | 30.11 | 65.49 | 35.95 | 53.82 | 24.58 | 67.54 |
| Deterministic | **25.88** | **74.01** | **32.46** | **63.14** | **21.27** | **72.82** |

### C. 实现细节

#### C.1 超参数

训练 4D VAE 所用的损失权重如下：

- Point Map Reconstruction Loss：$\lambda_{\text{point}} = 1.0$
- Per-pixel L1 Depth loss：$\lambda_{\text{L1-D}} = 1.0$
- Multi-Scale Depth Supervision：$\lambda_{\text{Patch-D}} = 1.0$
- Normal Consistency Loss：$\lambda_{\text{normal}} = 0.2$
- Scene Flow Reconstruction Loss：$\lambda_{\text{sceneflow}} = 1.0$
- Scene Flow Regulation Loss：$\lambda_{\text{reg}} = 0.01$

预训练 video diffusion U-Net 用以下 latent regression loss 以及（可选的）decoder loss 优化：

- Latent Regression Loss：$\lambda_{\text{latent}} = 1.0$
- Point Map Decoder Loss：$\lambda_{G} = 1.0$
- Scene Flow Decoder Loss：$\lambda_{M} = 1.0$

#### C.2 所用训练集

表 8 列出所用训练数据集，图 7 给出若干可视化样本。

**表 8. 训练数据集概览。** 为平衡训练，部分数据集只采样子集。

| Dataset | Domain | #Frames | #Videos |
| --- | --- | ---: | ---: |
| DynamicReplica | Indoor/Outdoor | 145K | 1126 |
| GTA-SfM | Outdoor | 19K | 234 |
| Kubric | Indoor | 137K | 5736 |
| MatrixCity | Outdoor-Driving | 452K | 3029 |
| MVS-Synth | Outdoor-Driving | 12K | 120 |
| Spring | Outdoor | 5K | 49 |
| Point Odyssey | Indoor | 18K | 120 |
| Synthia | Outdoor-Driving | 178K | 1276 |
| TartanAir | Outdoor | 306K | 2245 |
| VirtualKitti2 | Driving | 43K | 320 |
| BlinkVision | Indoor/Outdoor | 11K | 72 |
| OmniWorld | Indoor/Outdoor | 35K | 350 |
| Scannet++ | Indoor-Real | 310K | 2078 |
| Total | — | 1.67M | 16.8K |

#### C.3 模型信息

本系统采用 Stable Video Diffusion（SVD）的 VAE 与 video U-Net 骨干。管线由三个主要组件构成：（1）视频 VAE encoder，编码逐帧 latent；（2）4D VAE decoder，从 latent 空间重建几何与运动场；（3）3D 时空 U-Net，做 latent 去噪。各组件参数量如下：

- **Video U-Net：** 1524.62M 参数。该大型时空 U-Net 负责在空间与时间上对 latent 去噪，从而建模动态几何与运动。
- **Video VAE Encoder：** 34.16M 参数。该模块独立处理每一输入帧，编码到空间下采样因子为 $8\times$ 的 latent 空间。
- **4D VAE Decoder：** 99.00M 参数。该 decoder 从 latent 表示重建 4D point map 与 scene flow。
- **总参数量：** 1657.79M。

**推理耗时。** 全部计时在单张 40 GB 显存 GPU 上测量。对分辨率 $320\times 640$、长度为 25 帧的视频片段，平均每帧处理时间为：VAE 编码 52.0 ms，latent 去噪 13.4 ms，VAE 解码 73.5 ms，合计每帧 138.9 ms。这些测量反映几何–运动重建管线一次完整前向传播的端到端耗时。

#### C.4 评估指标

下面给出几何与运动重建所用评估指标的详细定义。

**Geometry Alignment。** 由于单目重建存在尺度歧义，预测的世界空间 point map $\hat{\mathbf{X}}_i$ 用每段序列的尺度 $s$ 与平移 $\mathbf{t}$ 对齐到 ground truth $\mathbf{X}_i$：

$$
\tilde{\mathbf{X}}_i = s \hat{\mathbf{X}}_i + \mathbf{t},
\tag{21}
$$

其中 $s$ 与 $\mathbf{t}$ 通过最小化下式优化：

$$
\min_{s,\mathbf{t}} \sum_i
\left\| s\hat{\mathbf{X}}_i + \mathbf{t} - \mathbf{X}_i \right\|_2^2.
\tag{22}
$$

**Relative Point Error（$\mathrm{Rel}^p$）。** 相对几何误差定义为：

$$
\mathrm{Rel}^p =
\frac{1}{N}
\sum_i
\frac{
\left\|\tilde{\mathbf{X}}_i - \mathbf{X}_i\right\|_2
}{
\left\|\mathbf{X}_i\right\|_2
}.
\tag{23}
$$

**Inlier Ratio（$\delta^p$）。** 计算相对误差低于阈值 $\tau$（实验中为 0.25）的点的百分比：

$$
\delta^p =
\frac{1}{N}
\sum_i
\mathbf{1}
\left(
\frac{
\left\|\tilde{\mathbf{X}}_i - \mathbf{X}_i\right\|_2
}{
\left\|\mathbf{X}_i\right\|_2
}
< \tau
\right).
\tag{24}
$$

**Scene Flow Alignment。** 预测的 scene flow $\hat{\mathbf{V}}_i$ 使用与几何相同的尺度 $s$ 缩放：

$$
\tilde{\mathbf{V}}_i = s \hat{\mathbf{V}}_i.
\tag{25}
$$

**End-Point Error（EPE）。** 计算预测与 ground-truth scene flow 之间的平均端点误差：

$$
\mathrm{EPE} =
\frac{1}{N}
\sum_i
\left\|
\tilde{\mathbf{V}}_i - \mathbf{V}_i
\right\|_2.
\tag{26}
$$

**Average Percent of Points within Delta（APD）。** APD 衡量误差低于阈值 $\gamma$ 的 scene flow 向量百分比：

$$
\mathrm{APD}_\gamma =
\frac{1}{N}
\sum_i
\mathbf{1}
\left(
\left\|
\tilde{\mathbf{V}}_i - \mathbf{V}_i
\right\|_2 < \gamma
\right).
\tag{27}
$$

### D. 更多可视化结果

我们从 Davis 数据集选取若干 in-the-wild 视频做 zero-shot 测试，结果见图 8。更直观的可视化见所附视频演示。我们也给出与其他方法的更多定性比较，见图 9、图 10、图 11。比较对应两项不同任务：（1）联合几何与运动估计；（2）仅几何重建。

![图 7. 训练集示例。我们从这些数据集中随机采样视频帧。几何训练时设置随机 stride，以不同间隔采样视频；运动训练时始终保持 stride 为 1，连续采样帧。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/trainingset_figure.png)

![图 8. Davis 数据集上的 zero-shot 结果。尽管用于训练 scene flow 估计的样本数量非常有限，本方法仍能跨不同场景类型良好泛化。得益于端到端模型设计以及世界坐标系中几何与运动的统一定义，全部结果由模型直接输出，无需任何 post-optimization。更直观的动态理解见视频可视化。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/zeroshot_figure.png)

![图 9. 与 state-of-the-art 方法 Zero-MSF 与 DELTA 的定性比较。第一例中，即便不像 Zero-MSF 那样在 Dynamic Replica 上训练，本方法的 scene flow 估计精度仍与之相当。其余例子中，本方法在几何结构与运动模式估计上均显著优于现有方法。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/results_compare_figure.png)

![图 10. 与 VGGT、Geo4D、ST4RTrack 的定性几何比较。对运动物体（如第一例中的手指），本方法估计的尺度与运动变化更准确。对室外场景，本方法估计的场景结构更准确。值得注意的是，本方法与 VGGT 一样能直接输出世界坐标点云，无需 Geo4D 那样的 post-optimization。此外，本方法的训练规模远小于 VGGT，却在动态场景中表现稳健。我们将其归因于 video diffusion 的预训练知识以及所提出的训练策略。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/geo_compare_figure.png)

![图 11. 与 ST4RTrack 在 zero-shot 泛化上的定性几何比较。相对 ST4RTrack，我们的结果多视图一致性更好、几何更平滑、游离斑点更少。](../../../arxiv/geometry_control/motioncrafter/extracted/figs/multiview_figure.png)
