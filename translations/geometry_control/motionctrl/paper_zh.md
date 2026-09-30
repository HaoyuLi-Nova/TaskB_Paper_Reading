# MotionCtrl：面向视频生成的统一、灵活运动控制器

**作者：** Zhouxia Wang$^{1,\ast}$、Ziyang Yuan$^{2,\ast}$、Xintao Wang$^{3,\dagger}$、Yaowei Li$^{4,\ast}$、Tianshui Chen$^{5}$、Menghan Xia$^{6}$、Ping Luo$^{7,\dagger}$、Ying Shan$^{3}$

**机构：** $^{1}$南洋理工大学 S-Lab；$^{2}$清华大学；$^{3}$腾讯 PCG ARC Lab；$^{4}$北京大学；$^{5}$广东工业大学；$^{6}$腾讯 AI Lab；$^{7}$香港大学

**贡献说明：** $^{\ast}$实习期间完成于腾讯 PCG ARC Lab；$\dagger$通讯作者

**出处：** SIGGRAPH Conference Papers 2024（arXiv:2312.03641；DOI: 10.1145/3641519.3657518）

**arXiv：** [2312.03641](https://arxiv.org/abs/2312.03641)

**项目页：** https://wzhouxiff.github.io/projects/MotionCtrl/

**代码：** https://github.com/TencentARC/MotionCtrl

**原文 TeX：** [`arxiv/geometry_control/motionctrl/extracted/motionctrl.tex`](../../../arxiv/geometry_control/motionctrl/extracted/motionctrl.tex)

---

## 摘要

视频中的运动主要由两类构成：相机运动（camera motion，由摄像机移动引起）与物体运动（object motion，由物体移动产生）。对二者进行精确控制，是视频生成的关键能力。然而，已有工作要么主要关注其中一类运动，要么未对二者做清晰区分，从而限制了控制能力与多样性。

本文提出 **MotionCtrl**：面向视频生成的统一、灵活运动控制器，旨在有效且独立地控制相机运动与物体运动。其架构与训练策略针对相机运动、物体运动的固有性质以及不完备训练数据而精心设计。相较以往方法，MotionCtrl 有三点主要优势：

1. 能有效且独立地控制相机运动与物体运动，从而实现更细粒度的运动控制，并便于二者灵活、多样地组合。
2. 运动条件由相机姿态（camera poses）与轨迹（trajectories）决定，与外观无关（appearance-free），对生成视频中物体的外观或形状影响很小。
3. 相对可泛化：一经训练，即可适配大量相机姿态与轨迹。

大量定性与定量实验表明，MotionCtrl 优于已有方法。项目页：https://wzhouxiff.github.io/projects/MotionCtrl/。

**关键词：** AIGC、视频生成、运动控制

---

![图 1](../../../arxiv/geometry_control/motionctrl/extracted/figures/teaser.png)

**图 1. MotionCtrl 的控制结果。** MotionCtrl 能控制视频生成模型所产出视频中的相机运动与物体运动，也可在同一视频中同时控制两类运动。**强烈建议读者到项目页查看视频结果，静态图无法充分展示。**

---

## 1. 引言

视频生成，例如 text-to-video（T2V）生成（Imagen Video、Make-A-Video、MagicVideo、LVDM、Align your Latents、VideoCrafter1），旨在产出符合给定文本提示、多样且高质量的视频。与图像生成不同，图像生成只需一张图，视频生成必须在一串生成图像之间创造出一致、流畅的运动。因此，运动控制在视频生成中至关重要，但近期研究对其关注仍然有限。

视频中主要有两类运动：由摄像机移动引起的全局运动，以及由物体移动产生的局部运动（示例见图 1(c) 的 zoom out 相机姿态与摇曳玫瑰）。本文始终将二者分别称为 **camera motion** 与 **object motion**。然而，以往与视频生成运动控制相关的工作，要么主要关注其中一类，要么未对这两类运动做清晰区分。例如，AnimateDiff、Gen-2 与 PikaLab 主要用独立的 LoRA 模型或额外相机参数（如 PikaLab 中的 `-camera zoom in`）执行或触发相机运动控制。VideoComposer 与 DragNUWA 则用同一类条件同时实现相机运动与物体运动：VideoComposer 用 motion vector，DragNUWA 用 trajectory。由于未清晰区分两类运动，这些方法难以在视频生成中实现细粒度、多样化的运动控制。

本文引入 **MotionCtrl**：面向视频生成的统一、灵活运动控制器，用统一模型独立控制相机运动与物体运动。该方法使视频生成能做细粒度运动控制，并便于两类运动的灵活、多样组合。

然而，构建这样的统一运动控制器面临两个显著挑战。第一，相机运动与物体运动在运动范围与模式上差异很大。相机运动是整幅场景沿时间维的全局变换，通常用一串随时间变化的相机姿态表示；物体运动则是场景中特定物体的时序位移，通常表示为与物体相关的一组像素的轨迹。第二，尚无现成数据集同时包含 caption、相机姿态与物体运动轨迹这一整套标注。构建这样的综合数据集需要大量人力与资源。

为应对上述挑战，MotionCtrl 采用精心设计的架构、训练策略与整理后的数据集。MotionCtrl 由两个模块组成：Camera Motion Control Module（CMCM）与 Object Motion Control Module（OMCM），分别针对相机运动与物体运动的特性。二者都作为 adapter-like 模块接入已有视频生成模型。具体地，CMCM 通过时间 transformer（temporal transformers）把一串相机姿态沿时间注入视频生成模型，使生成视频的全局运动与所给相机姿态对齐；OMCM 则把物体运动信息在空间上注入视频生成模型的卷积层，指示各生成帧中物体的空间位置。本研究以 VideoCrafter1（LVDM 的增强版本）作为底层视频生成模型，全文称之为 LVDM。

依托大规模预训练视频扩散模型，并配备 adapter-like 的 CMCM 与 OMCM，这两个模块可以分开训练，从而无需同时带有 caption、相机姿态与物体运动轨迹标注的综合数据集。因此，MotionCtrl 用两个数据集即可完成：一个带 caption 与相机姿态，另一个带 caption 与物体运动轨迹。

具体地，我们引入 augmented-RealEstate10K 数据集：该数据原本带有相机运动信息，我们再用 Blip2 生成 caption，使其适合训练视频生成中的相机运动控制。此外，我们对来自 WebVid 的视频，用 ParticleSfM 提出的运动分割算法合成物体运动轨迹；再结合其原有 caption，得到的 augmented-WebVid 数据集适合学习视频生成中的物体运动控制。

用这两个标注数据集依次、分别训练 CMCM 与 OMCM 后，MotionCtrl 框架即可在统一的视频生成模型中独立或联合控制相机运动与物体运动。该方法实现相对细粒度、灵活的运动控制，使用户对生成视频有更强掌控。

经上述设计，MotionCtrl 在三方面优于以往方法：

1. 独立控制相机运动与物体运动，从而能做细粒度调整与多种运动组合，如图 1 所示。
2. 用相机姿态与轨迹作为运动条件，不影响视觉外观，保持视频中物体的自然外观。例如，MotionCtrl 能生成相机运动贴近参考视频、埃菲尔铁塔外观写实的视频，见图 4(b)。相比之下，VideoComposer 依赖稠密 motion vector，误捕获参考视频中门的形状，导致埃菲尔铁塔不自然。
3. MotionCtrl 能控制多种相机运动与轨迹，无需为每一种相机或物体运动单独微调。

本文主要贡献如下：

1. 提出 MotionCtrl，面向视频生成的统一、灵活运动控制器，可独立或联合控制生成视频中的相机运动与物体运动，实现更细粒度、更多样的运动控制。
2. 根据相机运动、物体运动的固有性质以及不完备训练数据，仔细裁剪 MotionCtrl 的架构与训练策略，有效实现视频生成中的细粒度运动控制。
3. 进行大量实验，在定性与定量上表明 MotionCtrl 优于以往相关方法。

---

## 2. 相关工作

早期视频生成研究主要依赖 Generative Adversarial Networks（GANs）或 Variational Autoencoders（VAEs）。近年来，扩散模型在图像生成中展现出显著能力，视频生成研究随之转向扩散模型。进一步结合文本或图像引导后，扩散模型能生成具有特定内容的高保真视频。尤其是在潜空间中部署扩散模型，显著提升了视频生成的计算效率，并带动大量以扩散模型为中心的下游研究。MotionCtrl 即旨在利用扩散模型控制生成视频中的运动。

在生成视频的运动控制方面，许多已有方法通过参考特定或一系列模板视频来学习运动（Tune-A-Video、AnimateDiff、LAMP、MotionDirector）。这类方法对特定运动控制有效，但通常需要为不同模板训练新模型，因而受限。

另一些工作试图实现更泛化的运动控制。例如，VideoComposer 通过额外提供的 motion vector 引入运动控制；DragNUWA 提出以初始图像、所给轨迹与文本提示为条件的视频生成。然而，这些方法的运动控制相对粗粒度，未能细粒度地解耦视频中的相机运动与物体运动。

与这些工作不同，我们提出 MotionCtrl：统一、灵活的运动控制器，可用相机姿态、物体轨迹，或二者组合，来控制生成视频的运动，从而实现更细粒度、更灵活的视频生成控制。

---

## 3. 方法

![图 2](../../../arxiv/geometry_control/motionctrl/extracted/figures/framework_v2.png)

**图 2. MotionCtrl 框架。** MotionCtrl 在 LVDM 的 Denoising U-Net 上扩展 Camera Motion Control Module（CMCM）与 Object Motion Control Module（OMCM）。如 (b) 所示，CMCM 把相机姿态序列 $RT$ 接入 LVDM 的 temporal transformers：将 $RT$ 拼接到第二个 self-attention 模块的输入上，再用专门的轻量全连接层提取相机姿态特征以供后续处理。OMCM 用卷积层与下采样从 $Trajs$ 得到多尺度特征，并在空间上注入 LVDM 的卷积层以引导物体运动。再给定文本提示，LVDM 从噪声生成符合提示的视频，背景与物体运动分别反映指定的相机姿态与轨迹。结果视频中，马沿其轨迹移动，同时背景向左移动，与相机向右运动一致。

### 3.1 预备知识

Latent Video Diffusion Model（LVDM）旨在由文本提示引导，生成高质量、多样的视频。它在潜空间中使用去噪扩散模型（U-Net），以兼顾空间与时间效率。为此，它构建轻量 3D autoencoder，由编码器 $\mathcal{E}$ 与解码器 $\mathcal{D}$ 组成：分别把原始视频编码到潜空间，再把去噪后的潜特征重建为视频。其去噪 U-Net（记为 $\epsilon_\theta$）由一系列块构成，块中包含卷积层、spatial transformers 与 temporal transformers（见图 2）。优化采用噪声预测损失：

$$
\mathcal{L} = \mathbb{E}_{z_0, c, \epsilon \sim \mathcal{N}(0, I), t}\left[ \lVert \epsilon - \epsilon_\theta(z_t, t, c) \rVert_2^2 \right],
\tag{1}
$$

其中 $c$ 表示文本提示，$z_0$ 是用 $\mathcal{E}$ 得到的潜码，$t$（$t \in [0, T]$）表示时间步，$z_t$ 是把高斯噪声 $\epsilon$ 按权重加到 $z_0$ 上得到的含噪潜特征：

$$
z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon, \quad \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i,
\tag{2}
$$

其中 $\alpha_t$ 用于按时间步 $t$ 调度噪声强度。

### 3.2 MotionCtrl

图 2 展示 MotionCtrl 的框架。为解耦相机运动与物体运动，并独立控制这两类运动，MotionCtrl 包含两个主要组件：Camera Motion Control Module（CMCM）与 Object Motion Control Module（OMCM）。考虑到相机运动的全局性与物体运动的局部性，CMCM 与 LVDM 中的 temporal transformers 交互，OMCM 则在空间上与 LVDM 中的卷积层配合。此外，由于缺少同时带有高质量视频片段、caption、相机姿态与物体运动轨迹的训练数据，我们采用多步训练。下文将详细描述 CMCM、OMCM 及其对应的训练数据集与训练策略。

#### 3.2.1 Camera Motion Control Module（CMCM）

CMCM 是由若干全连接层构成的轻量模块。由于相机运动是视频帧之间的全局变换，CMCM 通过 LVDM 的 temporal transformers 与之配合。LVDM 的 temporal transformers 通常包含两个 self-attention 模块，用于视频帧之间的时间信息融合。为尽量减小对 LVDM 生成性能的影响，CMCM 只介入 temporal transformers 中的第二个 self-attention 模块。

具体地，CMCM 以一串相机姿态 $RT=\{RT_0, RT_1, \dots, RT_{L-1}\}$ 为输入。本文将相机姿态表示为其 $3\times 3$ 旋转矩阵与 $3\times 1$ 平移向量，因此 $RT \in \mathbb{R}^{L \times 12}$，其中 $L$ 为生成视频的长度。如图 2(b) 所示，在与 temporal transformer 中第一个 self-attention 的输出 $y_t \in \mathbb{R}^{H\times W\times L\times C}$ 沿最后一维拼接之前，先把 $RT$ 扩展为 $H \times W \times L \times 12$。其中 $H$、$W$ 是生成视频的潜空间空间尺寸，$C$ 是 $y_t$ 的通道数。拼接结果再用全连接层投影回 $H \times W \times L \times C$，然后送入 temporal transformer 的第二个 self-attention 模块。

#### 3.2.2 Object Motion Control Module（OMCM）

如图 2 所示，MotionCtrl 用轨迹（$Trajs$）控制生成视频的物体运动。通常，一条轨迹表示为空间位置序列 $\{(x_0, y_0), (x_1, y_1), \dots, (x_{L-1}, y_{L-1})\}$，其中 $(x_i, y_i)$，$i\in [0, L-1]$ 表示该轨迹在第 $i$ 帧经过空间位置 $(x, y)$。特别地，$x \in [0, \hat{W})$，$y \in [0, \hat{H})$，其中 $\hat{H}$、$\hat{W}$ 分别是 $z_T$ 的高与宽。为显式暴露物体的运动速度，我们将 $Trajs$ 表示为

$$
\{(0, 0), (u_{(x_1, y_1)}, v_{(x_1, y_1)}), \dots, (u_{(x_{L-1}, y_{L-1})}, v_{(x_{L-1}, y_{L-1})})\},
$$

其中

$$
u_{(x_i, y_i)} = x_{i} - x_{i-1};\quad v_{(x_i, y_i)} = y_{i} - y_{i-1};\quad 0 < i < L.
\tag{3}
$$

第一帧，以及后续帧中轨迹未经过的其他空间位置，均记为 $(0, 0)$。最终 $Trajs \in \mathbb{R}^{L\times \hat{H} \times \hat{W} \times 2}$。

$Trajs$ 通过 OMCM 注入 LVDM，对应图 2 中的紫色块。OMCM 由多层卷积与下采样组成，从 $Trajs$ 提取多尺度特征，并相应地加到 LVDM 卷积层的输入上。受 T2I-Adapter 启发，轨迹只作用于 Denoising U-Net 的 encoder，以在生成视频质量与物体运动控制能力之间取得平衡。

#### 3.2.3 训练策略与数据构建

要在用文本提示生成视频的同时控制相机与物体运动，训练数据集中的视频片段必须同时带有 caption、相机姿态与物体运动轨迹标注。目前尚无如此完备的数据集，组装一份也需大量人力与资源。为此，我们引入多步训练策略，并用分别针对其运动控制需求而增强的数据集，训练所提出的 CMCM 与 OMCM。

**学习 Camera Motion Control Module（CMCM）。** CMCM 只需带 caption 与相机姿态标注的视频片段。考虑到 RealEstate10K 含超过 6 万条视频，且相机姿态标注相对干净，我们将其作为 CMCM 的训练数据。但把 RealEstate10K 用于 MotionCtrl 有两个潜在问题：（1）场景多样性有限，主要来自房地产视频，可能损害生成视频质量；（2）缺少 T2V 模型所需的 caption。

针对第一个问题，我们采用 adapter-like 的控制模块（CMCM）：仅新增若干 MLP 层，以及 LVDM temporal transformers 中的第二个 self-attention 模块可训练，冻结 LVDM 的绝大部分参数以保留其生成质量。由于 temporal transformers 主要学习全局运动，RealEstate10K 有限的场景多样性几乎不影响 LVDM 的生成质量。表 2 的定量结果可资印证：FID 与 FVD 表明，MotionCtrl 生成的视频质量与 LVDM 相当。

针对第二个问题，我们用图像 caption 算法 Blip2 为 RealEstate10K 中每个视频片段生成 caption。细节见补充材料。

**学习 Object Motion Control Module（OMCM）。** OMCM 需要带 caption 与物体运动轨迹的视频片段，社区目前缺少此类数据。为满足需求，我们用 ParticleSfM 在 WebVid 上合成物体运动轨迹。WebVid 是大规模带 caption 的视频数据集，常用于 T2V 生成。ParticleSfM 虽主要是 structure-from-motion 系统，但包含基于轨迹的运动分割模块，用于滤除动态场景中影响相机轨迹估计的动态轨迹。该运动分割模块得到的动态轨迹，恰好满足 MotionCtrl 的需求；我们用该模块为 WebVid 中约 243,000 条视频合成运动物体轨迹。图 3(b) 给出一例：轨迹主要对应一个运动中的人。合成细节见补充材料。

为避免用户必须提供如图 3(b) 那样的稠密轨迹（这对用户不友好），MotionCtrl 需要能根据用户提供的稀疏（一条或少数几条）轨迹控制运动物体。因此，OMCM 用从合成稠密轨迹中随机选取的 $n \in [1, N]$ 条轨迹训练（$N$ 为每条视频的最大轨迹数，见图 3(c)）。

然而，这些选出的稀疏轨迹往往过于分散，不利于有效训练。受 DragNUWA 启发，我们对稀疏轨迹施加 Gaussian filter（图 3(d)），并先用稠密轨迹训练 OMCM，再用稀疏轨迹微调。

在此训练阶段，LVDM 与 CMCM 均已训练完毕并冻结，只训练 OMCM。该策略保证：在有限数据上为 OMCM 增加物体运动控制能力的同时，对 LVDM 与 CMCM 的影响最小。该阶段完成后，同时给出相机姿态与物体轨迹，即可灵活控制生成视频中的相机运动与物体运动。

![图 3](../../../arxiv/geometry_control/motionctrl/extracted/figures/trajectories.png)

**图 3. 用于物体运动控制的轨迹。** 用 ParticleSfM 从视频片段提取物体运动轨迹，有效把物体运动与相机引起的运动解耦。稠密轨迹可能编码物体形状，且推理时难以设计，故我们用从稠密轨迹中采样的稀疏轨迹训练 OMCM。这些稀疏轨迹过于分散、不利于有效学习，随后用 Gaussian filter 加以平滑。

---

## 4. 实验

### 4.1 实验设置

#### 实现细节

MotionCtrl 建立在 LVDM / VideoCrafter1 框架之上，该框架以 $256\times 256$ 分辨率、16 帧序列训练。它可较容易地适配到结构相似的其他视频生成模型（如 AnimateDiff），并遵循各模型自身设定。轨迹最大条数 $N$ 固定为 8。CMCM 与 OMCM 均用 Adam 优化器，batch size 为 128，学习率为 $1\times 10^{-4}$，在 8 张 NVIDIA Tesla V100 上训练。CMCM 通常约需 50,000 次迭代收敛。OMCM 先在稠密轨迹上训练 20,000 次迭代，再用稀疏轨迹微调 20,000 次迭代。

#### 评测数据集

1. **相机运动控制评测集**涵盖两类相机姿态：基础相机姿态（pan left、pan right、pan up、pan down、zoom in、zoom out、anticlockwise rotation、clockwise rotation），以及相对复杂的相机姿态。本文中 “complex camera poses” 指超出基础姿态的相机运动：基础姿态沿单一直线方向移动，复杂姿态包含多个方向的运动。复杂姿态来自 RealEstate10K 测试集，或用 ParticleSfM 在 WebVid 与 HD-VILA 的视频上合成。
2. **物体运动控制评测集**由 283 个样本构成，配合多样的手工轨迹与提示。

评测数据集的构建细节见补充材料。

#### 评测指标

1. 生成视频质量用 Fréchet Inception Distance（**FID**）、Fréchet Video Distance（**FVD**）与 CLIP Similarity（**CLIPSIM**）评估，分别衡量视觉质量、时间一致性，以及与文本的语义相似度。FID 与 FVD 的参考视频为来自 WebVid 的 1000 条视频。
2. 相机与物体运动控制的有效性，分别通过预测与 ground truth 相机姿态、物体轨迹之间的 Euclidean distance 量化。预测视频的相机姿态与物体轨迹用 ParticleSfM 提取。这两个指标分别称为 **CamMC** 与 **ObjMC**。
3. 我们还进行用户研究作为主观定量评测，细节因篇幅限制放在补充材料。

![图 4(a)](../../../arxiv/geometry_control/motionctrl/extracted/figures/fig_comparison_camerapose_basic.png)

**图 4(a). 基础姿态上的相机运动控制。** MotionCtrl 与 AnimateDiff 都能执行 zoom，但 MotionCtrl 可适配不同的相机运动速度。

![图 4(b)](../../../arxiv/geometry_control/motionctrl/extracted/figures/fig_comparison_camerapose_complex_v1.png)

**图 4(b). 相对复杂姿态上的相机运动控制。** VideoComposer 用 RealEstate10K 原始视频提取 motion vector，捕获到门等非预期形状，导致不自然结果（见第 12 帧）。MotionCtrl 则生成运动与相机姿态更接近、相对自然的视频。

### 4.2 与 SOTA 方法的比较

为验证 MotionCtrl 对相机运动与物体运动的控制能力，我们与两种领先方法比较：AnimateDiff 与 VideoComposer。AnimateDiff 用 8 个独立 LoRA 模型控制视频中的 8 种基础相机运动（如 panning 与 zooming）；VideoComposer 用 motion vector 操控视频运动，不区分相机运动与物体运动。DragNUWA 与本研究相关，但其代码未公开，无法直接比较。此外，DragNUWA 仅用从光流提取的轨迹学习运动控制，无法细粒度区分前景物体与背景的运动，从而限制其对相机与物体运动的精确控制。

我们分别比较相机运动控制与物体运动控制，并展示 MotionCtrl 在视频生成中灵活组合两类控制的能力。更多比较及视频比较见补充材料。

#### 相机运动控制

我们用基础姿态与相对复杂姿态评估相机运动控制。AnimateDiff 仅限于基础相机姿态；VideoComposer 通过从所给视频提取 motion vector 来处理复杂姿态。定性结果见图 4。对基础姿态，MotionCtrl 与 AnimateDiff 都能生成相机向前运动的视频，但 MotionCtrl 能生成不同速度的相机运动，AnimateDiff 不可调。对复杂姿态（相机先向左前方再向前），VideoComposer 能用提取的 motion vector 模仿参考视频的相机运动；但稠密 motion vector 会无意捕获物体形状——参考视频第 12 帧中门的轮廓——导致埃菲尔铁塔外观不自然。MotionCtrl 由旋转与平移矩阵引导，生成更自然、相机运动更接近参考的视频。

表 1 的定量结果表明，无论基础姿态还是相对复杂姿态，MotionCtrl 在 CamMC 上均优于 AnimateDiff 与 VideoComposer。此外，MotionCtrl 在 CLIPSIM、FID、FVD 衡量的文本相似度与质量指标上也更好。

**表 1.** 与 AnimateDiff、VideoComposer 的定量比较。**MotionCtrl** 在相机与物体运动控制上均优于竞品，同时更好地保持文本相似度与视频生成质量。

| Method | AnimateDiff | VideoComposer | **MotionCtrl** |
| --- | ---: | ---: | ---: |
| **CamMC**（Basic Poses） | 0.0548 | — | **0.0289** |
| **CamMC**（Complex Poses） | — | 0.0950 | **0.0735** |
| **ObjMC** | — | 36.8351 | **28.877** |
| **CLIPSIM $\uparrow$** | 0.2144 | 0.2214 | **0.2319** |
| **FID** | 157.73 | 130.97 | **124.09** |
| **FVD** | 1815.88 | 1004.99 | **852.15** |

![图 5](../../../arxiv/geometry_control/motionctrl/extracted/figures/fig_comparison_traj.png)

**图 5. 物体运动控制的定性比较。** VideoComposer 与 MotionCtrl 都能生成沿给定轨迹（红色曲线）移动的物体，但 MotionCtrl 在每一帧更精确地跟随轨迹，如绿色点所示。

#### 物体运动控制

我们将 MotionCtrl 与 VideoComposer 在物体运动控制上比较；VideoComposer 使用从轨迹提取的 motion vector。定性结果见图 5。红色曲线为给定轨迹，绿色点表示对应帧中物体的期望位置。视觉比较表明，MotionCtrl 能生成运动更接近给定轨迹的物体，而 VideoComposer 的结果在若干帧上偏离，凸显 MotionCtrl 更强的物体运动控制能力。表 1 中的 ObjMC 同样表明，MotionCtrl 的物体运动控制优于 VideoComposer。

#### 相机运动与物体运动的组合

MotionCtrl 不仅能在同一视频中独立控制相机运动或物体运动，也能对二者做一体化控制。如图 1(b)(c) 所示：仅施加一条轨迹时，MotionCtrl 主要生成沿该路径摇曳的玫瑰；再引入 zoom-out 相机姿态后，玫瑰与背景都会按指定轨迹与相机运动被动画化。

更多 MotionCtrl 结果见图 8、补充材料以及 demo 视频。

![图 8](../../../arxiv/geometry_control/motionctrl/extracted/figures/sig24_more_complex.png)

**图 8.** MotionCtrl 的更多结果：包括仅由相机姿态或仅由物体轨迹控制的结果，以及同时用相机姿态与物体轨迹控制的结果。

**表 2.** 相机运动控制的 ablation。接入 LVDM temporal transformers 的 Camera Motion Control Module（CMCM）能有效控制相机运动，并保持 LVDM 的视频质量。

| Method | **CamMC** | **CLIPSIM $\uparrow$** | **FID** | **FVD** |
| --- | ---: | ---: | ---: | ---: |
| LVDM | 0.9010 | 0.2359 | **130.62** | 1007.63 |
| Time Embedding | 0.0887 | 0.2361 | 132.74 | 1461.36 |
| Spatial Cross-Attention | 0.0857 | 0.2357 | 153.86 | 1306.78 |
| Spatial Self-Attention | 0.0902 | **0.2384** | 146.37 | 1303.58 |
| **Temporal Transformer** | **0.0289** | 0.2355 | 132.36 | **1005.24** |

### 4.3 消融实验

#### CMCM 的接入位置

我们测试把相机姿态与 LVDM 中的 time embedding、spatial cross-attention 或 spatial self-attention 模块结合，以实现相机运动控制。这类做法在 sketch、depth 等其他控制（ControlNet、T2I-Adapter）上取得过成功，但未能赋予 LVDM 相机控制能力，见表 2 的 CamMC 以及图 6 的可视化：其 CamMC 接近原始 LVDM。原因是这些组件主要关注空间内容生成，对编码在相机姿态中的相机运动不敏感。

相反，把 CMCM 接入 LVDM 的 temporal transformers 显著改善相机运动控制，表 2 中 CamMC 降至 $0.0289$。相机运动主要造成随时间的全局视角变换，把相机姿态融入 LVDM 的时间块符合这一性质，从而能在视频生成中有效控制相机运动。

![图 6](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_ablation_camerapose_paper.png)

**图 6. 关于 CMCM 与 LVDM 接入位置的 ablation 定性结果。** 把 MotionCtrl 的 CMCM 接入 LVDM 的 temporal transformers，相较其他设置显著改善相机运动控制。

#### 稠密轨迹 vs. 稀疏轨迹

OMCM 先用 ParticleSfM 提取的稠密物体运动轨迹训练，再用稀疏轨迹微调。我们将其与仅用稠密或仅用稀疏轨迹训练 OMCM 相比较。表 3 与图 7 表明：仅用稠密轨迹训练效果较差，原因是训练与推理阶段不一致（推理时提供的是稀疏轨迹）。

仅用稀疏轨迹训练虽优于仅用稠密轨迹，仍不及混合方法，因为稀疏轨迹单独提供的信息有限。稠密轨迹提供更丰富的信息、加快学习；随后用稀疏轨迹微调，使 OMCM 能适应推理时遇到的稀疏性。

**表 3.** 物体运动控制的 ablation。Object Motion Control Module（OMCM）先在稠密物体运动轨迹上训练、再用稀疏轨迹微调，优于仅用稠密或仅用稀疏轨迹训练的版本。

| Method | **ObjMC** | **CLIPSIM $\uparrow$** | **FID** | **FVD** |
| --- | ---: | ---: | ---: | ---: |
| Dense | 54.4114 | 0.2352 | 175.8622 | 2227.87 |
| Sparse | 34.6937 | 0.2365 | 158.5553 | 2385.39 |
| **Dense + Sparse** | **25.1198** | **0.2342** | **149.2754** | **2001.57** |

![图 7](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_ablation_traj_paper.png)

**图 7. “稠密轨迹 vs. 稀疏轨迹” ablation 的定性结果。** 仅用稠密轨迹训练的模型无法控制生成视频中的物体运动。先稠密、再稀疏微调的模型，物体运动控制精度优于仅用稀疏轨迹训练的模型。

#### 训练策略

受可用训练数据限制，我们为 MotionCtrl 提出多步训练策略：先用 RealEstate10K 训练 CMCM，再用合成的物体运动轨迹训练 OMCM。为充分评估该做法，我们实验颠倒顺序：先训练 OMCM 再训练 CMCM。该顺序不影响相机运动控制，因为 OMCM 组件不参与 CMCM 训练。但它导致物体运动控制性能下降：随后训练 CMCM 会调整 LVDM temporal transformers 的部分参数，破坏 OMCM 初始训练所获得的物体运动控制适应。因此，尽管多步策略是数据限制下的折中，其结构是刻意的：先 CMCM、后 OMCM，以保证相机与物体运动控制都有更好表现。

### 4.4 将 MotionCtrl 部署到 AnimateDiff

我们也将 MotionCtrl 部署到 AnimateDiff。因此，可以用我们调整后的 AnimateDiff，配合社区中各种 LoRA 模型，控制所生成视频的运动。复杂相机运动控制与物体运动控制的可视化结果见图 9 与图 10。更多结果见补充材料。

![图 9](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_animate_complex_paper.png)

**图 9.** 部署到 AnimateDiff 上的复杂相机运动控制结果。

![图 10](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_animate_traj_paper.png)

**图 10.** 部署到 AnimateDiff 上的物体运动控制结果。

---

## 5. 局限性

作为在统一视频生成模型中控制相机运动与物体运动的初步探索，MotionCtrl 展示了有前景、有启发的结果。然而，在同一视频中同时用复杂相机轨迹与复杂物体轨迹做控制时，需要仔细设计这些轨迹才能得到自然、和谐的结果，成功率相对较低。如何提高生成视频中同时控制相机与物体运动的精度，仍需进一步研究。

---

## 6. 结论

本文提出 MotionCtrl：统一、灵活的控制器，可独立或组合地控制视频生成模型所得视频中的相机运动与物体运动。为此，MotionCtrl 针对相机运动与物体运动的特定性质，仔细裁剪相机运动控制模块与物体运动控制模块，并采用多步训练策略，用精心增强的数据集训练这两个模块。包括定性与定量评估在内的全面实验，展示了所提出 MotionCtrl 在相机与物体运动控制上的优越性。

---

## 附录

补充材料提供所提出 MotionCtrl 的额外结果与更深入分析。**为获得更直观的理解，强烈建议读者访问项目页查看视频结果。** 补充材料结构如下：

- 训练数据构建细节（A 节）
- 评测数据集细节（B 节）
- 更多定量与定性结果（C 节）
- 将 MotionCtrl 扩展到 AnimateDiff 框架的更多结果（D 节）
- 与以往相关工作的更多讨论（E 节）

### A. 训练数据构建细节

**Augmented-RealEstate10K.** MotionCtrl 中的 CMCM 用从 RealEstate10K 增强的数据训练。RealEstate10K 原本包含带相机姿态标注的视频。为适配 MotionCtrl，我们进一步用图像 caption 算法 Blip2 为每条视频合成 caption。具体地，我们按特定间隔抽取帧——视频的首帧、四分之一处、一半处、四分之三处与末帧——再用 Blip2 预测其 caption。将这些 caption 拼接，形成每个视频片段的综合描述。有了这些 caption，我们在 RealEstate10K 上训练 CMCM，从而在 LVDM 等视频生成模型中实现有效的相机运动控制。

**Augmented-WebVid.** MotionCtrl 中的 OMCM 用从 WebVid 增强的数据训练。WebVid 是大规模带 caption 的视频数据集，常用于 T2V 生成。为适配 MotionCtrl，我们进一步用 ParticleSfM 为 WebVid 中的视频合成物体运动轨迹。ParticleSfM 虽主要是 structure-from-motion 系统，但包含基于轨迹的运动分割模块，用于滤除动态场景中影响相机轨迹生成的动态轨迹。该运动分割模块得到的动态轨迹恰好满足 MotionCtrl 的需求，我们用该模块合成 MotionCtrl 所需的运动物体轨迹。然而，尽管有效，ParticleSfM 并不省时：处理一段 32 帧视频约需 2 分钟。为缓解时间效率问题，我们从每条 WebVid 视频中随机选取 32 帧，帧间隔 $s \in [1, 16]$，以合成物体运动轨迹。该做法共得到 243,000 条满足 OMCM 训练需求的视频片段。

### B. 评测数据集细节

本文构建两个评测数据集，分别独立评估所提出 MotionCtrl 在相机运动控制与物体运动控制上的有效性。

**相机运动控制评测数据集。** 该数据集共含 **407** 个样本，覆盖两类相机姿态：

1. 80（$8\times 10$）个样本：8 种基础相机姿态序列（pan left、pan right、pan up、pan down、zoom in、zoom out、anticlockwise rotation、clockwise rotation）与 10 个提示。
2. 200（$20\times 10$）个样本：从 RealEstate10K 测试集随机选取的 20 条相对复杂相机姿态序列，与 10 个提示。
3. 100 个样本：用 ParticleSfM 在 WebVid 上合成的 100 条相对复杂相机姿态，以及来自 VBench 的 100 个提示。
4. 27 个样本：用 ParticleSfM 在 HD-VILA 上合成的 27 条相对复杂相机姿态，以及来自 VBench 的 27 个提示。

为直观感知相机运动，我们将 8 种基础相机姿态以及来自 RealEstate10K 的 20 条相对复杂相机姿态可视化于图 11。如正文所述，本文中 “complex camera poses” 指超出基础相机姿态列表的相机运动，包括同一相机姿态序列中的转向与自转。

**物体运动控制评测数据集。** 该评测集共含 283 个样本，由 74 条多样轨迹与 77 个提示构成。需要说明：为验证 MotionCtrl 在物体运动控制上的有效性，评测集将一条轨迹与若干不同提示配对，或将一个提示与若干不同轨迹配对。为直观感知手工轨迹，评测集中采用的 19 条轨迹绘于图 12。

这些评测数据集将会发布。

**请注意：我们构建的评测数据集主要用于定量评估所提出 MotionCtrl 在视频生成中对相机与物体运动控制的表现。MotionCtrl 能够处理评测集未包含的更广泛相机姿态与轨迹。**

![图 11(a)](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/camera_8_paper.png)

**图 11(a).** 8 种基础相机姿态。

![图 11(b)](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/camera_20_paper.png)

**图 11(b).** 来自 RealEstate10K 测试集的 20 条相对复杂相机姿态。

**图 11.** 相机运动控制评测数据集由 8 种基础相机姿态与 20 条相对复杂相机姿态组成，相对复杂姿态来自 RealEstate10K 测试集。该数据集用于定量评估所提出 MotionCtrl 在生成视频中控制广泛多样相机运动的有效性。

![图 12](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/traj_19_paper.png)

**图 12.** 物体运动控制评测数据集涵盖 19 条轨迹，绿色与蓝色点分别表示每条轨迹的起点与终点。该数据集用于定量评估所提出 MotionCtrl 在生成视频中控制物体运动的有效性。

### C. 更多定量与定性结果

#### 更多定量结果

**相对复杂相机运动控制的更多定量比较。** 正文中，相对复杂相机姿态的定量结果是对来自 RealEstate10K、WebVid 与 HD-VILA 的全部复杂相机姿态的统计。各数据集的统计结果见表 4：无论是从 RealEstate10K 提取的相机姿态，还是用 ParticleSfM 合成的姿态（WebVid 与 HD-VILA 的相机姿态），我们的 MotionCtrl 在相机运动控制、文本相似度与生成质量上均优于 VideoComposer。

**表 4.** 与 VideoComposer 的定量比较。我们的 **MotionCtrl** 在来自 RealEstate10K、WebVid 与 HD-VILA 的三组相对复杂相机姿态上均更好。

| Dataset | Method | CamMC | CLIPSIM $\uparrow$ | FID | FVD |
| --- | --- | ---: | ---: | ---: | ---: |
| RealEstate10K | VideoComposer | 0.1073 | 0.2219 | 134.97 | 1045.82 |
| RealEstate10K | **MotionCtrl** | **0.0840** | **0.2324** | **130.29** | **934.37** |
| WebVid | VideoComposer | 0.0702 | 0.2147 | 106.89 | 733.09 |
| WebVid | **MotionCtrl** | **0.0589** | **0.2268** | **102.13** | **612.84** |
| HD-VILA | VideoComposer | 0.0953 | 0.2429 | 190.54 | 1709.59 |
| HD-VILA | **MotionCtrl** | **0.0499** | **0.2473** | **159.52** | **1129.40** |

**用户研究。** 为更全面评估，我们组织 34 名参与者的用户研究，评估 VideoComposer 与 MotionCtrl 的结果。结果用物体轨迹以及覆盖 RealEstate10K、WebVid、HD-VILA 的相对复杂相机姿态生成。评估标准包括 Video Quality、Text Similarity 与 Motion Similarity。参与者还需对每对比较表达总体偏好。表 5 的统计结果表明，在所有评估方面，超过 90% 的参与者更偏好我们的结果。尽管 VideoComposer 在以 motion vector 为条件的运动控制上表现不错，但其生成视频往往因参考视频 motion vector 捕获到的物体形状而显得不自然、怪异。因此，用户更偏好我们相对自然的结果。

**表 5.** 用户研究。相较 VideoComposer 生成的结果，我们的 **MotionCtrl** 在所有评估方面获得更多偏好。

| Method | VideoComposer | **MotionCtrl** |
| --- | ---: | ---: |
| **Quality $\uparrow$** | 0.0628 | **0.9372** |
| **TextSimilarity $\uparrow$** | 0.0772 | **0.9228** |
| **MotionSimilarity $\uparrow$** | 0.086 | **0.9140** |
| **OverallPreference $\uparrow$** | 0.0739 | **0.9261** |

#### 更多定性结果

**与 VideoComposer 的更多定性比较。** 图 13 与图 14 分别给出 VideoComposer 与所提出 MotionCtrl 在相对复杂相机轨迹与物体轨迹上的额外定性结果。这些结果表明，MotionCtrl 在生成视频的相机与物体运动控制上均优于 VideoComposer。此外，MotionCtrl 生成的视频质量更高，生成内容与提示更对齐。

![图 13](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_more_camera.png)

**图 13. 与 VideoComposer 在相机运动控制上的更多定性比较。** MotionCtrl 生成的视频更能跟随相机姿态，无论姿态来自 RealEstate10K，还是用 ParticleSfM 在 WebVid 与 HD-VILA 视频上合成。此外，MotionCtrl 的结果质量更高。

![图 14](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_more_traj.png)

**图 14. 与 VideoComposer 在物体运动控制上的更多定性比较。** MotionCtrl 生成的视频在每一帧跟随轨迹的能力更强，整体质量也更高。

**更多 MotionCtrl 结果。** 本节展示 MotionCtrl 在相机运动控制、物体运动控制以及组合运动控制上的额外结果。**需要强调：所有结果均由同一个训练好的 MotionCtrl 模型得到，无需为不同相机姿态或轨迹额外微调。**

具体地，图 17 展示由 8 种基础相机姿态引导的 MotionCtrl 相机运动控制结果，包括 pan up、pan down、pan left、pan right、zoom in、zoom out、anticlockwise rotation 与 clockwise rotation。这些姿态可视化于图 11(a)。这表明我们的 MotionCtrl 能在统一模型中整合多种基础相机运动控制，而 AnimateDiff 则需要为每种相机运动配备不同的 LoRA 模型。

![图 17](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_more_8_pose_paper.png)

**图 17.** 部署在 LVDM 上的 MotionCtrl，由 8 种基础相机姿态引导的结果：pan up、pan down、pan left、pan right、zoom in、zoom out、anticlockwise rotation、clockwise rotation（这些相机姿态的可视化见图 11(a)）。**需要强调：所有结果均由同一个 MotionCtrl 模型得到，无需为不同相机姿态额外微调。**

图 15 展示由相对复杂相机姿态引导的 MotionCtrl 相机运动控制结果。**这些复杂相机姿态不同于基础相机姿态：它们在同一相机姿态序列中包含相机转向或自转。** 结果表明，给定一串相机姿态，我们的 MotionCtrl 能生成自然视频：内容与文本提示对齐，相机运动对应所给复杂相机姿态。

![图 15](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_more_compex_pose_part2.png)

**图 15.** 部署在 LVDM 上的 MotionCtrl，由相对复杂相机姿态引导的结果。**与只含简单方向运动的基础相机姿态不同，这些复杂相机姿态在同一相机姿态序列中包含转向或自转。** 生成视频中的相机运动紧密跟随引导姿态，生成内容与文本提示对齐。

图 18 展示由特定轨迹引导的 MotionCtrl 物体运动控制结果。给定相同轨迹与不同文本提示，MotionCtrl 能生成不同物体、但物体运动相同的视频。

![图 18](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_more_traj_paper.png)

**图 18. 部署在 LVDM 上的 MotionCtrl，由轨迹引导的结果。** 轨迹中的绿色点表示起点。给定相同轨迹，我们的模型能按文本提示生成不同物体，并保持相同的物体运动。当同一视频中存在多条轨迹时，模型能在同一生成视频中同时控制不同物体的运动。

图 16 给出同时组合相机运动控制与物体运动控制的结果。轨迹相同、相机姿态不同时，生成视频中的马有不同表现。

![图 16](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_combine_paper.png)

**图 16.** 部署在 LVDM 上的 MotionCtrl 组合相机运动与物体运动控制的结果。轨迹相同、相机姿态不同时，生成视频中的马有不同表现。

### D. 将 MotionCtrl 部署到 AnimateDiff 的更多结果

我们也将 MotionCtrl 部署到 AnimateDiff。因此，可以用我们微调后的 AnimateDiff，配合社区中各种 LoRA 模型，控制所生成视频的运动。相对复杂相机运动控制与物体运动控制的结果已在正文中给出；此处给出基础相机运动控制结果：图 19 与图 20。这些结果由我们的 MotionCtrl 配合 CIVITAI 提供的不同 LoRA 模型生成。它们表明 MotionCtrl 的泛化性：可适配不同的视频生成模型。

![图 19](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_animate_pose_paper.png)

**图 19.** 部署到 AnimateDiff 上的 MotionCtrl 相机运动控制结果。由 8 种基础相机姿态引导。

![图 20](../../../arxiv/geometry_control/motionctrl/extracted/figures_supp/supp_animate_zoomspeed_paper.png)

**图 20.** 部署到 AnimateDiff 上的 MotionCtrl 相机运动控制结果。我们的 MotionCtrl 不仅能控制生成视频的相机运动，也能控制其运动速度。

### E. 与相关工作的进一步讨论

为进一步说明所提出 MotionCtrl 的优势，我们与以往相关工作做了比较分析，详见表 6。

**表 6.** 所提出 MotionCtrl 与相关工作的差异。AnimateDiff（此处指其提供的运动控制 LoRA 模型）、Tune-A-Video、LAMP 与 MotionDirector 通过从一条或一系列模板视频提取运动来实现运动控制，且不同模板视频需要不同模型；所提出 MotionCtrl 使用统一模型。此外，这些方法学到的运动由模板视频决定，且不区分相机运动与物体运动。另一方面，尽管 MotionDirector 与 VideoComposer 分别用 motion vector 与轨迹、以统一模型实现运动控制，它们同样不区分相机运动与物体运动。相比之下，所提出 MotionCtrl 用统一模型，分别由相机姿态与轨迹引导，可独立、灵活地控制生成视频的相机运动与物体运动。

| Method | Require Fine-tuning | Motion sources | Distinguish Camera & Object Motion |
| --- | :---: | --- | :---: |
| AnimateDiff | ✓ | template videos | ✗ |
| Tune-A-Video | ✓ | template video | ✗ |
| LAMP | ✓ | template videos | ✗ |
| MotionDirector | ✓ | template videos | ✗ |
| VideoComposer | ✗ | motion vectors | ✗ |
| DragNUWA | ✗ | trajectories | ✗ |
| **MotionCtrl**（Ours） | ✗ | camera poses & trajectories | ✓ |

AnimateDiff（指其提供的运动控制 LoRA 模型）、Tune-A-Video、LAMP 与 MotionDirector 通过从一条或多条模板视频提取运动来实现运动控制。该方法需要为每个模板视频或模板视频集合训练不同模型。此外，这些方法学到的运动完全由模板视频决定，且未能区分相机运动与物体运动。

类似地，尽管 MotionDirector 与 VideoComposer 分别用 motion vector 与轨迹、以统一模型实现运动控制，它们也不区分相机运动与物体运动。

相比之下，所提出 MotionCtrl 利用统一模型，分别由相机姿态与轨迹引导，可独立、灵活地控制生成视频中广泛的相机运动与物体运动，从而对视频生成过程提供更细粒度的控制。
