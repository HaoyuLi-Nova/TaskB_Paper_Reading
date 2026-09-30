# CameraCtrl：为视频扩散模型赋予相机控制能力

**作者：** Hao He$^{1,2,*}$、Yinghao Xu$^{3}$、Yuwei Guo$^{1,2}$、Gordon Wetzstein$^{3}$、Bo Dai$^{2}$、Hongsheng Li$^{1,\dagger}$、Ceyuan Yang$^{2,\dagger}$

**机构：** $^{1}$香港中文大学；$^{2}$上海人工智能实验室；$^{3}$斯坦福大学

**贡献说明：** $^{*}$作者于上海人工智能实验室实习期间完成此工作；$\dagger$通讯作者

**出处：** ICLR 2025（arXiv:2404.02101）

**arXiv：** [2404.02101](https://arxiv.org/abs/2404.02101)

**项目页：** https://hehao13.github.io/projects-CameraCtrl/

**代码：** https://github.com/hehao13/CameraCtrl

**原文 TeX：** [`arxiv/geometry_control/cameractrl/extracted/main.tex`](../../../arxiv/geometry_control/cameractrl/extracted/main.tex)

---

## 摘要

可控性在视频生成中至关重要，它使用户能更精确地创建与编辑内容。然而，现有模型缺乏对相机姿态（camera pose）的控制——而相机姿态作为电影语言，能够表达更深层次的叙事细微差别。为缓解这一问题，我们提出 **CameraCtrl**，为视频扩散模型提供精确的相机姿态控制。本方法探索有效的相机轨迹参数化，并在视频扩散模型之上训练一个即插即用（plug-and-play）的相机姿态控制模块，基础模型的其余模块保持不变。此外，我们系统研究了不同训练数据集的影响，结果表明：相机分布多样、且外观与基础模型相近的视频，确实能提升可控性与泛化能力。实验表明，CameraCtrl 能在不同视频生成模型上实现精确的相机控制，标志着从文本与相机姿态输入出发、迈向动态且可定制视频叙事的一步。项目网站：https://hehao13.github.io/projects-CameraCtrl/ 。

---

## 1. 引言

近年来，扩散模型显著提升了从文本或其他输入生成视频的能力（Align-Your-Latents、Make-A-Video、Tune-A-Video、Imagen Video、AnimateDiff），并对数字内容设计流程产生变革性影响。在实际视频生成应用中，可控性至关重要，它允许按用户需求更好地定制结果，从而提升生成视频的质量、真实感与可用性。尽管文本与图像输入常被用于实现可控性，它们往往难以精确控制视觉内容与物体运动。为此，已有工作引入光流（optical flow；DragNUWA、MCDiff、Motion-I2V）、姿态骨架（pose skeleton；Follow-Your-Pose、DreamBooth）以及其他多模态信号（VideoComposer、MM-Diffusion 等），以更准确地引导视频生成。

![图 1. CameraCtrl 示意。前两行展示其可控制通用 T2V（AnimateDiff）与个性化 T2V（Realistic Vision）生成中的相机轨迹。第三行展示其可与 I2V 扩散模型（如 Stable Video Diffusion）配合使用；条件图像为第 3 行的首帧。CameraCtrl 还可与其他视觉控制器协作，例如 SparseCtrl 的 RGB encoder，在图像与文本条件下生成视频并管理相机运动。](../../../arxiv/geometry_control/cameractrl/extracted/images/teaser.png)

然而，现有模型仍缺乏在视频生成中调整或模拟相机视点（camera viewpoints）的精确控制。生成过程中控制相机视点，对虚拟现实、增强现实与游戏开发等应用至关重要。此外，娴熟的相机运动管理能帮助创作者强调情绪、突出人物关系并引导观众注意，这在电影与广告产业中具有显著价值。近期已有工作尝试在视频生成中引入相机控制。例如，AnimateDiff 在其 motion module 之上接入 MotionLoRA，从而支持若干特定类型的相机运动；但它难以泛化到用户自定义的相机轨迹。MotionCtrl 通过对视频扩散模型条件化一组相机姿态参数，提供更灵活的相机控制，但它仅依赖相机参数的数值、缺少相机姿态的几何线索，不足以保证精确控制。此外，MotionCtrl 也难以将相机控制泛化到其他个性化视频生成模型。

因此我们提出 **CameraCtrl**：学习一个精确的即插即用相机姿态控制模块，以在视频生成中控制相机视点。考虑到将相机控制模块无缝接入现有视频扩散模型并不容易，我们研究如何有效表示并注入相机姿态。具体地，我们采用 Plücker embeddings 作为相机姿态条件的主要形式。这一选择源于其对视频帧中每个像素编码几何解释，从而提供对相机姿态信息的全面描述。为确保训练后 CameraCtrl 的适用性与泛化性，我们引入仅以 Plücker embedding 为输入的相机控制模块，因而与训练数据的外观无关。为评估有效训练策略，我们系统研究了从写实到合成的各类训练数据的影响。实验表明，外观与原始基础模型相近、且相机姿态分布多样的数据（例如 RealEstate10K）在泛化性与可控性之间取得最佳折中。

我们首先在 AnimateDiff 上实现 CameraCtrl，使 text-to-video（T2V）生成能在多种个性化 T2V 模型上实现精确相机控制（见图 1 第 1–2 行）。我们还将 CameraCtrl 与 Stable Video Diffusion 结合，在 image-to-video（I2V）设定下实现相机控制（见图 1 第 3 行）。此外，如图 1 最后一行所示，CameraCtrl 也与其他即插即用模块兼容，例如 SparseCtrl，从而在文本与结构信息（如图像）条件下控制视频视点。

主要贡献有三：

1. 提出 CameraCtrl，为视频扩散模型赋予灵活、精确的相机视点可控性。
2. 即插即用的相机控制模块可适配多种视频生成模型，产出视觉上吸引人的相机控制效果。
3. 对用于训练相机控制模块的数据集给出全面分析，希望对后续研究有所助益。

---

## 2. 相关工作

**视频生成。** 受益于训练稳定性与成熟的开源社区，近期视频生成尝试主要采用扩散模型（DDPM、DDIM、DiT）。许多近期视频扩散模型是 T2V 模型（DreamPose、MM-Diffusion、I2VGen-XL、Latent Video Diffusion、VideoCrafter1、CogVideo、CogVideoX）。另一些工作将引导信号从文本换为图像，聚焦 I2V 设定（MCDiff、SEINE、Structure and Content-Guided Video Synthesis 等）。作为开创性工作，Video Diffusion Model 将 2D 图像扩散架构扩展到视频数据，并从零联合训练图像与视频。为利用强大的预训练图像生成器（如 Stable Diffusion），后续工作通过在预训练 2D 层之间插入时间层（temporal layers）来膨胀 2D 架构，并在大规模视频数据集（WebVid-10M）上微调。其中，Align-Your-Latents 通过对齐独立采样的噪声图，高效地把 text-to-image（T2I）模型变成视频生成器。Stable Video Diffusion（SVD）通过更精细的训练步骤与数据策展扩展 Align-Your-Latents。AnimateDiff 利用可插拔 motion module，在个性化图像骨干（DreamBooth）上实现高质量动画。为增强时间一致性，Lumiere 替换常用的时间超分辨率模块，直接生成全帧率视频。其他重要尝试包括采用可扩展 transformer 骨干（Latte）、在时空压缩潜空间中操作（例如 W.A.L.T. 与 Sora），以及用离散 token 与语言模型做视频生成（VideoPoet）。更全面的综述见扩散模型综述（Po et al., 2023）。

**可控视频生成。** 仅用文本或图像条件往往语义含糊，对视频扩散模型的控制偏弱。为提供更强引导，部分工作采用深度/骨架序列等信号，精确控制生成视频中的场景/人体运动（SparseCtrl、Control-A-Video、ControlVideo、Text2Video-Zero、Animate Anyone、MagicAnimate）。另有方法用草图图像作为控制信号，有助于高视频质量或准确的时间关系建模。相比之下，本文关注视频生成过程中的相机控制。AnimateDiff 采用高效 LoRA 微调，得到针对不同相机运动类型的模型权重。Direct-a-Video 提出 camera embedder 来控制生成视频的相机姿态，但仅条件化三个相机参数，控制能力限于 pan left 等最基本类型。MotionCtrl 以更多相机参数为输入来控制相机视点。然而，仅依赖相机参数的数值会限制控制精度；且需要微调视频扩散模型的部分参数，可能损害其在不同视频域上的泛化能力。本研究旨在精确控制视频生成过程中的相机姿态，并期望相应模型可用于多种视频生成模型。

---

## 3. CameraCtrl

将精确相机控制引入现有视频生成方法具有挑战，但对达成期望结果具有显著价值。为此，我们从三个关键问题出发：

1. 如何有效表示相机条件，以反映其在 3D 空间中的几何运动？
2. 如何将相机条件无缝注入现有视频生成器，而不损害帧质量与时间一致性？
3. 应使用何种训练数据，以确保模型得到恰当训练？

本节组织如下：第 3.1 节简要回顾视频生成模型背景；第 3.2 节介绍 CameraCtrl 所用的相机表示；第 3.3 节给出将相机表示注入视频扩散模型的相机模型 $\Phi_c$；数据选择过程在第 3.4 节讨论。

### 3.1 视频生成预备知识

**视频扩散模型。** T2V 扩散模型近年取得显著进展。一些方法（Make-A-Video、Video Diffusion Models）从零训练视频生成器；另一些（AnimateDiff、Align-Your-Latents）以强大的 T2I 扩散模型为预训练模型，并在其上训练若干时间块。还有方法用图像与视频联合训练视频生成器（CogVideoX）。尽管训练配方不同，这些模型往往沿用图像生成的原始形式化。具体地，一组 $N$ 张图像（或其潜特征）$z_0^{1:N}$ 先在 $T$ 步内逐步加入噪声 $\epsilon$，直至接近正态分布。给定第 $t$ 步的含噪输入 $z_t^{1:N}$，神经网络 $\hat{\epsilon}_{\theta}$ 被训练以预测所加噪声。训练时网络最小化其预测与真实噪声尺度之间的均方误差（MSE）；目标函数为：

$$
\mathcal{L}(\theta) = \mathbb{E}_{z_{0}^{1:N}, \epsilon, c_t, t}\bigl[\|\epsilon - \hat{\epsilon}_{\theta}(z_{t}^{1:N}, c_t, t)\|_2^2\bigr],
\tag{1}
$$

其中 $c_t$ 表示相应条件信号（如文本提示）的 embeddings。

**可控视频生成。** 在文本条件之外，可控性已有进一步进展。通过将额外结构控制信号 $s_t$（例如 depth maps 与 canny maps）纳入过程，图像与视频生成的可控性均可增强。通常，这些控制信号先送入额外 encoder $\Phi_s$，再通过多种运算注入生成器（ControlNet、T2I-Adapter、IP-Adapter）。因此，训练该 encoder 的目标可写为：

$$
\mathcal{L}(\theta) = \mathbb{E}_{z_{0}^{1:N}, \epsilon, c_t, s_t, t}\bigl[\|\epsilon - \hat{\epsilon}_{\theta}(z_{t}^{1:N}, c_t, \Phi_s(s_t), t)\|_2^2\bigr].
\tag{2}
$$

本文将相机姿态作为视频扩散模型的额外控制信号，并严格遵循式 (2) 的目标来训练相机 encoder $\Phi_c$。

### 3.2 相机姿态表示

在进入相机控制模块的架构与训练之前，我们先研究何种相机表示能精确反映 3D 空间中的相机运动。

**相机表示。** 通常，相机姿态指内参与外参，分别记为 $\mathbf{K} \in \mathbb{R}^{3 \times 3}$ 与 $\mathbf{E} = [\mathbf{R}; \mathbf{t}] \in \mathbb{R}^{3 \times 4}$，其中 $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ 是外参的旋转部分，$\mathbf{t} \in \mathbb{R}^{3 \times 1}$ 是平移部分。

要对视频生成器条件化相机姿态，一种直接选择是把相机参数的原始数值送入生成器。然而，这一选择可能不利于精确相机控制，原因包括：(1) 旋转矩阵 $\mathbf{R}$ 受正交性约束，而平移向量 $\mathbf{t}$ 的幅度通常无约束，导致相机控制模型学习过程中的失配；(2) 直接使用原始相机参数使模型难以将这些数值与图像像素关联，限制对视觉细节的精确控制。因此我们选择 Plücker embeddings 作为相机姿态表示。具体地，对图像坐标空间中每个像素 $(u, v)$，其 Plücker embedding 为 $\mathbf{p}_{u,v} = (\mathbf{o} \times \mathbf{d}_{u,v}, \mathbf{d}_{u,v}) \in \mathbb{R}^{6}$，其中 $\mathbf{o} \in \mathbb{R}^{3}$ 是世界坐标系中的相机中心，$\mathbf{d}_{u,v} \in \mathbb{R}^{3}$ 是世界坐标系中从相机中心指向像素 $(u, v)$ 的方向向量，计算为：

$$
\mathbf{d}_{u,v} = \mathbf{R}\mathbf{K}^{-1}[u, v, 1]^{\mathrm{T}} + \mathbf{t}.
\tag{3}
$$

随后将其归一化为单位长度。对视频序列中第 $i$ 帧，其 Plücker embedding 可表示为 $\mathbf{P}_i \in \mathbb{R}^{6 \times h \times w}$，其中 $h$ 与 $w$ 为帧的高与宽。

注意，式 (3) 表示相机投影的逆过程：相机投影通过矩阵 $\mathbf{E}$ 与 $\mathbf{K}$ 把 3D 世界坐标中的点映射到像素坐标系。因此，相较外参与内参矩阵的数值，Plücker embedding 对视频帧的每个像素具有更多几何解释，从而能为基础视频生成器提供更丰富的相机姿态描述。由此，它能更好地借用基础视频生成器的时间一致性能力，生成具有指定相机轨迹的视频片段。此外，Plücker embedding 各项的取值范围更均匀，有利于数据驱动模型的学习。不同相机表示的示意见附录图 6：相机矩阵与 Euler angles 均为数值，而 Plücker embedding 是逐像素的空间 embedding。得到第 $i$ 帧相机姿态的 Plücker embedding $\mathbf{P}_i$ 后，我们将整段视频的相机轨迹表示为 Plücker embedding 序列 $\mathbf{P} \in \mathbb{R}^{n \times 6 \times h \times w}$，其中 $n$ 为视频片段的总帧数。

![图 2. CameraCtrl 框架。(a) 给定预训练视频扩散模型（例如 AnimateDiff 与 SVD），CameraCtrl 在其上训练相机 encoder：以 Plücker embedding 为输入，输出多尺度相机表示。这些特征再按相应尺度融入 U-Net 的 temporal attention 层，以控制视频生成过程。(b) 相机注入过程细节。相机特征 $c_t$ 与潜特征 $z_t$ 先做逐元素相加；再用可学习线性层进一步融合两种表示，并送入每个 temporal block 的第一个 temporal attention 层。](../../../arxiv/geometry_control/cameractrl/extracted/images/fig2.png)

### 3.3 将相机可控性注入视频生成器

由于相机轨迹由 Plücker embedding 序列参数化为逐像素空间射线图（ray map），我们遵循文献（ControlNet、T2I-Adapter）：先用 encoder 提取 Plücker embedding 序列的特征，再将相机特征融入视频生成器。

**相机 encoder。** 给定具体的相机 encoder，可以像 ControlNet 那样同时将 Plücker embedding 序列与对应图像特征作为输入；也可以像 T2I-Adapter 那样仅将相机特征送入相机 encoder。经验分析表明，第一种做法会因使用输入图像的潜表示而从训练数据泄漏外观信息，使模型依赖训练数据固有的外观偏差，从而限制其在不同域上泛化相机姿态控制的能力。因此，如图 2(a) 所示，我们的相机 encoder $\Phi_c$ 仅以 Plücker embedding 为输入，并输出多尺度特征。基于 T2I-Adapter 所用 encoder，我们引入专为视频设计的相机 encoder。该相机 encoder 在每个卷积块之后加入 temporal attention 模块，以捕捉整段视频中相机姿态之间的时间关系。相机 encoder 的详细架构见附录 D.1。

**相机融合。** 得到多尺度相机特征后，我们希望将它们无缝整合进视频扩散模型的 U-Net 架构。因此我们进一步考察 U-Net 中不同类型的层，以确定应在何处纳入相机信息。回顾 U-Net 同时包含 spatial attention 与 temporal attention。我们将相机特征注入 temporal attention blocks。这一决策源于 temporal attention 层捕捉时间关系的能力，与相机轨迹固有的序列性与因果性相一致；而 spatial attention 层通常刻画单帧。该相机特征融合过程见图 2(b)。图像潜特征 $z_t$ 与相机姿态特征 $c_t$ 通过逐像素相加直接组合。随后，整合后的特征经过线性层，其输出直接送入每个 temporal attention 模块中固定的第一个 temporal attention 层。

### 3.4 以数据驱动方式学习相机分布

在视频生成器上训练上述相机 encoder 与融合线性层，通常需要大量带相机姿态标注的视频。对真实视频，可通过 structure-from-motion（SfM），例如 COLMAP，获得相机轨迹；也可从 Blender 等渲染引擎收集带真值相机姿态的视频。因此我们研究不同训练数据对相机控制生成器的影响。

**数据集选择。** 我们希望选择外观与基础视频扩散模型训练数据尽可能接近、且相机姿态分布尽可能宽的数据集。候选为三个数据集：Objaverse、MVImageNet 与 RealEstate10K。样本见附录图 5。

计算机生成影像的数据集（如 Objaverse）确实具有多样的相机分布，因为渲染过程中可以控制相机参数。然而，这些数据集在外观上往往与真实世界数据集存在分布差距，例如用于训练基础视频扩散模型的 WebVid-10M。处理真实世界数据集（如 MVImageNet 与 RealEstate10K）时，相机参数分布往往不够宽。此时需要在单条相机轨迹的复杂度与多条相机轨迹之间的多样性之间取得平衡：前者保证模型在每次训练中学会控制复杂轨迹，后者保证模型不过拟合到某些固定模式。实际上，MVImageNet 中单条轨迹的复杂度可能略高于 RealEstate10K，但 MVImageNet 的轨迹通常限于水平旋转。相比之下，RealEstate10K 展示出多种多样的相机轨迹。考虑到我们的目标是将模型应用于广泛的自定义轨迹，我们最终选择 RealEstate10K 作为训练数据集。此外，还有一些与 RealEstate10K 特征相似的数据集，例如 ACID 与 MannequinChallenge，但其数据量远小于 RealEstate10K。我们尝试将它们与 RealEstate10K 联合训练 CameraCtrl，但未发现收益。

**度量相机可控性。** 为监控相机 encoder 的训练过程，我们设计两个指标，通过量化输入相机条件与生成视频相机轨迹之间的误差来衡量相机控制质量。具体地，我们用 COLMAP 提取生成视频的相机姿态序列，包括旋转矩阵 $\mathbf{R}_{\mathrm{gen}} \in \mathbb{R}^{n \times 3 \times 3}$ 与平移向量 $\mathbf{T}_{\mathrm{gen}} \in \mathbb{R}^{n \times 3 \times 1}$。由于旋转角与平移尺度是两类不同的数学量，我们分别度量角度误差与平移误差，并称之为 RotErr 与 TransErr。受 SO(3) 旋转距离工作启发，RotErr 通过比较真值旋转矩阵 $\mathbf{R}_{\mathrm{gt}}$ 与 $\mathbf{R}_{\mathrm{gen}}$ 计算：

$$
\mathrm{RotErr} = \sum_{i=1}^{n} \arccos\frac{\mathrm{tr}(\mathbf{R}_{\mathrm{gen}}^{i} \mathbf{R}_{\mathrm{gt}}^{i\mathrm{T}}) - 1}{2},
\tag{4}
$$

其中 $\mathbf{R}_{\mathrm{gt}}^{i}$ 与 $\mathbf{R}_{\mathrm{gen}}^{i}$ 分别表示第 $i$ 帧的真值与生成旋转矩阵，$\mathrm{tr}$ 为矩阵的迹。为量化平移误差，我们使用真值平移向量 $\mathbf{T}_{\mathrm{gt}}$ 与 $\mathbf{T}_{\mathrm{gen}}$ 之间的 $L2$ 距离：

$$
\mathrm{TransErr} = \sum_{i=1}^{n} \|\mathbf{T}_{\mathrm{gt}}^{i} - \mathbf{T}_{\mathrm{gen}}^{i}\|_{2}^{2},
\tag{5}
$$

其中 $\mathbf{T}_{\mathrm{gt}}^{i}$ 与 $\mathbf{T}_{\mathrm{gen}}^{i}$ 为第 $i$ 帧的真值与生成平移向量。关于 RotErr 与 TransErr 的更多讨论见附录 D.5。

---

## 4. 实验

本节将 CameraCtrl 与其他方法比较，并展示其在不同视频生成设定中的应用。第 4.1 节给出实现细节；第 4.2 节将 CameraCtrl 与基线 AnimateDiff 与 MotionCtrl 比较；第 4.3 节给出全面消融；第 4.4 节展示 CameraCtrl 的多种应用。

### 4.1 实现细节

**基础视频扩散模型。** 在 T2V 设定中，以 AnimateDiff V3 为基础模型。AnimateDiff 可与不同风格的多种 T2I LoRAs 或基础模型集成，这一特性有助于评估 CameraCtrl 的泛化能力。在 I2V 设定中实现 CameraCtrl 时，基础模型为 SVD。

**训练。** 使用 AdamW 优化器，恒定学习率为 $1 \times 10^{-4}$（T2V）或 $3 \times 10^{-5}$（I2V）。如第 3.4 节所述，我们选择 RealEstate10K，约 65K 个视频片段用于训练。相机 encoder 与用于相机特征注入的线性层一同训练，batch size 为 32，共 50K 步。更多细节见附录 D.2。

**评估指标。** 为确保相机模型不损害原始视频扩散模型的外观质量，我们使用 Fréchet Video Distance（FVD）、CLIPSIM 与 Frame Consistency（FC）评估视频外观质量。此外，受 VBench 中 Dynamic Degree 指标启发，我们提出 Object Dynamic Degree（ODD；细节见附录 D.4）以评估物体运动程度。相机控制质量则用第 3.4 节引入的 RotErr 与 TransErr 评估。对 FVD、CLIPSim、FC、ODD 所用的参考视频和/或文本标题，我们从 WebVid-10M 随机采样 1,000 个视频。对 RotErr 与 TransErr，我们从 RealEstate10K 测试集随机选取 1,000 个视频及其对应相机姿态。

**表 1. 定量比较。** MotionCtrl$_{\mathrm{VC}}$ 与 MotionCtrl$_{\mathrm{SVD}}$ 分别表示以 VideoCrafter 与 SVD 为基础模型的 MotionCtrl。相应的，CameraCtrl$_{\mathrm{AD}}$ 与 CameraCtrl$_{\mathrm{SVD}}$ 分别表示以 AnimateDiff 与 SVD 为基础、接入 CameraCtrl 的模型。

| Method | FVD  | CLIPSIM $\uparrow$ | FC $\uparrow$ | ODD $\uparrow$ | TransErr  | RotErr  | User Preference Rate $\uparrow$ (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AnimateDiff | **1022.4** | 0.298 | 0.930 | **56.4** | Incapable | Incapable | 19.4 |
| MotionCtrl$_{\mathrm{VC}}$ | 1123.2 | 0.286 | 0.922 | 42.3 | 14.02 | 1.58 | 37.0 |
| **CameraCtrl$_{\mathrm{AD}}$** | 1088.9 | **0.301** | **0.941** | 49.8 | **12.98** | **1.29** | **43.6** |
| SVD | 371.2 | **0.312** | 0.957 | **47.5** | Incapable | Incapable | Incapable |
| MotionCtrl$_{\mathrm{SVD}}$ | 386.2 | 0.303 | 0.953 | 41.8 | 10.21 | 1.41 | 26.9 |
| **CameraCtrl$_{\mathrm{SVD}}$** | **360.3** | 0.298 | **0.960** | 46.5 | **9.02** | **1.18** | **73.1** |

### 4.2 与其他方法的比较

**定量比较。** 为证明 CameraCtrl 的有效性，我们将其与两种替代方法比较：带 MotionLoRA 的 AnimateDiff，以及 MotionCtrl。值得注意的是，AnimateDiff 仅支持八种基本相机运动，且我们没有这些运动的真值相机轨迹，因此无法为 AnimateDiff 计算 RotErr 与 TransErr。取而代之，我们进行用户研究（细节见附录 E），评估用户对不同模型相机控制能力的偏好。此外，我们在 T2V 与 I2V 两种设定下将 CameraCtrl 与 MotionCtrl 比较。定量结果见表 1。中间块为 T2V 设定，底部块为 I2V 设定。相较带 MotionLoRA 的 AnimateDiff 与 MotionCtrl，本方法在相机控制精度（TransErr、RotErr 与用户偏好）上更优。TransErr 与 RotErr 的下界见附录 F.3。此外，相较基础模型（AnimateDiff 与 SVD），CameraCtrl 并未牺牲生成视频的视觉质量与动态程度，这由 FVD、CLIPSIM、FC 与 ODD 上更好或相当的指标所证明。

**定性比较。** 我们还在图 3 中给出 CameraCtrl 与 MotionCtrl 在 T2V 与 I2V 设定下的定性比较。从前两行比较可见，MotionCtrl 未能跟随相机条件：它表现为场景旋转，而非相机运动。相比之下，CameraCtrl 能区分相机运动与场景运动，并严格跟随相机轨迹条件。此外，MotionCtrl 对微小相机运动不敏感。如第三行所示，MotionCtrl 的结果仅表现前向相机运动，忽略了条件轨迹中向左的小幅运动。相比之下，最后一行的 CameraCtrl 结果准确跟随前向与向左的相机运动。更多定性比较见附录 G。

![图 3. CameraCtrl 与 MotionCtrl 的定性比较。前两行为 T2V 设定，分别表示以 VideoCrafter 为基础的 MotionCtrl，以及以 AnimateDiff V3 为基础的 CameraCtrl。后两行是以 SVD 为基础、以图像为条件信号的 MotionCtrl 与 CameraCtrl。条件图像为每行的首帧。](../../../arxiv/geometry_control/cameractrl/extracted/images/figure3.png)

### 4.3 消融实验

我们将相机控制问题拆成三项挑战：第 3.2 节的相机表示选择、第 3.3 节的相机控制模型架构，以及第 3.4 节相机控制模型的学习过程。本节用 FVD、TransErr 与 RotErr 作为主要指标，全面消融各项设计选择。本节所有 CameraCtrl 模型均在 AnimateDiff V3 上实现。除非另有说明，我们使用与第 4.2 节相同的 RealEstate10K 中 1,000 个视频片段。

**Plücker embeddings 精确表示相机。** Plücker embeddings 天然是空间上、逐像素的图，不同位置取值不同。作为替代，我们可以直接使用内参矩阵 $\mathbf{K}$ 与外参矩阵 $\mathbf{E}$ 的数值，或把 $\mathbf{E}$ 的旋转矩阵转为 Euler angles，再在空间上重复它们，形成各位置内容相同的逐像素图。另一种做法是把射线方向（随像素变化）与重复的相机原点（空间位置上恒定）组合成空间像素图。实验结果见表 2(a)——以 Plücker embeddings 作为相机表示得到最佳相机控制结果。这源于 Plücker embedding 能为每个像素提供几何解释。相比之下，仅依赖数值可能导致数值失配，损害相机模型的学习效率。对射线方向加相机原点的表示，虽能提供准确的相机原点信息，但重复的相机原点参数增加冗余，可能使特征错位，妨碍模型理解相机运动。

**含噪潜变量作为输入会限制泛化。** 在消融相机 encoder 架构时，我们区分 ControlNet（输入为图像特征与 Plücker embedding 序列之和）与 T2I-Adapter（仅以 Plücker embedding 序列为输入）。这一区分至关重要：如 SparseCtrl 所述，使用含噪潜变量与外观泄漏相关，实际上限制了相机控制质量在不同域之间的泛化能力。此外，为增强帧间相机一致性，我们也考虑为每个 encoder 加入 temporal attention block。因此，在选择相机 encoder 架构时，实验覆盖四种配置：ControlNet、T2I-Adapter，以及二者的 temporal attention 增强变体。消融结果见表 2(b)。以 ControlNet 作为相机 encoder 时，外观质量欠佳，见表中前两行的 FVD。对使用 T2I-Adapter 的模型，可观察到带额外 temporal attention 模块的模型具有更好的相机控制能力。因此，我们选择带 temporal attention 模块的 T2I-Adapter encoder 作为相机 encoder。

**表 2. 关于相机表示、条件注入以及不同数据集影响的消融。**

**表 2(a). 如何表示相机参数。**

| Representation type | FVD  | TransErr  | RotErr  |
| --- | ---: | ---: | ---: |
| Raw Values | 230.1 | 13.88 | 1.51 |
| Euler angles | **221.2** | 13.71 | 1.43 |
| Direction + Origin | 232.3 | 13.21 | 1.57 |
| **Plücker embedding** | 222.1 | **12.98** | **1.29** |

**表 2(b). 相机 encoder 架构。**

| Encoder architecture type | FVD  | TransErr  | RotErr  |
| --- | ---: | ---: | ---: |
| ControlNet | 295.8 | 13.51 | 1.42 |
| ControlNet + Temporal | 283.4 | 13.13 | 1.33 |
| T2I Adaptor | 223.4 | 13.27 | 1.38 |
| T2I Adaptor + Temporal | **222.1** | **12.98** | **1.29** |

**表 2(c). 相机表示注入何处。**

| Attention | FVD  | TransErr  | RotErr  |
| --- | ---: | ---: | ---: |
| Spatial Self | 241.2 | 14.72 | 1.42 |
| Spatial Cross | 237.5 | 14.31 | 1.51 |
| Spatial Self + Cross | 240.1 | 14.52 | 1.60 |
| **Temporal** | **222.1** | **12.98** | **1.29** |

**表 2(d). 数据集的影响。**

| Datasets | FVD  | TransErr  | RotErr  |
| --- | ---: | ---: | ---: |
| Objaverse | 1435.4 | Incapable | Incapable |
| MVImageNet | 1143.5 | 13.87 | 1.52 |
| RealEstate10K + ACID | 1102.4 | 13.48 | 1.41 |
| RealEstate10K | **1088.9** | **12.99** | **1.39** |

**将相机条件注入 temporal attention。** 接下来，我们研究提取到的相机特征应插入预训练 U-Net 架构的何处。为此，我们进行四组实验，分别将特征插入 U-Net 的 spatial self attention、spatial cross attention、二者同时，以及 temporal attention 层。结果见表 2(c)，表明将相机特征插入 temporal attention 层效果更好。这一改进可归因于：相机运动通常引起跨帧的全局视点变化。将相机姿态与 U-Net 的 temporal blocks 整合，与这一动态特性相呼应。

**外观分布相近且相机多样的视频有助于可控性。** 为检验第 3.4 节关于数据集选择的论点，我们用不同数据集训练 CameraCtrl。Objaverse 的相机姿态分布最宽，但外观与 WebVid-10M 差异显著。对真实世界数据集，相较 MVImageNet，RealEstate10K 拥有更多样的相机轨迹。我们用多样数据源评估这些模型：WebVid-10M 用于 FVD，MannequinChallenge 的相机轨迹用于 TransErr 与 RotErr。如表 2(d) 所示，相较 RealEstate10K，MVImageNet 上的 FVD 分数与相机误差均显著更高。对 Objaverse，COLMAP 难以提取足够数量的相机姿态以得到有意义的 TransErr 与 RotErr。一种可能原因是数据集外观差异使模型难以有效区分相机姿态与外观，从而使 COLMAP 难以估计相机姿态。我们还用 RealEstate10K 与相似数据集 ACID 联合训练 CameraCtrl，但相机控制能力没有提升。该结果表明，要进一步改进 CameraCtrl，需要相机分布更大的数据集。

### 4.4 CameraCtrl 的应用

**将 CameraCtrl 应用于不同视频生成器。** 如第 3.3 节所述，我们的相机控制模型仅以 Plücker embeddings 为输入，因而独立于训练数据集的外观。此外，如第 3.4 节所述，我们选择外观与基础视频生成器训练数据高度相近的数据集。受益于这些设计，CameraCtrl 能专注于学习与相机控制相关的信息，从而可应用于多种视频生成器。我们在图 4、附录 H.1 与附录 H.2 中展示这一点。在 T2V 设定中，我们基于 AnimateDiff 实现 CameraCtrl。第一行采用 vanilla AnimateDiff 模型，描绘自然场景。第二、三行展示嵌入其他个性化图像生成器（Realistic Vision 与 ToonYou）的 AnimateDiff 结果。第二行代表偏离典型现实的视频风格：赛博朋克城市的建筑。第三行表现卡通角色视频。此外，我们在 SVD 上实现 CameraCtrl，并在 I2V 设定下采样一段视频，见最后一行。在这些不同的视频生成类型中，CameraCtrl 始终对相机轨迹表现出有效控制，展示其广泛适用性，以及通过动态相机轨迹控制增强视频叙事的能力。

![图 4. CameraCtrl 的应用。第一行是基础 AnimateDiff 生成的视频。随后两行展示两个个性化 T2V 生成器 RealisticVision 与 ToonYou 的结果。第四行展示 CameraCtrl 与另一视频控制方法 SparseCtrl 集成后生成的视频。最后一行由 I2V 生成器 SVD 产生，以该行首帧为条件。](../../../arxiv/geometry_control/cameractrl/extracted/images/figure4.png)

**将 CameraCtrl 与其他视频控制方法集成。** 得益于本方法的即插即用性质，它不仅可用于不同基础视频生成器的生成过程，还可与其他视频生成控制技术一起产生视频。例如，我们使用 SparseCtrl：一种通过操控少量稀疏帧来控制整体视频生成的近期方法。该控制可基于 RGB 图像、草图或深度。这里我们采用 SparseCtrl 的 RGB encoder 与 sketch encoder，结果分别见图 1 最后一行与图 4 第四行。如这两段视频所示，该方法使生成视频中的场景与物体与参考帧高度一致。此外，生成视频的相机运动与所给相机轨迹明显高度对齐。CameraCtrl 与 SparseCtrl 的成功集成进一步证明 CameraCtrl 的泛化能力，并拓宽其应用场景。更多可视化结果见附录 H.3。

---

## 5. 讨论

本文提出 CameraCtrl，针对现有模型在视频生成中缺乏精确相机控制的局限。通过学习即插即用的相机控制模块，CameraCtrl 实现对相机视点的准确控制。我们采用 Plücker embeddings 作为相机参数的主要表示，通过编码几何解释提供对相机姿态信息的全面描述。通过对训练数据的系统研究，发现使用外观与基础模型相近、且相机姿态分布多样的数据（如 RealEstate10K），能在泛化性与可控性之间取得最佳折中。实验结果证明了 CameraCtrl 的有效性。

**伦理声明。** CameraCtrl 通过提供对相机视点的精确控制来增强视频生成技术，显著提升许多应用的真实感与交互性。另一方面，CameraCtrl 也可能引发伦理关切，尤其是隐私以及制造误导性内容的潜在可能。亟需伦理监督与更先进的 deepfake 检测器来管理这些风险，并确保 CameraCtrl 被恰当使用。

**可复现性声明。** 我们在正文第 4.1 节与附录 D.2 中给出训练方法的详细实现，并在附录 D.1 中给出相机 encoder 的模型架构。

---

## 致谢

本项目部分受国家重点研发计划项目 2022ZD0161100 以及 NSFC-RGC 项目 N_CUHK498/24 资助。Hongsheng Li 为 InnoHK 下 CPII 的 PI。

---

## 附录

### 附录 A. 补充材料说明

本补充材料提供 CameraCtrl 与其他方法的讨论、数据集选择的更多讨论、实现细节、用户研究细节、额外消融实验、更多定性比较，以及 CameraCtrl 的更多可视化结果。

在所有可视化结果中，每行第一张图表示该视频的相机轨迹。图上每个小四面体表示一帧相机的位置与朝向：顶点代表相机位置，底面代表相机成像平面。红色箭头指示相机**位置**的运动，但**不**刻画相机旋转。相机旋转可通过四面体的朝向观察。**为更清楚地理解相机控制效果，强烈建议读者观看补充文件中提供的视频。**

本补充材料组织如下：附录 B 给出 CameraCtrl 与同期工作的若干讨论；附录 C 给出数据集选择过程的更多讨论；附录 D 给出更多实现细节；用户研究细节见附录 E；附录 F 给出关于模型架构、相机表示以及 RotErr 与 TransErr 下界的额外实验结果；随后在附录 G 给出 CameraCtrl 与 AnimateDiff、MotionCtrl 的更多定性比较；之后附录 H 展示更多可视化结果；最后附录 I 给出若干失败案例。

### 附录 B. 与同期相机控制工作的讨论

近期工作从不同侧面探索视频生成中的相机控制。VD3D 将相机控制接入基于 DiT 的模型，并在 spatiotemporal transformers 中使用新颖的相机表示模块。CamCo 利用对极约束（epipolar constraints）在 image-to-video 生成中获得 3D 一致性。CVD 使用相机控制方法并将其扩展，以支持具有跨视角一致性的多视角视频生成。Recapture 实现 video-to-video 相机控制，有效修改已有内容中的视点；但它限于较简单场景，在复杂或动态环境中表现吃力。Cavia 通过对多样数据集训练来增强多视角生成，改善跨视角一致性。另有工作在基于 DiT（Open-Sora）的模型中使用类似 classifier-free guidance 的机制来提升相机控制精度。尽管已有大量工作处理视频生成过程中的相机控制，据我们所知，CameraCtrl 属于在视频生成模型中实现精确相机控制的较早方法之一。它为视频生成以及 3D、4D 内容生成等相关领域的后续进展提供了有价值的洞见与坚实基础。

### 附录 C. 数据集选择的更多讨论

在为相机控制模型选择训练数据集时，我们首先选定三个候选：Objaverse、MVImageNet 与 RealEstate10K。

对 Objaverse，其图像由 Blender 等软件渲染，因而能实现高度复杂的相机姿态。然而，如图 5 第 1 至第 3 行所示，其内容主要是白背景上的物体。相比之下，许多视频扩散模型的训练数据（如 WebVid-10M）同时包含物体与场景，且背景更复杂。这一显著外观差异会削弱模型专注于学习相机控制的能力。在初步试验中，我们尝试用 Objaverse 训练 CameraCtrl：所得模型能很好地控制 Objaverse 风格视频（白背景上的单物体）中的相机轨迹；但在其他域中，相机控制模型难以在视频生成过程中很好地泛化对相机视点的控制。

对 MVImageNet，它具有一定背景以及复杂的单条相机轨迹。然而，如图 5 第 4 至第 6 行所示，MVImageNet 中大多数相机轨迹是水平旋转。因此其相机轨迹缺乏多样性，可能导致模型收敛到固定模式。

对 RealEstate10K，如图 5 第 7 至第 9 行所示，它同时包含室内外场景与物体。此外，RealEstate10K 中每条相机轨迹都较复杂，不同轨迹之间也存在相当大的多样性。因此我们选择 RealEstate10K 来训练相机控制模型。还有其他数据集具有与 RealEstate10K 相似的相机轨迹，例如 ACID 与 MannequinChallenge，但样本更少。我们尝试用 RealEstate10K 与 ACID 训练 CameraCtrl，但未发现相机控制精度提升，见表 2(d)。该结果表明，当前相机控制精度的瓶颈可能在于相机姿态分布的复杂度。

![图 5. 不同数据集的样本。第 1 至第 3 行为 Objaverse 样本，每张渲染图具有随机相机姿态。第 4 至第 6 行为 MVImageNet 样本。第 7 至第 9 行为 RealEstate10K 样本。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure5.png)

### 附录 D. 更多实现细节

#### D.1 相机 encoder $\Phi_c$ 架构

如第 3.3 节所述，我们采用带 temporal attention 增强的 T2I-Adapter encoder 作为相机 encoder $\Phi_c$，从 Plücker embeddings 提取相机特征。总体而言，相机 encoder 由一个 pixel unshuffle 层、一个卷积层以及 4 个 encoder scales 组成。它以一批 Plücker embedding 序列 $\mathbf{P} \in \mathbb{R}^{b \times n \times 6 \times h \times w}$ 为输入（其中 $b, n, h, w$ 分别表示 batch size、视频片段帧数、视频高与宽），并输出多尺度相机特征。各层输出特征形状见表 3。

**表 3. 相机 encoder 各层（encoder scale）的输出特征形状。** $c = 6 \times 8 \times 8 = 384$；$c_1, c_2, c_3, c_4$ 等于相应分辨率下对应 U-Net 输出特征的通道数。例如，对 Stable Diffusion 1.5 模型，$c_1, c_2, c_3, c_4$ 分别为 320、640、1280、1280。

| | 形状 |
| --- | --- |
| **input** | $b \times n \times 6 \times h \times w$ |
| Pixel unshuffle | $b \times n \times c \times \frac{h}{8} \times \frac{w}{8}$ |
| $3 \times 3$ conv layer | $b \times n \times c_1 \times \frac{h}{8} \times \frac{w}{8}$ |
| Encoder scale 1 | $b \times n \times c_1 \times \frac{h}{8} \times \frac{w}{8}$ |
| Encoder scale 2 | $b \times n \times c_2 \times \frac{h}{16} \times \frac{w}{16}$ |
| Encoder scale 3 | $b \times n \times c_3 \times \frac{h}{32} \times \frac{w}{32}$ |
| Encoder scale 4 | $b \times n \times c_4 \times \frac{h}{64} \times \frac{w}{64}$ |

此外，每个 encoder scale 由一个 downsample ResNet block（encoder scale 1 除外）与一个 ResNet block 组成，每个 block 后接一个 temporal attention block。更具体地，temporal attention block 由 temporal self-attention 层、layer normalizations 与 position-wise MLP 组成，如下：

$$
\begin{aligned}
\zeta &\leftarrow x + \mathrm{PosEmb}(x) \\
\zeta_1 &\leftarrow \mathrm{LayerNorm}(\zeta) \\
\zeta_2 &\leftarrow \mathrm{MultiHeadSelfAttention}(\zeta_1) + \zeta \\
\zeta_3 &\leftarrow \mathrm{LayerNorm}(\zeta_2) \\
x &\leftarrow \mathrm{MLP}(\zeta_3) + \zeta_2,
\end{aligned}
$$

其中 $\mathrm{PosEmb}$ 为时间位置编码。

#### D.2 训练

我们使用 LAVIS 为所用数据集（Objaverse、MVImageNet、RealEstate10K 与 ACID）的每个视频片段生成文本提示。在 text-to-video（T2V）设定中，以 AnimateDiff V3 为基础视频生成模型。为使相机控制模型更好地专注于学习相机姿态，与 AnimateDiff 类似，我们先在 RealEstate10K 的图像上训练一个 image LoRA。然后，在增强了该 LoRA 的 AnimateDiff 模型上训练相机控制模型。注意，相机控制模型训练完成后，image LoRA 可以移除。对 CameraCtrl 的每个训练样本，我们从一个视频片段中以 stride 8 采样 16 帧，再将分辨率调整为 $256 \times 384$。数据增强方面，对图像与姿态均以 50% 概率做随机水平翻转。采用 Adam 优化器训练模型，恒定学习率 $1 \times 10^{-4}$，$\beta_1=0.9$，$\beta_2=0.99$，weight decay 为 0.01。我们使用 *linear* beta 噪声日程，其中 $\beta_{\mathrm{start}}=0.00085$，$\beta_{\mathrm{end}}=0.012$，$T=1000$。使用 16 张 80G NVIDIA A100 GPU 训练 CameraCtrl，每卡 batch size 为 2，共 50K 步，约耗时 25 小时。

在 Image-to-Video（I2V）设定中，以 Stable Video Diffusion（SVD）为基础视频生成器。我们直接在 SVD 之上训练相机 encoder 与融合线性层。对每个训练样本，从一个视频片段中以 stride 8 采样 14 帧，再将分辨率调整为 $320 \times 576$。采用 Adam 优化器，恒定学习率 $3 \times 10^{-5}$，$\beta_1=0.9$，$\beta_2=0.99$，weight decay 为 0.01。遵循 SVD，我们使用 EDM 噪声调度器，并将所有超参数设为与 SVD 相同。使用 32 张 80G NVIDIA A100 GPU 训练，每卡 batch size 为 1，共 50K 步，约耗时 40 小时。

#### D.3 推理

利用 COLMAP 等 structure-from-motion 方法，结合已有视频，我们可以提取视频中的相机轨迹。提取到的相机轨迹再送入相机控制模型，以生成具有相似相机运动的视频。此外，我们也可以设计自定义相机轨迹，以产生具有期望相机运动的视频。推理时，我们对不同域的视频使用不同的 guidance scales，并对所有视频采用恒定的 25 步去噪。

#### D.4 Object Dynamic Degree（ODD）指标

我们首先使用 Grounded-SAM-2 分割视频中的主体物体。然后，遵循 VBench 中的 dynamic degree，使用 RAFT 估计光流，并仅保留属于主体物体的估计光流。再遵循 dynamic degree 指标，以这些光流为依据判断视频是否静止。最终的 object dynamic degree 分数通过测量模型所生成非静止视频的比例来计算。

#### D.5 RotErr 与 TransErr 的更多细节

用 COLMAP 提取生成视频的相机姿态时，可靠地得到相机姿态序列并不十分稳定。因此，当 COLMAP 失败时，我们手动过滤这些失败视频片段，不将其计入 RotErr 与 TransErr。此外，由于 COLMAP 具有尺度不变性，生成的相机姿态可能存在尺度问题。这些尺度问题仅对 TransErr 的计算有一定影响，不影响 RotErr。为处理该尺度问题，我们对 COLMAP 结果做了若干后处理。具体地，我们先通过将第一帧的齐次外参矩阵设为 $4 \times 4$ 单位矩阵，计算真值与生成相机姿态的相对姿态。然后，用真值相机轨迹归一化 COLMAP 结果的尺度。具体而言，我们计算生成与真值相机姿态前两帧之间的平移间隔以得到重缩放因子，再用该因子归一化其余生成相机姿态，以对齐两条相机轨迹的尺度。这一归一化有助于缓解 COLMAP 结果中的尺度问题，使评估指标更有说服力。

### 附录 E. 用户研究细节

第 3.4 节提出的相机误差 TransErr 与 RotErr 需要 COLMAP 提取生成视频的相机姿态。然而，COLMAP 无法在短视频（T2V 为 16 帧，I2V 为 14 帧）中稳定提取精确相机姿态。为从另一视角比较 CameraCtrl 与 AnimateDiff、MotionCtrl 的相机控制质量，我们进行若干用户研究。具体地，由于 AnimateDiff 在 T2V 设定中仅能生成八种基本相机运动的视频，我们用这些基本相机运动对三种方法采样，让用户观看视频并判断哪段视频更符合条件相机轨迹。然后计算每种方法的认可率，结果见表 1 中间块的 User Preference Rate 列。此外，我们从 RealEstate10K 测试集提取若干复杂相机轨迹，在 T2V 设定下条件化 MotionCtrl 与 CameraCtrl。生成视频交给用户，判断哪一段与参考视频的相机轨迹对齐更好。MotionCtrl 与 CameraCtrl 的用户偏好率分别为 27.6% 与 72.4%。

随后，在 I2V 设定中，我们用从 RealEstate10K 提取的复杂相机轨迹对 MotionCtrl 与 CameraCtrl 采样。基于这些视频进行另一项用户研究，让用户选择哪段视频的相机轨迹条件表现更好。结果见表 1 底部块的 User Preference Rate 列。这些用户研究结果进一步证明 CameraCtrl 在视频生成过程中控制相机轨迹的优越性。我们邀请 50 名用户完成全部用户研究。考虑到这些用户教育水平的差异，我们将用户研究设计得尽可能简单，以获得更可靠的结果。

### 附录 F. 额外实验

#### F.1 额外消融

**将相机特征同时注入 U-Net 的 encoder 与 decoder。** 在原始 T2I-Adapter 中，提取的控制特征仅送入 U-Net 的 encoder。本部分探索将相机特征同时注入 U-Net encoder 与 decoder 是否能带来性能提升。实验结果见表 4。TransErr 与 RotErr 的改进表明，相较仅将相机特征送入 U-Net encoder，同时注入 encoder 与 decoder 能增强相机控制精度。这一结果可归因于：与文本 embedding 类似，Plücker embedding 本身缺乏结构信息。因此，这一整合选择使 U-Net 能更有效地利用相机特征。我们最终选择将相机特征同时送入 U-Net 的 encoder 与 decoder。

**表 4. 相机特征注入位置的消融。**

| Injection Place | FVD  | TransErr  | RotErr  |
| --- | ---: | ---: | ---: |
| U-Net Encoder | **210.9** | 13.91 | 1.51 |
| **U-Net Encoder + Decoder** | 222.1 | **12.98** | **1.29** |

![图 6. 不同相机表示。左图用内参 $K_i$ 与外参矩阵 $E_i$（由旋转矩阵 $R_i$ 与平移向量 $t_i$ 组成）表示相机。中图将旋转矩阵 $R_i$ 转为 Euler angles $\alpha_i, \beta_i, \gamma_i$。右图给出 Plücker embeddings：将内参与外参矩阵转换为 Plücker embeddings，形成逐像素空间 embedding。左、中两种相机表示并非天然的逐像素相机表示。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/camera_representations.png)

![图 7. 使用不同相机表示的定性比较。第一行使用原始相机矩阵数值作为相机表示。第二行采用射线方向与相机原点作为相机表示。最后一行以 Plücker embedding 作为相机表示。所有结果使用相同的相机轨迹与文本提示。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/different_camera_representation.png)

#### F.2 不同相机表示的定性比较

这里给出使用不同相机表示的定性比较，结果见图 7。所给相机轨迹主要向前运动，末尾带有向右偏移。从图中可见，使用原始相机矩阵数值作为表示时，模型忽略了最后的向右运动。使用混合相机姿态表示时，模型在最后几帧出现突然偏移以实现向右运动。相比之下，以 Plücker embedding 作为相机表示能得到更平滑的生成视频，最后的向右运动自然、连贯。这些结果进一步证明以 Plücker embedding 作为相机表示的有效性。

#### F.3 RealEstate10K 测试集上 TransErr 与 RotErr 的下界

由于 COLMAP 并非 100% 精确，我们需要知道 TransErr 与 RotErr 指标的下界。对 RealEstate10K 测试集中采样的视频片段（每段 16 帧），我们在这些片段上运行 COLMAP 得到估计的相机姿态。用这些相机姿态与真值相机姿态计算 TransErr 与 RotErr，结果见表 5。

**表 5. RealEstate10K 测试集上 TransErr 与 RotErr 的下界。**

| | TransErr  | RotErr  |
| --- | ---: | ---: |
| Lower Bounds | 6.93 | 1.02 |

### 附录 G. 更多定性比较

本节首先在 text-to-video（T2V）设定下，用基本相机轨迹给出 CameraCtrl 与 AnimateDiff 的更多定性比较。然后，仍在 T2V 设定下，用从 RealEstate10K 测试集提取的复杂相机轨迹给出 CameraCtrl 与 MotionCtrl 的更多定性比较。最后给出 CameraCtrl 与 MotionCtrl 在 image-to-video（I2V）设定下的更多定性比较。

![图 8. AnimateDiff 与 CameraCtrl 的定性比较。第 1、3、5 行来自 AnimateDiff。CameraCtrl 的结果见第 2、4、6、7 行。第 1、2 行使用相同相机轨迹 pan down。第 3、4 行采用 pan left。第 5、6 行使用 pan down。最后一行用 pan left down 生成。第 1、2 行条件于同一文本提示，第 3 至第 7 行条件于另一文本提示。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure6.png)

![图 9. T2V 设定下 MotionCtrl 与 CameraCtrl 的定性比较。第 1、3、5、7 行由 MotionCtrl 生成，CameraCtrl 的结果见第 2、4、6、8 行。每相邻两行使用相同文本提示与相机轨迹。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure7.png)

#### G.1 T2V 设定下的定性比较

**AnimateDiff 与 CameraCtrl 的比较。** 结果见图 8。在第 1 行，我们发现 AnimateDiff 生成的视频表现为 pan up，而非给定的相机运动 pan down。相比之下，第 2 行 CameraCtrl 生成的视频跟随了期望的相机运动。因此可以认为，在某些情况下 AnimateDiff 无法区分物体运动与相机运动。第 3、5 行表明，尽管 AnimateDiff 有时能生成具有期望相机运动的视频，它无法在整段视频中保持物体一致。相比之下，CameraCtrl 能生成内容一致的视频并严格跟随条件相机轨迹，见第 4、6 行。此外，AnimateDiff 仅支持若干简单相机轨迹。对其他更复杂的相机轨迹，它甚至不支持两种基本轨迹的组合（例如 pan left 与 pan down 的组合），而 CameraCtrl 可以支持，见图 8 最后一行。

**MotionCtrl 与 CameraCtrl 的比较。** 在图 9 中，我们给出 T2V 设定下 MotionCtrl 与 CameraCtrl 的更多定性比较。对平移或旋转幅度较小的轨迹，例如第一条轨迹（第 1、2 行）中的向左相机平移，以及第二条轨迹（第 3、4 行）开头的向左旋转，MotionCtrl 并不十分敏感。它只关注主要相机运动——向前平移。相比之下，CameraCtrl 生成的视频（第 2、4 行）准确服从这些微小相机轨迹。第三（第 5、6 行）与第四（第 7、8 行）条相机轨迹同时包含相机旋转与平移。MotionCtrl 生成的视频（第 5、7 行）更关注相机旋转而忽略相机平移。相比之下，CameraCtrl 能在相机旋转与平移之间取得良好平衡并生成满意视频，见图 9 第 6、8 行。

#### G.2 I2V 设定下的定性比较

与 T2V 结果类似，MotionCtrl 在 I2V 设定中仍不能很好地处理微小相机运动。

![图 10. I2V 设定下 MotionCtrl 与 CameraCtrl 的定性比较。条件图像见每行首帧。这些图像由 SDXL 以每两行下方的文本提示为输入生成。注意，MotionCtrl 与 CameraCtrl 都仅条件于条件图像，不包含文本提示。第 1、3、5、7 行为 MotionCtrl 的结果，CameraCtrl 的结果在第 2、4、6、8 行。每相邻两行用相同条件图像与相同相机轨迹生成。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure8.png)

在图 10 中，对前两条相机轨迹的结果，相较我们的 CameraCtrl（第 2、4 行），MotionCtrl 生成的视频（第 1、3 行）忽略了轨迹最开始的微小相机旋转。对第三、四条相机轨迹，MotionCtrl 结果（第 5、7 行）的相机运动幅度明显小于 CameraCtrl 结果（第 6、8 行）。CameraCtrl 的结果能揭示轨迹 3 与 4 的实际相机运动幅度。**我们强烈建议读者观看补充文件中提供的视频，以获得更直接的理解。**

注意，在 I2V 设定中，MotionCtrl 与 CameraCtrl 都实现在同一视频扩散模型 SVD 上，从而排除了不同基础视频生成器带来的影响。因此，生成视频中更好的相机视点控制得益于 CameraCtrl 更好的设计选择。

### 附录 H. 更多可视化结果

本节给出 CameraCtrl 的额外可视化结果。具体地，附录 H.1 给出在 T2V 设定下将 CameraCtrl 与 AnimateDiff 集成后生成的多域视频。附录 H.2 展示以 Stable Video Diffusion（SVD）为基础视频生成器时，CameraCtrl 在 I2V 设定下生成的视频。随后，附录 H.3 给出将 CameraCtrl 与另一视频控制方法 SparseCtrl 结合的视频结果。最后，附录 H.4 展示 CameraCtrl 的灵活性。

#### H.1 多域 T2V 视频的可视化结果

**RealEstate10K 域的视觉结果。** 首先，结合前述在 RealEstate10K 上训练的 image LoRA 模型，并使用来自 RealEstate10K 的标题与相机轨迹，CameraCtrl 能够在 RealEstate10K 域内生成视频。结果见图 11：生成视频中的相机运动紧密跟随控制相机姿态，生成内容也与文本提示对齐。

![图 11. RealEstate10K 视觉结果。CameraCtrl 的视频生成结果。控制相机轨迹与标题均来自 RealEstate10K 测试集。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure9.png)

**原始 T2V 模型域的视觉结果。** 我们选择在 WebVid-10M 上训练的 AnimateDiff V3 作为视频生成基础模型。在不使用 RealEstate10K image LoRA 的情况下，CameraCtrl 可用于控制自然物体与场景视频生成中的相机姿态。如图 12 所示，在相同文本提示下，以不同相机轨迹为输入，CameraCtrl 能生成几乎相同的场景，并紧密跟随相机轨迹。此外，图 13 给出自然物体与场景的更多视觉结果。

![图 12. 同一标题、不同相机轨迹下使用 CameraCtrl。CameraCtrl 的相机控制结果。相机轨迹来自 RealEstate10K 测试集，所有视频使用相同文本提示。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure10.png)

![图 13. 自然物体与场景的视觉结果。CameraCtrl 的自然视频生成结果。CameraCtrl 可用于控制自然物体与场景视频生成过程中的相机姿态。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure11.png)

**若干个性化视频域的视觉结果。** 通过将 T2V 模型的图像生成器骨干替换为某些个性化生成器，CameraCtrl 可用于控制个性化视频中的相机姿态。使用个性化生成器 RealisticVision，图 14 展示若干风格化物体与场景的结果，例如风景与海岸线中不常见的配色。此外，使用另一个性化生成器 ToonYou，CameraCtrl 可用于卡通角色视频生成。部分结果见图 15。在两个域中，生成视频中的相机轨迹都紧密跟随控制相机姿态。

![图 14. 风格化物体与场景的视觉结果。借助个性化生成器 RealisticVision，CameraCtrl 可用于风格化视频的生成过程。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure12.png)

![图 15. 卡通角色的视觉结果。借助个性化生成器 ToonYou，CameraCtrl 可用于卡通角色视频的生成过程。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure13.png)

#### H.2 与 SVD 集成的 I2V 可视化结果

以 SVD 为基础视频生成器实现 CameraCtrl，我们可以在 I2V 设定下采样具有期望相机轨迹的视频。图 16 展示其中若干结果。生成视频的相机视点严格跟随输入相机轨迹，视频内容也与条件图像对齐。

![图 16. 在 I2V 设定下将 CameraCtrl 与 SVD 集成。条件图像位于每行第一张图的右下角。这些图像由 text-to-image 模型 SDXL 以每行下方文本提示为输入生成。生成视频的条件信号仅为图像，不包含文本提示。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure14.png)

#### H.3 将 CameraCtrl 与其他视频控制方法集成

图 17 给出将 CameraCtrl 与另一视频控制方法 SparseCtrl 集成后的若干生成结果。生成视频的内容紧密跟随输入 RGB 图像或草图，视频的相机轨迹也与条件相机轨迹有效对齐。

![图 17. 将 CameraCtrl 与其他视频生成控制方法集成。第 1 至第 3 行是将 CameraCtrl 与 SparseCtrl 的 RGB encoder 集成的结果；第 4 至第 6 行展示用 SparseCtrl 的 sketch encoder 产生的视频。条件 RGB 图像与草图见每行第二张图的右下角。注意，最后一行的相机轨迹为 zoom-in。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure15.png)

#### H.4 CameraCtrl 的灵活性

**不同的相机运动强度。** 通过调整相邻两个相机姿态平移向量之间的间隔，我们可以控制相机运动的整体强度。如图 18 所示，我们可以使相机运动更剧烈或更平缓。

![图 18. 相机运动强度。前两行以 pan down 相机轨迹为输入，第二行的相机平移间隔为第一行的四倍。第三、四行的相机轨迹为 zoom in，第四行的相机平移间隔为第三行的四倍。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/enlarged_gap.png)

**通过调整内参控制相机运动。** 由于计算 Plücker embedding 时需要内参，我们可以通过修改相机内参来实现相机运动。如图 19 所示，通过改变相机主点位置 $(c_x, c_y)$，可以实现相机平移（见前三行）。通过调整焦距 $(f_x, f_y)$，可以实现 zoom-in 与 zoom-out 效果，见最后两行。

![图 19. 通过调整内参控制相机运动。前三行分别展示使用相机 pan left、left up、right down 的生成结果。最后两行以 zoom in、zoom out 相机轨迹为输入。每条相机轨迹中，所有相机姿态具有相同外参矩阵；相机运动通过调整内参实现：前三行为 $c_x$ 与 $c_y$，最后两行为 $f_x$ 与 $f_y$。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/differentk.png)

### 附录 I. 失败案例

图 20 给出 CameraCtrl 的若干失败案例。这些案例的主要问题是：当相机轨迹的旋转幅度较大时，CameraCtrl 无法恰当生成具有足够旋转的视频。第 1、2 行以垂直匀速旋转 100 度为输入，但生成视频无法旋转 100 度。第 3、4 行存在同样问题：我们期望水平匀速旋转 150 度，然而生成视频大约只有 90 度旋转。这一失败情况的主要原因可能在于训练数据集（RealEstate10K）不包含足够多的大角度旋转相机轨迹。因此，要进一步提升相机轨迹表现，需要外观与 RealEstate10K 相近、且相机姿态分布更大的数据集。

![图 20. 失败案例。所有结果均由实现于 AnimateDiff V3 的 CameraCtrl 在 T2V 设定下生成。第 1、2 行的相机轨迹为垂直匀速旋转 100 度。第 3、4 行生成时使用水平匀速旋转 150 度的轨迹。](../../../arxiv/geometry_control/cameractrl/extracted/images_supp/figure16.png)
