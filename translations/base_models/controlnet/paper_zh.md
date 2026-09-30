# 为文本到图像扩散模型添加条件控制（Adding Conditional Control to Text-to-Image Diffusion Models）

**作者：** Lvmin Zhang、Anyi Rao、Maneesh Agrawala  
**机构：** Stanford University  
**会议：** ICCV 2023  
**邮箱：** {lvmin, anyirao, maneesh}@cs.stanford.edu  
**arXiv：** [2302.05543](https://arxiv.org/abs/2302.05543)  
**原文 TeX：** [`arxiv/base_models/controlnet/extracted/controlnet.tex`](../../../arxiv/base_models/controlnet/extracted/controlnet.tex)

---

![图1：用学到的条件控制 Stable Diffusion。ControlNet 允许用户添加 Canny 边缘（上）、人体姿态（下）等条件，以控制大型预训练扩散模型的图像生成。默认结果使用提示**a high-quality, detailed, and professional image**。用户也可可选地给出如**chef in kitchen**这类提示。](../../../arxiv/base_models/controlnet/extracted/imgs/tea.png)

## 摘要

我们提出 ControlNet：一种神经架构，用于为大型预训练文本到图像扩散模型添加空间条件控制。ControlNet **锁定**生产就绪的大型扩散模型，并复用其在数十亿图像上预训练得到的深层、稳健编码层，作为学习多种条件控制的强骨干。该网络通过**零卷积**（零初始化卷积层）连接，使参数从零逐步生长，并保证微调过程中不会引入有害噪声。我们在 Stable Diffusion 上测试了多种条件控制，例如边缘、深度、分割、人体姿态等，支持单条件或多条件、有提示或无提示。实验表明，ControlNet 在小规模（$<$50k）与大规模（$>$1m）数据上训练均较稳健。大量结果表明，ControlNet 有望拓宽对图像扩散模型的可控应用。

---

## 1. 引言

我们许多人都曾有过**灵光一现**的视觉灵感，并希望将其捕捉为一张独特的图像。随着文本到图像扩散模型的出现，我们只需输入文本提示，就能创造视觉上令人惊叹的图像。然而，文本到图像模型对图像**空间构图**的控制有限：仅靠文本提示，很难精确表达复杂布局、姿态、形状与形态。要生成与心中意象精确匹配的图像，往往需要反复**改提示 → 看结果 → 再改提示**的试错循环。

我们能否通过让用户提供额外图像、直接指定期望的图像构图，从而实现更细粒度的空间控制？在计算机视觉与机器学习中，这些额外图像（例如边缘图、人体姿态骨架、分割图、深度、法线等）常被视为对图像生成过程的**条件**。图像到图像翻译模型学习从条件图像到目标图像的映射。研究社区也已尝试用空间掩码、图像编辑指令、经微调的个性化等方式控制文本到图像模型。虽然一些问题（例如生成图像变体、修复）可用无需训练的技术解决——如约束去噪扩散过程或编辑注意力层激活——但更广泛的问题（如深度到图像、姿态到图像等）需要端到端学习与数据驱动方案。

以端到端方式为大型文本到图像扩散模型学习条件控制具有挑战性。特定条件的训练数据量可能远小于通用文本到图像训练可用的数据。例如，各类特定问题（物体形状/法线、人体姿态提取等）的最大数据集通常约 10 万量级，约为训练 Stable Diffusion 所用 LAION-5B 的五万分之一。用有限数据直接微调或继续训练大型预训练模型，可能导致过拟合与灾难性遗忘。已有研究表明，通过限制可训练参数数量或秩，可缓解此类遗忘。对我们的问题而言，处理野外条件图像中的复杂形状与多样高层语义，可能需要设计更深或更定制化的神经架构。

本文提出 ControlNet：一种端到端神经架构，为大型预训练文本到图像扩散模型（实现中为 Stable Diffusion）学习条件控制。ControlNet 通过锁定其参数来保留大模型的质量与能力，同时对其编码层制作一份**可训练副本**。该架构将大型预训练模型视为学习多样条件控制的强骨干。可训练副本与原始锁定模型通过**零卷积**层连接：权重初始化为零，从而在训练中逐步生长。该架构保证训练初期不会向大型扩散模型的深层特征注入有害噪声，并保护可训练副本中大规模预训练骨干不被此类噪声损坏。

实验表明，ControlNet 可用多种条件输入控制 Stable Diffusion，包括 Canny 边缘、Hough 线、用户涂鸦、人体关键点、分割图、形状法线、深度等（图 1）。我们在单张条件图像上测试了有/无文本提示的情形，并展示如何组合多种条件。此外，我们报告 ControlNet 在不同规模数据集上训练稳健且可扩展；对某些任务（如深度到图像），在单块 NVIDIA RTX 3090Ti 上训练 ControlNet，即可达到与在大型计算集群上训练的工业模型相竞争的结果。最后，我们进行消融研究以考察各组件贡献，并通过用户研究与若干强条件图像生成基线比较。

总结：（1）我们提出 ControlNet，一种可通过高效微调向预训练文本到图像扩散模型添加空间局部输入条件的神经架构；（2）我们给出在 Canny 边缘、Hough 线、用户涂鸦、人体关键点、分割图、形状法线、深度与卡通线稿等条件下控制 Stable Diffusion 的预训练 ControlNet；（3）我们用消融实验与若干替代架构比较以验证方法，并在不同任务上针对若干先前基线开展用户研究。

---

## 2. 相关工作

### 2.1 微调神经网络

微调神经网络的一种方式是用额外训练数据直接继续训练。但该方法可能导致过拟合、模式崩塌与灾难性遗忘。大量研究致力于开发能避免这些问题的微调策略。

**HyperNetwork** 起源于自然语言处理（NLP）社区，旨在训练一个小型循环神经网络以影响更大网络的权重。该方法已被应用于 GAN 的图像生成。Heathen 等人与 Kurumuz 为 Stable Diffusion 实现了 HyperNetwork，以改变其输出图像的艺术风格。

**Adapter** 方法在 NLP 中广泛用于通过向预训练 Transformer 嵌入新模块层，将其定制到其他任务。在计算机视觉中，adapter 用于增量学习与域自适应。该技术常与 CLIP 结合，将预训练骨干迁移到不同任务。最近，adapter 在视觉 Transformer 与 ViT-Adapter 上取得成功。与我们同期的工作 T2I-Adapter 将 Stable Diffusion 适配到外部条件。

**加法学习（Additive Learning）** 通过冻结原模型权重、并用学到的权重掩码、剪枝或硬注意力等方式加入少量新参数，以避免遗忘。Side-Tuning 使用旁路模型，通过按预定义混合权重日程线性混合冻结模型与新增网络的输出，以学习额外功能。

**低秩适配（LoRA）** 基于许多过参数化模型位于低内在维子空间的观察，用低秩矩阵学习参数偏移，从而防止灾难性遗忘。

**零初始化层** 被 ControlNet 用于连接网络块。关于神经网络权重初始化与操控已有大量讨论。例如，高斯初始化可能比零初始化风险更低。更近地，Nichol 等人讨论了如何缩放扩散模型中卷积层的初始权重以改善训练，其**zero_module**实现是将权重缩放到零的极端情形。Stability 的模型卡也提到在神经层中使用零权重。操控初始卷积权重亦见于 ProGAN、StyleGAN 与 Noise2Noise。

### 2.2 图像扩散

**图像扩散模型** 最初由 Sohl-Dickstein 等人提出，近期被应用于图像生成。潜在扩散模型（LDM）在潜在图像空间中执行扩散步骤，从而降低计算成本。文本到图像扩散模型通过 CLIP 等预训练语言模型将文本编码为潜向量，达到当前最优的图像生成效果。Glide 是支持图像生成与编辑的文本引导扩散模型。Disco Diffusion 用 CLIP 引导处理文本提示。Stable Diffusion 是潜在扩散的大规模实现。Imagen 用金字塔结构直接在像素上扩散，不使用潜图像。商业产品包括 DALL·E 2 与 Midjourney。

**控制图像扩散模型** 有助于个性化、定制化或任务特定的图像生成。扩散过程本身对颜色变化与修复提供一定控制。文本引导控制方法聚焦于调整提示、操控 CLIP 特征与修改交叉注意力。MakeAScene 将分割掩码编码为 token 以控制图像生成。SpaText 将分割掩码映射为局部化 token 嵌入。GLIGEN 在扩散模型注意力层中学习新参数以做 grounded 生成。Textual Inversion 与 DreamBooth 可通过用少量用户示例图像微调扩散模型，个性化生成内容。基于提示的图像编辑提供用提示操控图像的实用工具。Voynov 等人提出将扩散过程与草图拟合的优化方法。同期工作考察了控制扩散模型的多种途径。

### 2.3 图像到图像翻译

条件 GAN 与 Transformer 可学习不同图像域之间的映射，例如 Taming Transformer 是视觉 Transformer 方法；Palette 是从零训练的条件扩散模型；PITI 是基于预训练的条件扩散图像到图像翻译模型。操控预训练 GAN 可处理特定图像到图像任务，例如 StyleGAN 可由额外编码器控制，更多应用见相关文献。

---

## 3. 方法

![图2：一个神经块以特征图 $x$ 为输入并输出另一特征图 $y$，如 (a) 所示。要为这样的块添加 ControlNet，我们锁定原始块并创建可训练副本，用零卷积层（即权重与偏置均初始化为零的 $1\times 1$ 卷积）将二者连接。此处 $c$ 是我们希望加入网络的条件向量，如 (b) 所示。](../../../arxiv/base_models/controlnet/extracted/imgs/he.png)

ControlNet 是一种神经架构，可用空间局部、任务特定的图像条件增强大型预训练文本到图像扩散模型。我们先在第 3.1 节介绍 ControlNet 基本结构，再在第 3.2 节描述如何将其应用于图像扩散模型 Stable Diffusion。第 3.3 节阐述训练，第 3.4 节详述推理时的若干额外考虑（如组合多个 ControlNet）。

### 3.1 ControlNet

ControlNet 将额外条件注入神经网络的各个块（图 2）。此处**网络块**指常被组合为神经网络单一单元的一组神经层，例如 ResNet 块、conv-bn-relu 块、多头注意力块、Transformer 块等。设 $\mathcal{F}(\cdot;\Theta)$ 是这样一个已训练的神经块，参数为 $\Theta$，将输入特征图 $\bm{x}$ 变换为另一特征图 $\bm{y}$：

$$\bm{y}=\mathcal{F}(\bm{x};\Theta).$$

在我们的设定中，$\bm{x}$ 与 $\bm{y}$ 通常为二维特征图，即 $\bm{x}\in\mathbb{R}^{h\times w \times c}$，其中 $\{h, w, c\}$ 分别为高、宽与通道数（图 2a）。

要为这样的预训练神经块添加 ControlNet，我们锁定（冻结）原块参数 $\Theta$，同时将该块克隆为参数为 $\Theta_\text{c}$ 的**可训练副本**（图 2b）。可训练副本以外部条件向量 $\bm{c}$ 为输入。当该结构应用于 Stable Diffusion 这类大模型时，锁定参数保留了在数十亿图像上训练好的生产就绪模型；可训练副本则复用这一大规模预训练模型，为处理多样输入条件建立深层、稳健、强大的骨干。

可训练副本通过**零卷积**层 $\mathcal{Z}(\cdot;\cdot)$ 连接到锁定模型。具体地，$\mathcal{Z}(\cdot;\cdot)$ 是权重与偏置均初始化为零的 $1\times 1$ 卷积层。构建 ControlNet 时，我们使用两个零卷积实例，参数分别为 $\Theta_\text{z1}$ 与 $\Theta_\text{z2}$。完整 ControlNet 计算为：

$$\bm{y}_\text{c}=\mathcal{F}(\bm{x};\Theta)+\mathcal{Z}\big(\mathcal{F}(\bm{x}+\mathcal{Z}(\bm{c};\Theta_\text{z1});\Theta_\text{c});\Theta_\text{z2}\big),\tag{1}$$

其中 $\bm{y}_\text{c}$ 是 ControlNet 块的输出。在第一个训练步，由于零卷积层的权重与偏置均初始化为零，式 (1) 中两个 $\mathcal{Z}(\cdot;\cdot)$ 项均求值为零，从而：

$$\bm{y}_\text{c} = \bm{y}.\tag{2}$$

这样，训练开始时有害噪声不会影响可训练副本中神经网络层的隐藏状态。此外，由于 $\mathcal{Z}(\bm{c};\Theta_\text{z1})=\bm{0}$，且可训练副本也接收输入图像 $\bm{x}$，可训练副本功能完整，并保留大型预训练模型的能力，从而可作为进一步学习的强骨干。零卷积通过在初始训练步消除作为梯度的随机噪声来保护该骨干。零卷积的梯度计算详见补充材料。

![图3：Stable Diffusion 的 U-Net 架构与 ControlNet 在编码器块与中间块上的连接。锁定的灰色块展示 Stable Diffusion V1.5（或 V2.1，二者使用相同 U-Net 架构）的结构。蓝色可训练块与白色零卷积层被加入以构建 ControlNet。](../../../arxiv/base_models/controlnet/extracted/imgs/sd.png)

### 3.2 面向文本到图像扩散的 ControlNet

我们以 Stable Diffusion 为例，展示 ControlNet 如何为大型预训练扩散模型添加条件控制。Stable Diffusion 本质上是一个带有编码器、中间块与跳跃连接解码器的 U-Net。编码器与解码器各含 12 个块，全模型含 25 个块（含中间块）。在这 25 个块中，8 个是下采样或上采样卷积层，其余 17 个是主块，每个含 4 个 ResNet 层与 2 个 Vision Transformer（ViT）。每个 ViT 含若干交叉注意力与自注意力机制。例如，在图 3a 中，**SD Encoder Block A**含 4 个 ResNet 层与 2 个 ViT，而**$\times 3$**表示该块重复三次。文本提示用 CLIP 文本编码器编码，扩散时间步用位置编码经时间编码器编码。

ControlNet 结构被应用于 U-Net 的每一编码器层级（图 3b）。具体地，我们用 ControlNet 为 Stable Diffusion 的 12 个编码块与 1 个中间块创建可训练副本。12 个编码块处于 4 种分辨率（$64\times64$、$32\times32$、$16\times16$、$8\times8$），每种重复 3 次。输出被加到 U-Net 的 12 条跳跃连接与 1 个中间块上。由于 Stable Diffusion 是典型的 U-Net 结构，该 ControlNet 架构很可能也适用于其他模型。

我们连接 ControlNet 的方式在计算上高效——由于锁定副本参数被冻结，微调时原锁定编码器无需梯度计算。这加速了训练并节省 GPU 内存。在单块 NVIDIA A100 PCIE 40GB 上测试：相对无 ControlNet 优化 Stable Diffusion，带 ControlNet 的优化每个训练迭代约多需 23% GPU 内存与 34% 时间。

图像扩散模型学习逐步去噪图像，并从训练域生成样本。去噪可发生在像素空间，或发生在由训练数据编码得到的**潜在**空间。Stable Diffusion 以潜图像为训练域，因在该空间工作已被证明可稳定训练过程。具体地，Stable Diffusion 使用类似 VQ-GAN 的预处理，将 $512\times 512$ 像素空间图像转换为更小的 $64\times 64$ **潜图像**。要将 ControlNet 加入 Stable Diffusion，我们先将每个输入条件图像（边缘、姿态、深度等）从 $512\times 512$ 转换为与 Stable Diffusion 尺寸匹配的 $64\times 64$ 特征空间向量。具体地，我们使用一个由四层卷积组成的微型网络 $\mathcal{E}(\cdot)$（$4\times 4$ 核、$2\times 2$ 步长，ReLU 激活，通道分别为 16、32、64、128；用高斯权重初始化，并与全模型联合训练），将图像空间条件 $\bm{c}_\text{i}$ 编码为特征空间条件向量 $\bm{c}_\text{f}$：

$$\bm{c}_\text{f}=\mathcal{E}(\bm{c}_\text{i}).$$

条件向量 $\bm{c}_\text{f}$ 被送入 ControlNet。

### 3.3 训练

给定输入图像 $\bm{z}_0$，图像扩散算法逐步向图像加噪并产生噪声图像 $\bm{z}_t$，其中 $t$ 表示加噪次数。给定包括时间步 $\bm{t}$、文本提示 $\bm{c}_t$ 以及任务特定条件 $\bm{c}_\text{f}$ 在内的一组条件，图像扩散算法学习网络 $\epsilon_\theta$，以预测加到噪声图像 $\bm{z}_t$ 上的噪声：

$$\mathcal{L} = \mathbb{E}_{\bm{z}_0, \bm{t}, \bm{c}_t, \bm{c}_\text{f}, \epsilon \sim \mathcal{N}(0, 1) }\Big[ \Vert \epsilon - \epsilon_\theta(\bm{z}_{t}, \bm{t}, \bm{c}_t, \bm{c}_\text{f})) \Vert_{2}^{2}\Big],\tag{3}$$

其中 $\mathcal{L}$ 是整个扩散模型的总体学习目标。该目标被直接用于带 ControlNet 的扩散模型微调。

训练过程中，我们随机将 50% 的文本提示 $\bm{c}_t$ 替换为空字符串。该方法增强了 ControlNet 直接识别输入条件图像（边缘、姿态、深度等）中语义、以替代提示的能力。

![图4：**突然收敛**现象。由于零卷积，ControlNet 在整个训练过程中始终能预测高质量图像。在训练的某一步（例如加粗标出的第 6133 步），模型突然学会跟随输入条件。](../../../arxiv/base_models/controlnet/extracted/imgs/train.png)

![图5：无分类器引导（CFG）与所提出的 CFG 分辨率加权（CFG-RW）的效果。](../../../arxiv/base_models/controlnet/extracted/imgs/cfg.png)

![图6：多条件组合。我们展示同时使用深度与姿态的应用。](../../../arxiv/base_models/controlnet/extracted/imgs/multi.png)

训练过程中，由于零卷积不向网络加噪，模型应始终能够预测高质量图像。我们观察到：模型并非逐渐学会控制条件，而是突然成功跟随输入条件图像；通常在少于 1 万次优化步内发生。如图 4 所示，我们称之为**突然收敛现象**。

![图7：在无提示情况下用多种条件控制 Stable Diffusion。顶行为输入条件，其余行为输出。我们使用空字符串作为输入提示。所有模型均在通用域数据上训练。模型必须识别输入条件图像中的语义内容才能生成图像。](../../../arxiv/base_models/controlnet/extracted/imgs/qua.png)

### 3.4 推理

我们可用若干方式进一步控制 ControlNet 的额外条件如何影响去噪扩散过程。

**无分类器引导的分辨率加权。** Stable Diffusion 依赖称为无分类器引导（Classifier-Free Guidance, CFG）的技术以生成高质量图像。CFG 表述为 $\epsilon_\text{prd}=\epsilon_\text{uc}+\beta_\text{cfg}(\epsilon_\text{c}-\epsilon_\text{uc})$，其中 $\epsilon_\text{prd}$、$\epsilon_\text{uc}$、$\epsilon_\text{c}$、$\beta_\text{cfg}$ 分别为模型最终输出、无条件输出、条件输出与用户指定权重。当通过 ControlNet 加入条件图像时，可将其同时加入 $\epsilon_\text{uc}$ 与 $\epsilon_\text{c}$，或仅加入 $\epsilon_\text{c}$。在困难情形（例如无提示）下，同时加入二者会完全去除 CFG 引导（图 5b）；仅使用 $\epsilon_\text{c}$ 会使引导过强（图 5c）。我们的方案是：先将条件图像加入 $\epsilon_\text{c}$，再按各块分辨率对 Stable Diffusion 与 ControlNet 之间的每条连接乘以权重 $w_i=64/h_i$，其中 $h_i$ 是第 $i$ 个块的尺寸，例如 $h_1=8, h_2=16, \ldots, h_{13}=64$。通过降低 CFG 引导强度，我们可得到图 5d 的结果，并称之为 CFG 分辨率加权（CFG Resolution Weighting）。

**组合多个 ControlNet。** 要将多种条件图像（例如 Canny 边缘与姿态）应用于同一 Stable Diffusion 实例，可直接将对应 ControlNet 的输出加到 Stable Diffusion 模型上（图 6）。此类组合无需额外加权或线性插值。

| 方法 | 结果质量 $\uparrow$ | 条件保真度 $\uparrow$ |
|------|---------------------|----------------------|
| PITI (sketch) | $1.10 \pm 0.05$ | $1.02 \pm 0.01$ |
| Sketch-Guided ($\beta=1.6$) | $3.21 \pm 0.62$ | $2.31 \pm 0.57$ |
| Sketch-Guided ($\beta=3.2$) | $2.52 \pm 0.44$ | $3.28 \pm 0.72$ |
| ControlNet-lite | $3.93 \pm 0.59$ | $4.09 \pm 0.46$ |
| ControlNet | **$4.22 \pm 0.43$** | **$4.28 \pm 0.45$** |

表 1：结果质量与条件保真度的平均用户排序（AUR）。我们报告不同方法的用户偏好排序（1 到 5 表示从最差到最好）。

![图8：在草图条件与不同提示设定下对不同架构的消融研究。每种设定展示未经挑选的随机 6 样本批次。图像为 $512\times 512$，放大查看效果更佳。左侧绿色**conv**块为用高斯权重初始化的标准卷积层。](../../../arxiv/base_models/controlnet/extracted/imgs/ablat.png)

---

## 4. 实验

我们用 Stable Diffusion 实现 ControlNet，以测试多种条件，包括 Canny 边缘、深度图、法线图、M-LSD 线、HED 软边缘、ADE20K 分割、OpenPose 以及用户草图。每种条件的示例及详细训练与推理参数见补充材料。

### 4.1 定性结果

图 1 展示了若干提示设定下的生成图像。图 7 展示了无提示时多种条件下的结果：ControlNet 稳健地解释多样输入条件图像中的内容语义。

### 4.2 消融研究

![图9：与先前方法的比较。我们给出相对 PITI、Sketch-Guided Diffusion 与 Taming Transformers 的定性比较。](../../../arxiv/base_models/controlnet/extracted/imgs/compa.png)

我们通过以下方式研究 ControlNet 的替代结构：（1）用高斯权重初始化的标准卷积层替换零卷积；（2）用单个卷积层替换每个块的可训练副本，称为 ControlNet-lite。这些消融结构的完整细节见补充材料。

我们给出 4 种提示设定，以测试真实用户可能的行为：（1）无提示；（2）不足以完全覆盖条件图像中物体的不充分提示，例如本文默认提示**a high-quality, detailed, and professional image**；（3）改变条件图像语义的冲突提示；（4）描述必要内容语义的完美提示，例如**a nice house**。图 8a 显示 ControlNet 在全部 4 种设定下均成功。轻量 ControlNet-lite（图 8c）不足以解释条件图像，在不充分提示与无提示条件下失败。当零卷积被替换时，ControlNet 性能下降到约与 ControlNet-lite 相当，表明可训练副本的预训练骨干在微调中被破坏（图 8b）。

| ADE20K (GT) | VQGAN | LDM | PITI | ControlNet-lite | ControlNet |
|-------------|-------|-----|------|-----------------|------------|
| $0.58 \pm 0.10$ | $0.21 \pm 0.15$ | $0.31 \pm 0.09$ | $0.26 \pm 0.16$ | $0.32 \pm 0.12$ | **$0.35 \pm 0.14$** |

表 2：语义分割标签重建（ADE20K）的交并比（IoU $\uparrow$）评估。

### 4.3 定量评估

**用户研究。** 我们采样 20 张未见过的手绘草图，并将每张草图分配给 5 种方法：PITI 的草图模型、默认边缘引导尺度（$\beta=1.6$）的 Sketch-Guided Diffusion (SGD)、相对较高边缘引导尺度（$\beta=3.2$）的 SGD、前述 ControlNet-lite，以及 ControlNet。我们邀请 12 名用户分别就**显示图像的质量**与**对草图的保真度**对这 20 组、每组 5 个结果进行排序。由此我们得到 100 次结果质量排序与 100 次条件保真度排序。我们使用平均人类排序（Average Human Ranking, AHR）作为偏好指标，用户按 1 到 5 分对每个结果排序（越低越差）。平均排序见表 1。

**与工业模型比较。** Stable Diffusion V2 Depth-to-Image (SDv2-D2I) 在大规模 NVIDIA A100 集群上训练，耗时数千 GPU 小时，训练图像超过 1200 万。我们为 SD V2 训练相同深度条件的 ControlNet，但仅使用 20 万训练样本、单块 NVIDIA RTX 3090Ti 与 5 天训练。我们用每种方法生成的 100 张图像教 12 名用户区分两种方法；之后生成 200 张图像，请用户判断每张由哪个模型生成。用户平均精度为 $0.52 \pm 0.17$，表明两种方法结果几乎无法区分。

| 方法 | FID  | CLIP-score $\uparrow$ | CLIP-aes. $\uparrow$ |
|------|------------------|----------------------|---------------------|
| Stable Diffusion | 6.09 | 0.26 | 6.32 |
| VQGAN (seg.)* | 26.28 | 0.17 | 5.14 |
| LDM (seg.)* | 25.35 | 0.18 | 5.15 |
| PITI (seg.) | 19.74 | 0.20 | 5.77 |
| ControlNet-lite | 17.92 | 0.26 | 6.30 |
| ControlNet | 15.27 | 0.26 | 6.31 |

表 3：语义分割条件图像生成的评估。我们报告 FID、CLIP 文本-图像分数与 CLIP 美学分数。亦报告无分割条件的 Stable Diffusion 性能。标*****的方法为从零训练。

**条件重建与 FID 分数。** 我们用 ADE20K 测试集评估条件保真度。当前最优分割方法 OneFormer 在真值集上达到 IoU 0.58。我们用不同方法按 ADE20K 分割生成图像，再以 OneFormer 检测分割并计算重建 IoU（表 2）。此外，我们用 Frechet Inception Distance (FID) 度量不同分割条件方法随机生成的 $512\times512$ 图像集之间的分布距离，以及文本-图像 CLIP 分数与 CLIP 美学分数（表 3）。详细设定见补充材料。

### 4.4 与先前方法比较

图 9 给出基线与我们方法（Stable Diffusion + ControlNet）的视觉比较。具体展示 PITI、Sketch-Guided Diffusion 与 Taming Transformers 的结果。（注意 PITI 的骨干是 OpenAI GLIDE，视觉质量与性能不同。）我们观察到 ControlNet 能稳健处理多样条件图像，并取得清晰干净的结果。

![图10：不同训练数据集规模的影响。扩展示例见补充材料。](../../../arxiv/base_models/controlnet/extracted/imgs/datasize.png)

![图11：解释内容。若输入模糊且用户未在提示中提及物体内容，结果看起来像模型在尝试解释输入形状。](../../../arxiv/base_models/controlnet/extracted/imgs/guess.png)

![图12：将预训练 ControlNet 迁移到社区模型而无需再次训练神经网络。](../../../arxiv/base_models/controlnet/extracted/imgs/trans.png)

### 4.5 讨论

**训练数据集规模的影响。** 我们在图 10 中展示 ControlNet 训练的稳健性。训练在仅 1k 图像时不会崩塌，并允许模型生成可识别的狮子。当提供更多数据时，学习可扩展。

**解释内容的能力。** 我们在图 11 中展示 ControlNet 从输入条件图像捕获语义的能力。

**迁移到社区模型。** 由于 ControlNet 不改变预训练 SD 模型的网络拓扑，它可直接应用于 Stable Diffusion 社区中的各种模型，例如 Comic Diffusion 与 Protogen 3.4（图 12）。

---

## 5. 结论

ControlNet 是一种为大型预训练文本到图像扩散模型学习条件控制的神经网络结构。它复用源模型的大规模预训练层，构建深层强编码器以学习特定条件。原模型与可训练副本通过**零卷积**层连接，在训练中消除有害噪声。大量实验验证：ControlNet 能有效用单条件或多条件、有提示或无提示控制 Stable Diffusion。在多样条件数据集上的结果表明，ControlNet 结构很可能适用于更广泛的条件，并促进相关应用。

---

## 致谢

本工作部分得到 Stanford Institute for Human-Centered AI 与 Brown Institute for Media Innovation 的支持。
