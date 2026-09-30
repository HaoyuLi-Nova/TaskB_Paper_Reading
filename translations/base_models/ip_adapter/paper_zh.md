# IP-Adapter：面向文本到图像扩散模型的文本兼容图像提示适配器（IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models）

**作者：** Hu Ye、Jun Zhang（通讯作者）、Sibo Liu、Xiao Han、Wei Yang
**机构：** Tencent AI Lab
**邮箱：** {huye, junejzhang, siboliu, haroldhan, willyang}@tencent.com
**年份：** 2023
**arXiv：** [2308.06721](https://arxiv.org/abs/2308.06721)
**项目页：** https://ip-adapter.github.io
**原文 TeX：** [`arxiv/base_models/ip_adapter/extracted/templateArxiv.tex`](../../../arxiv/base_models/ip_adapter/extracted/templateArxiv.tex)

---

## 摘要

近年来，大规模文本到图像扩散模型展现出强大能力，能够以令人印象深刻的生成能力创作高保真图像。然而，仅使用文本提示生成期望图像往往非常棘手，因为这通常涉及复杂的提示工程。文本提示的一种替代方案是图像提示，正所谓**一图胜千言**。尽管从预训练模型直接微调现有方法是有效的，但它们需要大量计算资源，且与其他基座模型、文本提示及结构控制不兼容。在本文中，我们提出 IP-Adapter，一种高效且轻量的适配器，为预训练文本到图像扩散模型实现图像提示能力。我们 IP-Adapter 的关键设计是解耦交叉注意力机制，将文本特征与图像特征的交叉注意力层分离。尽管方法简单，仅 22M 参数的 IP-Adapter 即可达到与全量微调图像提示模型相当甚至更优的性能。由于我们冻结预训练扩散模型，所提出的 IP-Adapter 不仅可泛化到从同一基座模型微调的其他自定义模型，还可与现有可控工具结合实现可控生成。得益于解耦交叉注意力策略，图像提示也可与文本提示良好配合，实现多模态图像生成。项目页面见 https://ip-adapter.github.io。

---

## 1. 引言

在 GLIDE、DALL-E 2、Imagen、Stable Diffusion（SD）、eDiff-I 和 RAPHAEL 等近期大规模文本到图像扩散模型取得成功的推动下，图像生成取得了显著进展。用户可以使用强大的文本到图像扩散模型，通过书写文本提示来生成图像。但写出能生成期望内容的好文本提示并不容易，因为通常需要复杂的提示工程。此外，文本不足以表达复杂场景或概念，这可能阻碍内容创作。考虑到文本提示的上述局限，我们可能会问：是否还有其他提示类型可用于生成图像。一个自然的选择是使用图像提示，因为与文本相比，图像能表达更多内容与细节，正如常言所说：**一图胜千言**。DALL-E 2 首次尝试支持图像提示，扩散模型条件于图像嵌入而非文本嵌入，并需要 prior 模型来实现文本到图像能力。然而，大多数现有文本到图像扩散模型条件于文本来生成图像，例如流行的 SD 模型条件于从冻结 CLIP 文本编码器提取的文本特征。图像提示能否也在这些文本到图像扩散模型上得到支持？我们的工作尝试以简单方式为这些文本到图像扩散模型启用图像提示的生成能力。

先前工作，如 SD Image Variations[^1] 和 Stable unCLIP[^2]，已证明直接在图像嵌入上微调文本条件扩散模型以实现图像提示能力的有效性。然而，该方法的缺点显而易见。首先，它消除了使用文本生成图像的原有能力，且此类微调通常需要大量计算资源。其次，微调后的模型通常不可复用，因为图像提示能力无法直接迁移到从同一文本到图像基座模型衍生的其他自定义模型。此外，新模型往往与 ControlNet 等现有结构控制工具不兼容，给下游应用带来重大挑战。由于微调的缺点，一些研究[^3] 选择用图像编码器替换文本编码器，同时避免微调扩散模型。尽管该方法有效且简单，但仍存在若干缺点。首先，仅支持图像提示，使用户无法同时使用文本和图像提示来创作图像。此外，仅微调图像编码器往往不足以保证图像质量，并可能导致泛化问题。

[^1]: https://huggingface.co/lambdalabs/sd-image-variations-diffusers
[^2]: https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip
[^3]: Xu et al., SeeCoder

在本研究中，我们好奇是否可以在不修改原始文本到图像模型的情况下实现图像提示能力。幸运的是，先前的工作令人鼓舞。可控图像生成的近期进展，如 ControlNet 和 T2I-adapter，已证明额外网络可有效插入现有文本到图像扩散模型以引导图像生成。大多数研究聚焦于带额外结构控制的图像生成，例如用户绘制的草图、深度图、语义分割图等。此外，由参考图像提供风格或内容的图像生成也已通过简单适配器实现，例如 T2I-adapter 的风格适配器以及 Uni-ControlNet 的全局控制器。为实现此目标，从 CLIP 图像编码器提取的图像特征通过可训练网络映射为新特征，然后与文本特征拼接。通过替换原始文本特征，合并后的特征被送入扩散模型的 UNet 以引导图像生成。这些适配器可视为具备图像提示能力的一种方式，但生成图像仅部分忠实于提示图像。结果往往不如微调的图像提示模型，更不用说从头训练的模型。

![图 1：将所提出的 IP-Adapter 应用于不同风格预训练文本到图像扩散模型的各种图像合成。右侧示例展示图像变体、多模态生成以及带图像提示的修复结果；左侧示例展示带图像提示及额外结构条件的可控生成结果。](../../../arxiv/base_models/ip_adapter/extracted/figures/fig0.png)

我们认为，上述方法的主要问题在于文本到图像扩散模型的交叉注意力模块。预训练扩散模型中交叉注意力层的键（key）和值（value）投影权重是为适配文本特征而训练的。因此，将图像特征与文本特征合并到交叉注意力层中，仅能实现图像特征向文本特征的对齐，但这可能丢失一些图像特有信息，并最终导致参考图像仅能实现粗粒度可控生成（例如图像风格）。

为此，我们提出更有效的图像提示适配器 IP-Adapter，以避免先前方法的缺点。具体而言，IP-Adapter 对文本特征和图像特征采用解耦交叉注意力机制。对于扩散模型 UNet 中的每个交叉注意力层，我们额外增加一个仅用于图像特征的交叉注意力层。在训练阶段，仅训练新交叉注意力层的参数，而原始 UNet 模型保持冻结。我们提出的适配器轻量但非常高效：仅 22M 参数的 IP-Adapter 的生成性能可与从文本到图像扩散模型全量微调的图像提示模型相当。更重要的是，我们的 IP-Adapter 展现出优秀的泛化能力，且与文本提示兼容。借助所提出的 IP-Adapter，可轻松实现各种图像生成任务，如图 1 所示。

综上所述，我们的贡献如下：

- 我们提出 IP-Adapter，一种面向现有文本到图像扩散模型的轻量图像提示适配方法，采用解耦交叉注意力策略。定量和定性实验结果表明，约 22M 参数的小型 IP-Adapter 在基于图像提示的生成上可与全量微调模型相当甚至更优。
- 我们的 IP-Adapter 可复用且灵活。在基座扩散模型上训练的 IP-Adapter 可泛化到从同一基座扩散模型微调的其他自定义模型。此外，IP-Adapter 与 ControlNet 等其他可控适配器兼容，便于将图像提示与结构控制轻松结合。
- 由于解耦交叉注意力策略，图像提示与文本提示兼容，可实现多模态图像生成。

---

## 2. 相关工作

我们聚焦于为现有文本到图像扩散模型设计图像提示适配器。本节回顾文本到图像扩散模型的近期工作，以及大模型适配器相关研究。

### 2.1 文本到图像扩散模型

大规模文本到图像模型主要分为两类：自回归模型和扩散模型。早期工作，如 DALL-E、CogView 和 Make-A-Scene，属于自回归模型。对于自回归模型，使用 VQ-VAE 等图像 tokenizer 将图像转换为 token，然后训练条件于文本 token 的自回归 Transformer 来预测图像 token。然而，自回归模型通常需要大量参数和计算资源才能生成高质量图像，如 Parti 所示。

近年来，扩散模型（DMs）已成为文本到图像生成的新 state-of-the-art 模型。作为先驱，GLIDE 采用级联扩散架构，包含 $64\times64$ 分辨率的 3.5B 文本条件扩散模型和 $256\times256$ 分辨率的 1.5B 文本条件上采样扩散模型。DALL-E 2 采用条件于图像嵌入的扩散模型，并训练 prior 模型以根据文本提示生成图像嵌入。DALL-E 2 不仅支持文本提示进行图像生成，也支持图像提示。为增强文本理解，Imagen 采用 T5——在纯文本数据上预训练的大型 Transformer 语言模型——作为扩散模型的文本编码器。Re-Imagen 使用检索信息提升对罕见或未见实体生成图像的保真度。SD 基于潜扩散模型，在潜空间而非像素空间操作，使 SD 仅凭扩散模型即可生成高分辨率图像。为改善文本对齐，eDiff-I 设计了文本到图像扩散模型集成，利用多种条件，包括 T5 文本、CLIP 文本和 CLIP 图像嵌入。Versatile Diffusion 提出统一的多流扩散框架，在单一模型内支持文本到图像、图像到文本和变体生成。为实现可控图像合成，Composer 在条件于图像嵌入的预训练扩散模型上提出联合多种条件的微调策略。RAPHAEL 将混合专家（MoEs）策略引入文本条件图像扩散模型，以增强图像质量与美学吸引力。

DALL-E 2 的一个吸引人的特性是也可使用图像提示生成图像变体。因此，也有一些工作探索为仅条件于文本的文本到图像扩散模型支持图像提示。SD Image Variations 模型从修改后的 SD 模型微调而来，其中文本特征被 CLIP 图像编码器的图像嵌入替换。Stable unCLIP 也是在 SD 上微调的模型，其中图像嵌入被加到时间嵌入中。尽管微调模型可成功使用图像提示生成图像，但通常需要相对较大的训练成本，且无法与现有工具（如 ControlNet）兼容。

### 2.2 大模型的 Adapter

由于微调大型预训练模型效率低下，一种替代方案是使用 Adapter，即增加少量可训练参数但冻结原始模型。Adapter 在 NLP 领域已长期使用。近期，Adapter 也被用于实现大型语言模型的视觉-语言理解。

随着近期文本到图像模型的流行，Adapter 也被用于为文本到图像模型的生成提供额外控制。ControlNet 首先证明，可与预训练文本到图像扩散模型一起训练 Adapter 以学习任务特定输入条件，例如 Canny 边缘。几乎同时，T2I-adapter 采用简单轻量的 Adapter 实现对生成图像颜色和结构的细粒度控制。为降低微调成本，Uni-ControlNet 提出多尺度条件注入策略，以学习适用于各种局部控制的 Adapter。

除用于结构控制的 Adapter 外，也有工作实现条件于所提供图像的内容与风格的可控生成。ControlNet Shuffle[^4] 训练用于重组图像，可用于由用户提供的图像引导生成。此外，ControlNet Reference-only[^5] 通过简单特征注入在 SD 模型上实现图像变体，无需训练。在 T2I-adapter 的更新版本中，设计了风格适配器，通过将 CLIP 图像编码器提取的图像特征附加到文本特征，使用参考图像控制生成图像的风格。Uni-ControlNet 的全局控制 Adapter 也将 CLIP 图像编码器的图像嵌入通过小型网络投影为条件嵌入，并与原始文本嵌入拼接，用于以参考图像的风格和内容引导生成。SeeCoder 提出语义上下文编码器，替换原始文本编码器以生成图像变体。

[^4]: https://github.com/lllyasviel/ControlNet-v1-1-nightly
[^5]: https://github.com/Mikubill/sd-webui-controlnet

尽管上述 Adapter 是轻量的，其性能难以与微调的图像提示模型相比，更不用说从头训练的模型。在本研究中，我们引入解耦交叉注意力机制，以实现更有效的图像提示 Adapter。所提出的 Adapter 保持简单且小型，但优于先前的 Adapter 方法，甚至可与微调模型相当。

![图 2：所提出 IP-Adapter 的整体架构（解耦交叉注意力策略）。仅新添加的模块（红色）被训练，预训练文本到图像模型被冻结。](../../../arxiv/base_models/ip_adapter/extracted/figures/fig1.png)

---

## 3. 方法

本节首先介绍文本到图像扩散模型的一些预备知识，然后详细描述所提出 IP-Adapter 的动机与设计。

### 3.1 预备知识

扩散模型是一类生成模型，包含两个过程：扩散过程（亦称前向过程），通过固定 $T$ 步马尔可夫链逐步向数据添加高斯噪声；以及去噪过程，用可学习模型从高斯噪声生成样本。扩散模型也可条件于其他输入，例如文本到图像扩散模型中的文本。通常，预测噪声的扩散模型 $\boldsymbol{\epsilon}_{\theta}$ 的训练目标定义为变分界的简化形式：

$$
L_{\text{simple}}=\mathbb{E}_{\boldsymbol{x}_{0},\boldsymbol{\epsilon}\sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \boldsymbol{c}, t} \| \boldsymbol{\epsilon}- \boldsymbol{\epsilon}_\theta\big(\boldsymbol{x}_t, \boldsymbol{c}, t\big)\|^2,
$$

其中 $\boldsymbol{x}_{0}$ 表示带额外条件 $\boldsymbol{c}$ 的真实数据，$t\in [0, T]$ 表示扩散过程的时间步，$\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$ 是 $t$ 步的噪声数据，$\alpha_t$、$\sigma_t$ 是 $t$ 的预定义函数，决定扩散过程。模型 $\boldsymbol{\epsilon}_{\theta}$ 训练完成后，可通过迭代方式从随机噪声生成图像。通常，推理阶段采用 DDIM、PNDM 和 DPM-Solver 等快速采样器以加速生成过程。

对于条件扩散模型，分类器引导是一种直接技术，利用单独训练分类器的梯度平衡图像保真度与样本多样性。为消除独立训练分类器的需求，常采用无分类器引导作为替代方法。在该方法中，通过在训练时随机丢弃 $\boldsymbol{c}$，联合训练条件与无条件扩散模型。在采样阶段，预测噪声基于条件模型 $\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_t, \boldsymbol{c}, t)$ 和无条件模型 $\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_t, t)$ 的预测计算：

$$
\hat{\boldsymbol{\epsilon}}_{\theta}(\boldsymbol{x}_t, \boldsymbol{c}, t) = w\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_t, \boldsymbol{c}, t)+(1-w)\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_t, t),
$$

此处 $w$（常称为引导尺度或引导权重）是调整与条件 $\boldsymbol{c}$ 对齐程度的标量。对于文本到图像扩散模型，无分类器引导在增强生成样本的图像-文本对齐方面起关键作用。

在本研究中，我们使用开源 SD 模型作为实现 IP-Adapter 的示例基座模型。SD 是条件于从冻结 CLIP 文本编码器提取的文本特征的潜扩散模型。扩散模型的架构基于带注意力层的 UNet。与 Imagen 等基于像素的扩散模型相比，SD 更高效，因为它构建于预训练自编码器模型的潜空间之上。

### 3.2 图像提示 Adapter

在本文中，图像提示 Adapter 的设计旨在使预训练文本到图像扩散模型能够使用图像提示生成图像。如前文所述，当前 Adapter 难以匹配微调图像提示模型或从头训练模型的性能。主要原因在于图像特征无法有效嵌入预训练模型。大多数方法简单地将拼接特征送入冻结的交叉注意力层，阻止扩散模型从图像提示中捕获细粒度特征。为解决该问题，我们提出解耦交叉注意力策略，其中图像特征由新添加的交叉注意力层嵌入。所提出 IP-Adapter 的整体架构如图 2 所示。所提出的 IP-Adapter 包含两部分：从图像提示提取图像特征的图像编码器，以及将图像特征嵌入预训练文本到图像扩散模型的带解耦交叉注意力的适配模块。

#### 3.2.1 图像编码器

遵循大多数方法，我们使用预训练 CLIP 图像编码器模型从图像提示提取图像特征。CLIP 模型是在包含图像-文本对的大规模数据集上通过对比学习训练的多模态模型。我们利用 CLIP 图像编码器的全局图像嵌入，其与图像描述良好对齐，可表示图像丰富的内容与风格。在训练阶段，CLIP 图像编码器被冻结。

为有效分解全局图像嵌入，我们使用小型可训练投影网络，将图像嵌入投影为长度 $N$ 的特征序列（本研究中 $N=4$），图像特征的维度与预训练扩散模型中文本特征的维度相同。本研究使用的投影网络由线性层和 Layer Normalization 组成。

#### 3.2.2 解耦交叉注意力

图像特征通过带解耦交叉注意力的适配模块集成到预训练 UNet 模型中。在原始 SD 模型中，CLIP 文本编码器的文本特征通过送入交叉注意力层插入 UNet 模型。给定 query 特征 $\mathbf{Z}$ 和文本特征 $\boldsymbol{c}_{t}$，交叉注意力输出 $\mathbf{Z}'$ 可由下式定义：

$$
\mathbf{Z}'=\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}}\right)\mathbf{V},
$$

其中 $\mathbf{Q}=\mathbf{Z}\mathbf{W}_q$、$\mathbf{K}=\boldsymbol{c}_{t}\mathbf{W}_k$、$\mathbf{V}=\boldsymbol{c}_{t}\mathbf{W}_v$ 分别是注意力操作的 query、key 和 value 矩阵，$\mathbf{W}_q$、$\mathbf{W}_k$、$\mathbf{W}_v$ 是可训练线性投影层的权重矩阵。

插入图像特征的一种直接方法是将图像特征与文本特征拼接，然后送入交叉注意力层。然而，我们发现该方法效果不足。相反，我们提出解耦交叉注意力机制，其中文本特征与图像特征的交叉注意力层相互分离。具体而言，我们在原始 UNet 模型的每个交叉注意力层旁新增一个交叉注意力层以插入图像特征。给定图像特征 $\boldsymbol{c}_{i}$，新交叉注意力输出 $\mathbf{Z}''$ 计算如下：

$$
\mathbf{Z}''=\text{Attention}(\mathbf{Q},\mathbf{K}',\mathbf{V}') = \text{Softmax}\left(\frac{\mathbf{Q}(\mathbf{K}')^{\top}}{\sqrt{d}}\right)\mathbf{V}',
$$

其中 $\mathbf{Q}=\mathbf{Z}\mathbf{W}_q$、$\mathbf{K}'=\boldsymbol{c}_{i}\mathbf{W}'_k$、$\mathbf{V}'=\boldsymbol{c}_{i}\mathbf{W}'_v$ 是来自图像特征的 query、key 和 value 矩阵，$\mathbf{W}'_k$ 和 $\mathbf{W}'_v$ 是相应权重矩阵。应注意，图像交叉注意力与文本交叉注意力使用相同的 query。因此，每个交叉注意力层仅需增加两个参数 $\mathbf{W}'_k$、$\mathbf{W}'_v$。为加速收敛，$\mathbf{W}'_k$ 和 $\mathbf{W}'_v$ 从 $\mathbf{W}_k$ 和 $\mathbf{W}_v$ 初始化。

然后，我们将图像交叉注意力的输出简单加到文本交叉注意力的输出上。因此，解耦交叉注意力的最终形式定义为：

$$
\begin{split}
\mathbf{Z}^{\text{new}}&=\text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}}\right)\mathbf{V}+\text{Softmax}\left(\frac{\mathbf{Q}(\mathbf{K}')^{\top}}{\sqrt{d}}\right)\mathbf{V}'\\
\text{where}\ \mathbf{Q}&=\mathbf{Z}\mathbf{W}_q, \mathbf{K}=\boldsymbol{c}_{t}\mathbf{W}_k, \mathbf{V}=\boldsymbol{c}_{t}\mathbf{W}_v,\\
\mathbf{K}'&=\boldsymbol{c}_{i}\mathbf{W}'_k, \mathbf{V}'=\boldsymbol{c}_{i}\mathbf{W}'_v
\end{split}
$$

由于我们冻结原始 UNet 模型，上述解耦交叉注意力中只有 $\mathbf{W}'_k$ 和 $\mathbf{W}'_v$ 可训练。

#### 3.2.3 训练与推理

在训练期间，我们仅优化 IP-Adapter，同时保持预训练扩散模型参数固定。IP-Adapter 也在图像-文本对数据集上训练[^6]，使用与原始 SD 相同的训练目标：

$$
L_{\text{simple}}=\mathbb{E}_{\boldsymbol{x}_{0},\boldsymbol{\epsilon}, \boldsymbol{c}_{t}, \boldsymbol{c}_{i}, t} \| \boldsymbol{\epsilon}- \boldsymbol{\epsilon}_\theta\big(\boldsymbol{x}_t, \boldsymbol{c}_{t}, \boldsymbol{c}_{i}, t\big)\|^2.
$$

我们还在训练阶段随机丢弃图像条件，以在推理阶段启用无分类器引导：

$$
\hat{\boldsymbol{\epsilon}}_{\theta}(\boldsymbol{x}_t, \boldsymbol{c}_{t}, \boldsymbol{c}_{i}, t) = w\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_t, \boldsymbol{c}_{t}, \boldsymbol{c}_{i}, t)+(1-w)\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_t, t)
$$

此处，若丢弃图像条件，我们简单将 CLIP 图像嵌入置零。

由于文本交叉注意力与图像交叉注意力相互分离，我们也可在推理阶段调整图像条件的权重：

$$
\mathbf{Z}^{\text{new}}=\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) + \lambda\cdot\text{Attention}(\mathbf{Q},\mathbf{K}',\mathbf{V}')
$$

其中 $\lambda$ 是权重因子，若 $\lambda=0$，模型变为原始文本到图像扩散模型。

[^6]: 注意，也可在不使用文本提示的情况下训练模型，因为仅使用图像提示就足以引导最终生成。

---

## 4. 实验

### 4.1 实验设置

#### 4.1.1 训练数据

为训练 IP-Adapter，我们构建了一个多模态数据集，包含约 1000 万图文对，来自两个开源数据集——LAION-2B 和 COYO-700M。

#### 4.1.2 实现细节

我们的实验基于 SD v1.5[^7]，并使用 OpenCLIP ViT-H/14 作为图像编码器。SD 模型中有 16 个交叉注意力层，我们为每个层新增一个图像交叉注意力层。我们的 IP-Adapter 总可训练参数（包括投影网络和适配模块）约 22M，使 IP-Adapter 相当轻量。我们使用 HuggingFace diffusers 库实现 IP-Adapter，并采用 DeepSpeed ZeRO-2 进行快速训练。IP-Adapter 在单台机器 8 块 V100 GPU 上训练 1M 步，每 GPU batch size 为 8。我们使用 AdamW 优化器，固定学习率 0.0001，权重衰减 0.01。训练期间，将图像最短边 resize 到 512，然后中心裁剪为 $512\times512$ 分辨率。为启用无分类器引导，以 0.05 概率单独丢弃文本和图像，以 0.05 概率同时丢弃文本和图像。在推理阶段，采用 DDIM 采样器 50 步，引导尺度设为 7.5。仅使用图像提示时，将文本提示设为空且 $\lambda=1.0$。

[^7]: https://huggingface.co/runwayml/stable-diffusion-v1-5

**表 1：所提出 IP-Adapter 与其他方法在 COCO 验证集上的定量对比。最优结果加粗。**

| 方法 | 可复用至自定义模型 | 兼容可控工具 | 多模态提示 | 可训练参数 | CLIP-T ↑ | CLIP-I ↑ |
|------|-------------------|--------------|------------|------------|----------|----------|
| *从头训练* | | | | | | |
| Open unCLIP | ✗ | ✗ | ✗ | 893M | **0.608** | **0.858** |
| Kandinsky-2-1 | ✗ | ✗ | ✗ | 1229M | 0.599 | 0.855 |
| Versatile Diffusion | ✗ | ✗ | ✓ | 860M | 0.587 | 0.830 |
| *从文本到图像模型微调* | | | | | | |
| SD Image Variations | ✗ | ✗ | ✗ | 860M | 0.548 | 0.760 |
| SD unCLIP | ✗ | ✗ | ✗ | 870M | 0.584 | 0.810 |
| *Adapter* | | | | | | |
| Uni-ControlNet (Global Control) | ✓ | ✓ | ✓ | 47M | 0.506 | 0.736 |
| T2I-Adapter (Style) | ✓ | ✓ | ✓ | 39M | 0.485 | 0.648 |
| ControlNet Shuffle | ✓ | ✓ | ✓ | 361M | 0.421 | 0.616 |
| **IP-Adapter** | ✓ | ✓ | ✓ | 22M | **0.588** | **0.828** |

![图 3：所提出 IP-Adapter 与其他方法在不同种类和风格图像条件下的视觉对比。](../../../arxiv/base_models/ip_adapter/extracted/figures/result1.png)

![图 4：不同扩散模型与所提出 IP-Adapter 的生成图像。IP-Adapter 仅训练一次。](../../../arxiv/base_models/ip_adapter/extracted/figures/result2.png)

![图 5：带图像提示及额外结构条件的生成样本可视化。注意我们无需微调 IP-Adapter。](../../../arxiv/base_models/ip_adapter/extracted/figures/result3.png)

![图 6：所提出 IP-Adapter 与其他方法在不同结构条件下的对比。](../../../arxiv/base_models/ip_adapter/extracted/figures/result4.png)

![图 7：所提出 IP-Adapter 使用图像提示进行图生图与修复的示例。](../../../arxiv/base_models/ip_adapter/extracted/figures/result5.png)

### 4.2 与现有方法对比

为证明我们方法的有效性，我们将 IP-Adapter 与其他现有方法在图像提示生成上进行对比。我们选择三类方法：从头训练、从文本到图像模型微调、以及 Adapter。对于从头训练的方法，我们选择 3 个开源模型：open unCLIP[^8]（DALL-E 2 的复现）、Kandinsky-2-1[^9]（DALL-E 2 与潜扩散的混合）、以及 Versatile Diffusion。对于微调模型，我们选择 SD Image Variations 和 SD unCLIP。对于 Adapter，我们将 IP-Adapter 与 T2I-Adapter 的风格适配器、Uni-ControlNet 的全局控制器、ControlNet Shuffle、ControlNet Reference-only 和 SeeCoder 进行对比。

[^8]: https://github.com/kakaobrain/karlo
[^9]: https://github.com/ai-forever/Kandinsky-2

#### 4.2.1 定量对比

我们使用 COCO2017 验证集[^10]进行定量评估，该验证集包含 5000 张带 caption 的图像。为公平对比，我们对数据集中每个样本以图像提示为条件生成 4 张图像，每种方法共生成 20000 张图像。我们使用两个指标评估与图像条件的对齐：

- CLIP-I：生成图像与图像提示在 CLIP 图像嵌入上的相似度。
- CLIP-T：生成图像与图像提示 caption 的 CLIPScore。

我们使用 CLIP ViT-L/14[^11] 模型在所有生成图像上计算两个指标的平均值。由于开源 SeeCoder 与额外结构控制一起使用，且 ControlNet Reference-only 在 Web 框架下发布，我们仅进行定性评估。对比结果见表 1。如我们所观察，我们的方法远优于其他 Adapter，且仅 22M 参数即可与微调模型相当甚至更优。

[^10]: Lin et al., COCO2017
[^11]: https://huggingface.co/openai/clip-vit-large-patch14

#### 4.2.2 定性对比

我们还选择各种种类和风格的图像对方法进行定性评估。出于隐私原因，含真人面部的图像为合成图像。对于 SeeCoder，我们还使用 ControlNet 的 scribble 控制生成图像。对于 ControlNet Reference-only，我们还输入 BLIP caption 模型生成的 caption。对于每个图像提示，我们随机生成 4 个样本，并为每种方法选择最佳样本以确保公平。如图 3 所示，所提出的 IP-Adapter 在图像质量及与参考图像的对齐方面大多优于其他 Adapter。此外，我们的方法略优于微调模型，且在大多数情况下与从头训练的模型相当。

综上所述，所提出的 IP-Adapter 是一种轻量且有效的方法，为预训练文本到图像扩散模型实现图像提示的生成能力。

### 4.3 更多结果

尽管所提出的 IP-Adapter 旨在实现图像提示生成，其鲁棒的泛化能力允许更广泛的应用。如表 1 所示，我们的 IP-Adapter 不仅可复用至自定义模型，还与现有可控工具及文本提示兼容。本节展示我们 Adapter 可生成的更多结果。

#### 4.3.1 可泛化至自定义模型

由于在训练阶段冻结原始扩散模型，IP-Adapter 也可像其他 Adapter（如 ControlNet）一样泛化到从 SD v1.5 微调的自定义模型。换言之，IP-Adapter 一旦训练完成，可直接复用于从同一基座模型微调的自定义模型。为验证这一点，我们从 HuggingFace 模型库[^12] 选择三个社区模型：Realistic Vision V4.0、Anything v4 和 ReV Animated。这些模型均从 SD v1.5 微调。如图 4 所示，我们的 IP-Adapter 在这些社区模型上表现良好。此外，生成图像可混合社区模型的风格，例如使用动漫风格模型 Anything v4 时可生成动漫风格图像。有趣的是，我们的 Adapter 可直接应用于 SD v1.4，因为 SD v1.5 是在 SD v1.4 基础上以更多步数训练的。

[^12]: https://huggingface.co/models

![图 8：所提出 IP-Adapter 使用多模态提示的生成示例。](../../../arxiv/base_models/ip_adapter/extracted/figures/result6.png)

![图 9：所提出 IP-Adapter 与其他方法在多模态提示下的对比。](../../../arxiv/base_models/ip_adapter/extracted/figures/result7.png)

![图 10：所提出 IP-Adapter 与简单 Adapter 的对比结果。简单 Adapter 未使用解耦交叉注意力策略。](../../../arxiv/base_models/ip_adapter/extracted/figures/result8.png)

![图 11：使用全局特征的 IP-Adapter 与使用细粒度特征的 IP-Adapter 之间生成样本的视觉差异。](../../../arxiv/base_models/ip_adapter/extracted/figures/result9.png)

#### 4.3.2 结构控制

对于文本到图像扩散模型，一项流行应用是可通过额外结构控制创作图像。由于我们的 Adapter 不改变原始网络结构，我们发现 IP-Adapter 与现有可控工具完全兼容。因此，我们也可使用图像提示和额外条件生成可控图像。此处，我们将 IP-Adapter 与两个现有可控工具 ControlNet 和 T2I-Adapter 结合。图 5 展示了带图像提示和不同结构控制生成的各种样本：前两行样本由 ControlNet 模型生成，最后一行样本由 T2I-Adapter 生成。我们的 Adapter 可有效与这些工具配合，无需微调即可产生更多可控图像。

我们还在结构控制生成上将 Adapter 与其他 Adapter 对比，结果如图 6 所示。对于 T2I-Adapter 和 Uni-ControlNet，我们使用默认的可组合多条件。对于 SeeCoder 和我们的 IP-Adapter，我们使用 ControlNet 实现结构控制。对于 ControlNet Shuffle 和 ControlNet Reference-only，我们使用 multi-ControlNet。如我们所见，我们的方法不仅在图像质量上优于其他方法，且生成的图像与参考图像对齐更好。

#### 4.3.3 图生图与修复

除文本到图像生成外，文本到图像扩散模型也可通过 SDEdit 实现文本引导的图生图与修复。如图 7 所示，我们也可通过简单将文本提示替换为图像提示，获得图像引导的图生图与修复。

#### 4.3.4 多模态提示

对于全量微调的图像提示模型，原有文本到图像能力几乎丧失。然而，借助所提出的 IP-Adapter，我们可使用包括图像提示和文本提示在内的多模态提示生成图像。我们发现该能力在社区模型上表现特别好。在多模态提示的推理阶段，我们调整 $\lambda$ 以平衡图像提示与文本提示。图 8 展示了使用 Realistic Vision V4.0 模型的多模态提示各种结果。如我们所见，可使用额外文本提示生成更多样化的图像。例如，可使用简单文本描述，在图像提示条件下编辑属性并改变主体场景。

我们还将 IP-Adapter 与 Versatile Diffusion、BLIP Diffusion、Uni-ControlNet、T2I-Adapter、ControlNet Shuffle 和 ControlNet Reference-only 等其他方法对比。对比结果如图 9 所示。与其他现有方法相比，我们的方法在多模态提示的图像质量和对齐方面均可生成更优结果。

### 4.4 消融实验

#### 4.4.1 解耦交叉注意力的重要性

为验证解耦交叉注意力策略的有效性，我们还对比了一个不带解耦交叉注意力的简单 Adapter：图像特征与文本特征拼接，然后嵌入预训练交叉注意力层。为公平对比，两种 Adapter 在相同配置下训练 200000 步。图 10 提供了解耦交叉注意力 IP-Adapter 与简单 Adapter 的对比示例。如我们所观察，IP-Adapter 不仅可生成比简单 Adapter 更高质量的图像，且可生成与图像提示更一致的图像。

#### 4.4.2 细粒度特征与全局特征的对比

由于我们的 IP-Adapter 利用 CLIP 图像编码器的全局图像嵌入，它可能丢失参考图像中的一些信息。因此，我们设计了条件于细粒度特征的 IP-Adapter。首先，我们从 CLIP 图像编码器倒数第二层提取 grid 特征。然后，使用小型 query 网络学习特征。具体而言，定义 16 个可学习 token，通过轻量 Transformer 模型从 grid 特征中提取信息。query 网络的 token 特征作为交叉注意力层的输入。

两种 Adapter 的结果如图 11 所示。尽管细粒度特征的 IP-Adapter 可生成与图像提示更一致的图像，但它也可能学习到空间结构信息，这可能降低生成图像的多样性。然而，额外条件（如文本提示和结构图）可与图像提示结合以生成更多样化的图像。例如，我们可在额外人体姿态引导下合成新颖图像。

---

## 5. 结论与未来工作

在本工作中，我们提出 IP-Adapter，为预训练文本到图像扩散模型实现图像提示能力。我们 IP-Adapter 的核心设计基于解耦交叉注意力策略，为图像特征引入独立的交叉注意力层。定量和定性实验结果表明，仅 22M 参数的 IP-Adapter 性能可与部分全量微调图像提示模型及现有 Adapter 相当甚至更优。此外，我们的 IP-Adapter 在仅训练一次后，可直接与从同一基座模型衍生的自定义模型及现有结构可控工具集成，从而扩展其适用性。更重要的是，图像提示可与文本提示结合以实现多模态图像生成。

尽管我们的 IP-Adapter 有效，它只能生成在内容和风格上类似参考图像的图像。换言之，它无法像 Textual Inversion 和 DreamBooth 等某些现有方法那样，合成与给定图像主体高度一致的图像。未来，我们旨在开发更强大的图像提示 Adapter 以增强一致性。
