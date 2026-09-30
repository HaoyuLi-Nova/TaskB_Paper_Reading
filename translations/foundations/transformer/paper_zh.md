# 注意力就是你所需要的一切（Attention Is All You Need）

**作者：** Ashish Vaswani\*、Noam Shazeer\*、Niki Parmar\*、Jakob Uszkoreit\*、Llion Jones\*、Aidan N. Gomez\*$^{\dag}$、Łukasz Kaiser\*、Illia Polosukhin\*$^{\ddagger}$  
（\* 同等贡献。Jakob 提出用自注意力替换 RNN 并启动了评估该想法的工作。Ashish 与 Illia 设计并实现了第一批 Transformer 模型，并深入参与各项工作。Noam 提出缩放点积注意力、多头注意力与无参数位置表示，并几乎参与所有细节。Niki 在原始代码库与 tensor2tensor 中设计、实现、调参并评估了大量模型变体。Llion 也试验了新变体，负责初始代码库以及高效推理与可视化。Łukasz 与 Aidan 长期参与设计并实现 tensor2tensor，替换早期代码库，显著提升结果并极大加速研究。）  
$^{\dag}$ 工作完成于 Google Brain。  
$^{\ddagger}$ 工作完成于 Google Research。  
**机构：** Google Brain / Google Research / University of Toronto  
**会议：** NeurIPS 2017  
**代码：** https://github.com/tensorflow/tensor2tensor  
**原文 TeX：** [`arxiv/foundations/transformer/extracted/ms.tex`](../../../arxiv/foundations/transformer/extracted/ms.tex)

---

## 摘要

当前主导的序列转换（sequence transduction）模型基于包含编码器与解码器的复杂循环或卷积神经网络。表现最好的模型还通过注意力机制连接编码器与解码器。我们提出一种全新的简单网络架构——**Transformer**，**完全基于注意力机制**，彻底摒弃循环与卷积。在两个机器翻译任务上的实验表明，这些模型在质量上更优，同时更易并行化，且训练所需时间显著更少。我们的模型在 WMT 2014 英语到德语翻译任务上达到 **28.4 BLEU**，相对此前最佳结果（含集成模型）提升超过 2 BLEU。在 WMT 2014 英语到法语翻译任务上，我们的模型在八块 GPU 上训练 3.5 天后，确立了新的单模型当时最优 BLEU 分数 **41.8**，训练开销仅为文献中最佳模型的一小部分。我们还表明 Transformer 可很好地泛化到其他任务：在英语成分句法分析上，无论训练数据量大或小，均成功应用。

---

## 1. 引言

循环神经网络，尤其是长短期记忆网络（LSTM）[hochreiter1997] 与门控循环网络 [gruEval14]，已被牢固确立为序列建模与转换问题（如语言建模与机器翻译）[sutskever14, bahdanau2014neural, cho2014learning] 的先进方法。此后大量工作继续推进循环语言模型与编码器–解码器架构的边界 [wu2016google, luong2015effective, jozefowicz2016exploring]。

循环模型通常沿输入与输出序列的符号位置分解计算。将位置与计算时间步对齐后，它们生成隐藏状态序列 $h_t$，其中 $h_t$ 是前一隐藏状态 $h_{t-1}$ 与位置 $t$ 输入的函数。这种固有的顺序性阻碍了训练样本内的并行化；在更长序列上这一点尤为关键，因为内存限制会进一步制约跨样本的批处理。近期工作通过因式分解技巧 [Kuchaiev2017Factorization] 与条件计算 [shazeer2017outrageously] 显著提升了计算效率，后者还提升了模型性能。然而，顺序计算的根本约束仍然存在。

注意力机制已成为多种任务中出色的序列建模与转换模型的组成部分，使模型能够建模与输入或输出序列中距离无关的依赖关系 [bahdanau2014neural, structuredAttentionNetworks]。但除少数例外 [decomposableAttnModel] 外，此类注意力机制几乎总是与循环网络配合使用。

在本文中，我们提出 **Transformer**：一种摒弃循环、转而完全依赖注意力机制以在输入与输出之间建立全局依赖的模型架构。Transformer 允许显著更强的并行化，并在八块 P100 GPU 上仅训练十二小时后，即可在翻译质量上达到新的最先进水平。

---

## 2. 背景

减少顺序计算这一目标，也构成了 Extended Neural GPU [extendedngpu]、ByteNet [NalBytenet2017] 与 ConvS2S [JonasFaceNet2017] 等工作的基础；它们都以卷积神经网络为基本构建模块，并行计算所有输入与输出位置的隐藏表示。在这些模型中，关联任意两个输入或输出位置的信号所需的操作数随位置间距离增长：对 ConvS2S 为线性增长，对 ByteNet 为对数增长。这使得学习远距离位置之间的依赖更加困难 [hochreiter2001gradient]。在 Transformer 中，这一操作数被降为常数，代价是对注意力加权位置求平均可能降低有效分辨率；我们用第 3.2 节所述的**多头注意力（Multi-Head Attention）**来抵消该效应。

**自注意力（self-attention）**，有时也称 **intra-attention**，是一种在单一序列的不同位置之间建立关联、以计算该序列表示的注意力机制。自注意力已成功用于多种任务，包括阅读理解、抽象摘要、文本蕴含，以及学习与任务无关的句子表示 [cheng2016long, decomposableAttnModel, paulus2017deep, lin2017structured]。

端到端记忆网络基于循环注意力机制而非序列对齐的循环，并已表明在简单语言问答与语言建模任务上表现良好 [sukhbaatar2015]。

然而，据我们所知，Transformer 是首个**完全依赖自注意力**来计算其输入与输出表示、而不使用序列对齐 RNN 或卷积的转换模型。在后续各节中，我们将描述 Transformer，论证使用自注意力的动机，并讨论其相对于 Neural GPU、ByteNet 与 ConvS2S 等模型的优势。

---

## 3. 模型架构

![图 1：Transformer——模型架构。](../../../arxiv/foundations/transformer/extracted/Figures/ModalNet-21.png)

*图 1：Transformer——模型架构。*

大多数竞争性神经序列转换模型具有编码器–解码器结构 [cho2014learning, bahdanau2014neural, sutskever14]。其中，编码器将符号表示的输入序列 $(x_1, \ldots, x_n)$ 映射为连续表示序列 $\mathbf{z} = (z_1, \ldots, z_n)$。给定 $\mathbf{z}$，解码器随后一次一个元素地生成符号输出序列 $(y_1, \ldots, y_m)$。在每一步，模型都是自回归的 [graves2013generating]：在生成下一个符号时，将先前已生成的符号作为额外输入。

Transformer 遵循这一总体架构，对编码器与解码器均使用堆叠的自注意力层与逐位置全连接层，分别如图 1 左半与右半所示。

### 3.1 编码器与解码器堆栈

**编码器：** 编码器由 $N=6$ 个相同层堆叠而成。每层有两个子层。第一个是多头自注意力机制，第二个是简单的逐位置全连接前馈网络。我们在两个子层中的每一个周围使用残差连接 [he2016deep]，后接层归一化 [layernorm2016]。也就是说，每个子层的输出为 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$，其中 $\mathrm{Sublayer}(x)$ 是该子层自身实现的函数。为便于这些残差连接，模型中所有子层以及嵌入层的输出维度均为 $d_{\text{model}}=512$。

**解码器：** 解码器同样由 $N=6$ 个相同层堆叠而成。除编码器每层中的两个子层外，解码器还插入第三个子层，对编码器堆栈的输出执行多头注意力。与编码器类似，我们在每个子层周围使用残差连接，后接层归一化。我们还修改解码器堆栈中的自注意力子层，以防止各位置关注到后续位置。该掩码与输出嵌入右移一个位置这一事实相结合，确保位置 $i$ 的预测只能依赖于位置小于 $i$ 的已知输出。

### 3.2 注意力

注意力函数可描述为：将一个 query 与一组 key-value 对映射到一个输出，其中 query、key、value 与输出均为向量。输出计算为 value 的加权和，分配给每个 value 的权重由 query 与对应 key 的相容性函数计算得到。

#### 3.2.1 缩放点积注意力（Scaled Dot-Product Attention）

我们将我们的特定注意力称为**缩放点积注意力**（图 2 左）。输入由维度为 $d_k$ 的 query 与 key，以及维度为 $d_v$ 的 value 组成。我们计算 query 与所有 key 的点积，各自除以 $\sqrt{d_k}$，再应用 softmax 函数得到 value 上的权重。

实践中，我们同时对一组 query 计算注意力函数，并将它们打包为矩阵 $Q$。key 与 value 也分别打包为矩阵 $K$ 与 $V$。输出矩阵计算为：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

最常用的两种注意力函数是加性注意力 [bahdanau2014neural] 与点积（乘性）注意力。点积注意力与我们的算法相同，只是没有缩放因子 $\frac{1}{\sqrt{d_k}}$。加性注意力用带单个隐层的前馈网络计算相容性函数。二者在理论复杂度上相近，但点积注意力在实践中更快且更省空间，因为它可用高度优化的矩阵乘法代码实现。

当 $d_k$ 较小时，两种机制表现相近；而对较大的 $d_k$，不加缩放的点积注意力不如加性注意力 [BritzGLL17]。我们怀疑：当 $d_k$ 很大时，点积的幅度会变得很大，从而把 softmax 函数推入梯度极小的区域。[^scale] 为抵消该效应，我们将点积按 $\frac{1}{\sqrt{d_k}}$ 缩放。

[^scale]: 为说明点积为何会变大，假设 $q$ 与 $k$ 的各分量是均值为 0、方差为 1 的独立随机变量。则其点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0、方差为 $d_k$。

#### 3.2.2 多头注意力（Multi-Head Attention）

<p align="center">
  <img src="../../../arxiv/foundations/transformer/extracted/Figures/ModalNet-19.png" alt="缩放点积注意力" width="45%"/>
  <img src="../../../arxiv/foundations/transformer/extracted/Figures/ModalNet-20.png" alt="多头注意力" width="45%"/>
</p>

*图 2：（左）缩放点积注意力。（右）多头注意力由若干并行运行的注意力层组成。*

我们发现：与其用 $d_{\text{model}}$ 维的 key、value 与 query 执行单一注意力函数，不如将 query、key 与 value 分别用不同的、可学习的线性投影投影 $h$ 次，投影到 $d_k$、$d_k$ 与 $d_v$ 维。然后在这些投影后的 query、key 与 value 上并行执行注意力函数，得到 $d_v$ 维的输出 value。将这些输出拼接后再投影一次，得到最终值，如图 2 右所示。

多头注意力使模型能够在不同位置、从不同表示子空间**联合**关注信息。若只有单个注意力头，平均会抑制这一点。

$$
\begin{aligned}
\mathrm{MultiHead}(Q, K, V) &= \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)W^O \\
\text{其中}\quad \mathrm{head}_i &= \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}
$$

其中投影为参数矩阵 $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$，以及 $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$。

在本工作中，我们使用 $h=8$ 个并行注意力层（即头）。对每一个头，我们使用 $d_k=d_v=d_{\text{model}}/h=64$。由于每个头的维度降低，总计算代价与全维度的单头注意力相近。

#### 3.2.3 注意力在我们模型中的应用

Transformer 以三种不同方式使用多头注意力：

- 在**编码器–解码器注意力**层中，query 来自上一层解码器，而记忆的 key 与 value 来自编码器的输出。这使得解码器中的每个位置都能关注输入序列中的所有位置。这模仿了序列到序列模型中典型的编码器–解码器注意力机制 [wu2016google, bahdanau2014neural, JonasFaceNet2017]。

- 编码器包含自注意力层。在自注意力层中，所有 key、value 与 query 来自同一处，此处即编码器中上一层的输出。编码器中的每个位置都可以关注编码器上一层中的所有位置。

- 类似地，解码器中的自注意力层允许解码器中的每个位置关注到该位置及之前的所有解码器位置。为保持自回归性质，我们需要阻止解码器中的向左信息流。我们在缩放点积注意力内部通过掩码实现这一点：将 softmax 输入中对应于非法连接的值设为 $-\infty$。见图 2。

### 3.3 逐位置前馈网络

除注意力子层外，编码器与解码器的每一层还包含一个全连接前馈网络，它被**分别且相同地**应用于每个位置。它由两次线性变换组成，中间夹一个 ReLU 激活：

$$
\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2
$$

虽然不同位置上的线性变换相同，但层与层之间使用不同参数。另一种描述方式是：两个核大小为 1 的卷积。输入与输出的维度为 $d_{\text{model}}=512$，内层维度为 $d_{\text{ff}}=2048$。

### 3.4 Embedding 与 Softmax

与其他序列转换模型类似，我们使用可学习的 embedding，将输入 token 与输出 token 转换为维度为 $d_{\text{model}}$ 的向量。我们也使用通常的可学习线性变换与 softmax 函数，将解码器输出转换为预测的下一 token 概率。在我们的模型中，我们在两个 embedding 层与 pre-softmax 线性变换之间共享同一权重矩阵，做法类似于 [press2016using]。在 embedding 层中，我们将这些权重乘以 $\sqrt{d_{\text{model}}}$。

### 3.5 位置编码

由于我们的模型不含循环也不含卷积，为使模型能够利用序列的顺序，我们必须向序列中 token 的相对或绝对位置注入一些信息。为此，我们在编码器与解码器堆栈底部，将**位置编码**加到输入 embedding 上。位置编码与 embedding 具有相同的维度 $d_{\text{model}}$，以便二者可以相加。位置编码有许多选择，包括可学习的与固定的 [JonasFaceNet2017]。

在本工作中，我们使用不同频率的正弦与余弦函数：

$$
\begin{aligned}
PE_{(pos,2i)} &= \sin\!\left(pos / 10000^{2i/d_{\text{model}}}\right) \\
PE_{(pos,2i+1)} &= \cos\!\left(pos / 10000^{2i/d_{\text{model}}}\right)
\end{aligned}
$$

其中 $pos$ 是位置，$i$ 是维度。也就是说，位置编码的每个维度对应一个正弦曲线。波长构成从 $2\pi$ 到 $10000\cdot 2\pi$ 的几何级数。我们选择该函数是因为我们假设它将使模型容易学会按相对位置进行注意力：对任意固定偏移 $k$，$PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性函数。

我们还试验了改用可学习的位置 embedding [JonasFaceNet2017]，并发现两个版本产生几乎相同的结果（见表 3 第 (E) 行）。我们选择正弦版本，是因为它可能使模型外推到比训练中遇到的更长的序列长度。

---

## 4. 为何使用自注意力

在本节中，我们将自注意力层与常用于把变长符号表示序列 $(x_1, \ldots, x_n)$ 映射为等长序列 $(z_1, \ldots, z_n)$（其中 $x_i, z_i \in \mathbb{R}^d$）的循环层与卷积层进行比较，例如典型序列转换编码器或解码器中的隐藏层。为论证使用自注意力，我们考虑三个期望性质。

其一是**每层的总计算复杂度**。  
其二是**可并行化的计算量**，以所需的最少顺序操作数衡量。  
其三是网络中长程依赖之间的**路径长度**。学习长程依赖是许多序列转换任务中的关键挑战。影响学习此类依赖能力的一个关键因素，是前向与反向信号在网络中必须穿越的路径长度。输入与输出序列中任意位置组合之间的这些路径越短，就越容易学习长程依赖 [hochreiter2001gradient]。因此，我们还比较由不同类型层组成的网络中、任意两个输入与输出位置之间的最大路径长度。

**表 1：** 不同类型层的最大路径长度、每层复杂度与最少顺序操作数。$n$ 为序列长度，$d$ 为表示维度，$k$ 为卷积核大小，$r$ 为受限自注意力中的邻域大小。

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| --- | --- | --- | --- |
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

如表 1 所示，自注意力层用常数次顺序执行的操作连接所有位置，而循环层需要 $O(n)$ 次顺序操作。在计算复杂度方面，当序列长度 $n$ 小于表示维度 $d$ 时，自注意力层快于循环层；而在机器翻译中先进模型所用的句子表示（如 word-piece [wu2016google] 与 byte-pair [sennrich2015neural] 表示）上，这一点最为常见。为提升涉及很长序列任务的计算性能，可将自注意力限制为只考虑以相应输出位置为中心、大小为 $r$ 的输入邻域。这将使最大路径长度增加到 $O(n/r)$。我们计划在未来工作中进一步研究该方法。

单个核宽 $k < n$ 的卷积层并不能连接所有输入与输出位置对。要做到这一点，在连续核的情况下需要堆叠 $O(n/k)$ 层卷积，在扩张卷积的情况下需要 $O(\log_k(n))$ 层 [NalBytenet2017]，从而增加网络中任意两位置之间最长路径的长度。卷积层通常比循环层更昂贵，约为其 $k$ 倍。不过，可分离卷积 [xception2016] 可将复杂度显著降为 $O(k \cdot n \cdot d + n \cdot d^2)$。即便取 $k=n$，可分离卷积的复杂度也等于自注意力层与逐点前馈层的组合——而这正是我们模型采用的方法。

作为附带好处，自注意力可能产生更可解释的模型。我们检查了模型中的注意力分布，并在附录中给出并讨论了示例。不仅各个注意力头明显学到执行不同任务，许多头似乎还表现出与句子句法与语义结构相关的行为。

---

## 5. 训练

本节描述我们模型的训练设定。

### 5.1 训练数据与批处理

我们在标准的 WMT 2014 英语–德语数据集上训练，该数据集约含 450 万句对。句子使用 byte-pair encoding [BritzGLL17] 编码，源–目标共享词表约 37000 个 token。对于英语–法语，我们使用显著更大的 WMT 2014 英语–法语数据集，含 3600 万句，并将 token 切分为 32000 的 word-piece 词表 [wu2016google]。句对按近似序列长度批处理在一起。每个训练 batch 包含一组句对，约含 25000 个源 token 与 25000 个目标 token。

### 5.2 硬件与进度安排

我们在一台配有 8 块 NVIDIA P100 GPU 的机器上训练模型。对使用全文所述超参数的 base 模型，每一步训练约需 0.4 秒。我们共训练 base 模型 100,000 步，即 12 小时。对我们的 big 模型（见表 3 底行），每步时间为 1.0 秒。big 模型训练了 300,000 步（3.5 天）。

### 5.3 优化器

我们使用 Adam 优化器 [kingma2014adam]，其中 $\beta_1=0.9$，$\beta_2=0.98$，$\epsilon=10^{-9}$。我们在训练过程中按以下公式变化学习率：

$$
\mathrm{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\big(\mathrm{step\_num}^{-0.5},\; \mathrm{step\_num} \cdot \mathrm{warmup\_steps}^{-1.5}\big)
$$

这对应于：在前 $\mathrm{warmup\_steps}$ 个训练步中线性增加学习率，之后按步数平方根的反比下降。我们使用 $\mathrm{warmup\_steps}=4000$。

### 5.4 正则化

训练中我们采用以下正则化：

**残差 Dropout**　我们在每个子层的输出上、在它被加到子层输入并归一化之前应用 dropout [srivastava2014dropout]。此外，我们还对编码器与解码器堆栈中 embedding 与位置编码之和应用 dropout。对 base 模型，我们使用比率 $P_{\mathrm{drop}}=0.1$。

**标签平滑**　训练期间，我们采用值为 $\epsilon_{ls}=0.1$ 的标签平滑 [SzegedyVISW15]。这会损害困惑度，因为模型学会更加不确定，但会提升准确率与 BLEU 分数。

---

## 6. 实验结果

### 6.1 机器翻译

**表 2：** Transformer 在英语到德语与英语到法语 newstest2014 测试上取得优于此前最先进模型的 BLEU，且训练开销仅为其一小部分。

| Model | BLEU<br>EN-DE | BLEU<br>EN-FR | Training Cost (FLOPs)<br>EN-DE | Training Cost (FLOPs)<br>EN-FR |
| --- | ---: | ---: | ---: | ---: |
| ByteNet | 23.75 | | | |
| Deep-Att + PosUnk | | 39.2 | | $1.0\cdot10^{20}$ |
| GNMT + RL | 24.6 | 39.92 | $2.3\cdot10^{19}$ | $1.4\cdot10^{20}$ |
| ConvS2S | 25.16 | 40.46 | $9.6\cdot10^{18}$ | $1.5\cdot10^{20}$ |
| MoE | 26.03 | 40.56 | $2.0\cdot10^{19}$ | $1.2\cdot10^{20}$ |
| Deep-Att + PosUnk Ensemble | | 40.4 | | $8.0\cdot10^{20}$ |
| GNMT + RL Ensemble | 26.30 | 41.16 | $1.8\cdot10^{20}$ | $1.1\cdot10^{21}$ |
| ConvS2S Ensemble | 26.36 | **41.29** | $7.7\cdot10^{19}$ | $1.2\cdot10^{21}$ |
| Transformer (base model) | 27.3 | 38.1 | **$3.3\cdot10^{18}$** | **$3.3\cdot10^{18}$** |
| Transformer (big) | **28.4** | **41.8** | $2.3\cdot10^{19}$ | $2.3\cdot10^{19}$ |

（原文表中 Transformer 两行将 EN-DE / EN-FR 的 Training Cost 合并为一格；此处两列填同一数值以对应原文合并单元格。）

在 WMT 2014 英语到德语翻译任务上，big transformer 模型（表 2 中的 Transformer (big)）比此前报告的最佳模型（含集成）高出超过 $2.0$ BLEU，确立了新的最先进 BLEU 分数 $28.4$。该模型配置列于表 3 底行。训练在 $8$ 块 P100 GPU 上耗时 $3.5$ 天。即便我们的 base 模型也超越了此前所有已发表的模型与集成，且训练开销仅为任一竞争模型的零头。

在 WMT 2014 英语到法语翻译任务上，我们的 big 模型达到 BLEU $41.8$，超越此前所有已发表的单模型，训练开销不足此前最先进模型的 $1/4$。用于英语到法语的 Transformer (big) 使用 dropout 比率 $P_{\mathrm{drop}}=0.1$，而非 $0.3$。

对 base 模型，我们使用将最后 5 个检查点（每 10 分钟写入一次）平均后得到的单一模型。对 big 模型，我们平均最后 20 个检查点。我们使用束搜索，束宽为 $4$，长度惩罚 $\alpha=0.6$ [wu2016google]。这些超参数在开发集上实验后选定。推理时我们将最大输出长度设为输入长度 $+\,50$，但在可能时提前终止 [wu2016google]。

表 2 总结了我们的结果，并将翻译质量与训练开销与文献中其他模型架构进行对比。我们通过将训练时间、所用 GPU 数量与每块 GPU 持续单精度浮点能力的估计相乘，来估计训练模型所用的浮点操作数。[^flops]

[^flops]: 我们对 K80、K40、M40 与 P100 分别使用 2.8、3.7、6.0 与 9.5 TFLOPS。

### 6.2 模型变体

**表 3：** Transformer 架构上的变体。未列出的取值与 base 模型相同。所有指标均在英语到德语翻译开发集 newstest2013 上。所列困惑度为按我们的 byte-pair encoding 的每 wordpiece 困惑度，不应与每词困惑度比较。

| | $N$ | $d_{\text{model}}$ | $d_{\text{ff}}$ | $h$ | $d_k$ | $d_v$ | $P_{\mathrm{drop}}$ | $\epsilon_{ls}$ | train steps | PPL (dev) | BLEU (dev) | params $\times10^6$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.8 | 65 |
| (A) | | | | 1 | 512 | 512 | | | | 5.29 | 24.9 | |
| | | | | 4 | 128 | 128 | | | | 5.00 | 25.5 | |
| | | | | 16 | 32 | 32 | | | | 4.91 | 25.8 | |
| | | | | 32 | 16 | 16 | | | | 5.01 | 25.4 | |
| (B) | | | | | 16 | | | | | 5.16 | 25.1 | 58 |
| | | | | | 32 | | | | | 5.01 | 25.4 | 60 |
| (C) | 2 | | | | | | | | | 6.11 | 23.7 | 36 |
| | 4 | | | | | | | | | 5.19 | 25.3 | 50 |
| | 8 | | | | | | | | | 4.88 | 25.5 | 80 |
| | | 256 | | | 32 | 32 | | | | 5.75 | 24.5 | 28 |
| | | 1024 | | | 128 | 128 | | | | 4.66 | 26.0 | 168 |
| | | | 1024 | | | | | | | 5.12 | 25.4 | 53 |
| | | | 4096 | | | | | | | 4.75 | 26.2 | 90 |
| (D) | | | | | | | 0.0 | | | 5.77 | 24.6 | |
| | | | | | | | 0.2 | | | 4.95 | 25.5 | |
| | | | | | | | | 0.0 | | 4.67 | 25.3 | |
| | | | | | | | | 0.2 | | 5.47 | 25.7 | |
| (E) | | | | | | | | | | 4.92 | 25.7 | |
| big | 6 | 1024 | 4096 | 16 | | | 0.3 | | 300K | **4.33** | **26.4** | 213 |

（表 3 第 (E) 行原文中间各超参数列合并标注为：*positional embedding instead of sinusoids*；其余未列出的取值与 base 相同。）

为评估 Transformer 不同组件的重要性，我们以不同方式改变 base 模型，并在英语到德语翻译开发集 newstest2013 上测量性能变化。我们使用上一节所述的束搜索，但不做检查点平均。结果见表 3。

在表 3 第 (A) 行中，我们改变注意力头数以及注意力 key 与 value 维度，并如第 3.2.2 节所述保持计算量恒定。单头注意力比最佳设置差 0.9 BLEU；头数过多时质量也会下降。

在表 3 第 (B) 行中，我们观察到减小注意力 key 大小 $d_k$ 会损害模型质量。这表明判断相容性并不容易，比点积更复杂的相容性函数可能有益。我们进一步在第 (C)、(D) 行观察到：正如预期，更大的模型更好，且 dropout 对避免过拟合非常有帮助。在第 (E) 行中，我们用可学习位置 embedding [JonasFaceNet2017] 替换正弦位置编码，并观察到与 base 模型几乎相同的结果。

### 6.3 英语成分句法分析

**表 4：** Transformer 很好地泛化到英语成分句法分析（结果在 WSJ 第 23 节上）。

| Parser | Training | WSJ 23 F1 |
| --- | --- | ---: |
| Vinyals & Kaiser et al. (2014) | WSJ only, discriminative | 88.3 |
| Petrov et al. (2006) | WSJ only, discriminative | 90.4 |
| Zhu et al. (2013) | WSJ only, discriminative | 90.4 |
| Dyer et al. (2016) | WSJ only, discriminative | 91.7 |
| Transformer (4 layers) | WSJ only, discriminative | 91.3 |
| Zhu et al. (2013) | semi-supervised | 91.3 |
| Huang & Harper (2009) | semi-supervised | 91.3 |
| McClosky et al. (2006) | semi-supervised | 92.1 |
| Vinyals & Kaiser et al. (2014) | semi-supervised | 92.1 |
| Transformer (4 layers) | semi-supervised | 92.7 |
| Luong et al. (2015) | multi-task | 93.0 |
| Dyer et al. (2016) | generative | 93.3 |

为评估 Transformer 能否泛化到其他任务，我们在英语成分句法分析上进行了实验。该任务带来特定挑战：输出受强结构约束，且显著长于输入。此外，RNN 序列到序列模型在小数据设定下未能达到最先进结果 [KVparse15]。

我们在 Penn Treebank 的 Wall Street Journal（WSJ）部分 [marcus1993building] 上训练一个 4 层、$d_{\text{model}}=1024$ 的 transformer，约 40K 训练句。我们还在半监督设定下训练它，使用来自约 1700 万句的更大 high-confidence 与 BerkleyParser 语料 [KVparse15]。WSJ-only 设定使用 16K token 词表，半监督设定使用 32K token 词表。

我们仅进行少量实验，在第 22 节开发集上选择 dropout（注意力与残差，见第 5.4 节）、学习率与束宽；所有其他参数保持与英语到德语 base 翻译模型不变。推理时，我们将最大输出长度增至输入长度 $+\,300$。对 WSJ-only 与半监督设定，我们均使用束宽 $21$ 与 $\alpha=0.3$。

表 4 中的结果表明：尽管缺少针对任务的调参，我们的模型表现得出奇地好，优于此前报告的所有模型，仅 Recurrent Neural Network Grammar [dyer-rnng:16] 除外。

与 RNN 序列到序列模型 [KVparse15] 不同，即便仅在 40K 句的 WSJ 训练集上训练，Transformer 也优于 BerkeleyParser [petrov-EtAl:2006:ACL]。

---

## 7. 结论

在本工作中，我们提出了 Transformer——首个**完全基于注意力**的序列转换模型，用多头自注意力替换编码器–解码器架构中最常用的循环层。

对翻译任务，Transformer 的训练可显著快于基于循环或卷积层的架构。在 WMT 2014 英语到德语与 WMT 2014 英语到法语翻译任务上，我们都达到了新的最先进水平。在前者任务上，我们的最佳模型甚至超越了此前报告的所有集成结果。

我们对基于注意力的模型的未来感到兴奋，并计划将其应用于其他任务。我们计划将 Transformer 扩展到文本以外的输入与输出模态，并研究局部、受限的注意力机制，以高效处理图像、音频与视频等大型输入与输出。使生成更少顺序化是我们的另一研究目标。

我们用于训练与评估模型的代码见：https://github.com/tensorflow/tensor2tensor。

**致谢**　我们感谢 Nal Kalchbrenner 与 Stephan Gouws 富有成效的评论、指正与启发。

---

## 附录：注意力可视化

![附录图：编码器第 5 层（共 6 层）自注意力中的长距离依赖示例。许多注意力头关注动词 making 的远距离依存，完成短语 making…more difficult。此处仅展示对词 making 的注意力。不同颜色表示不同头。建议彩色查看。](../../../arxiv/foundations/transformer/extracted/vis/making_more_difficult5_new.png)

*图：编码器第 5 层（共 6 层）自注意力中跟随长距离依赖的注意力机制示例。许多注意力头关注动词 `making` 的远距离依存，完成短语 `making...more difficult`。此处仅展示对词 `making` 的注意力。不同颜色表示不同头。*

<p align="center">
  <img src="../../../arxiv/foundations/transformer/extracted/vis/anaphora_resolution_new.png" width="48%"/>
  <img src="../../../arxiv/foundations/transformer/extracted/vis/anaphora_resolution2_new.png" width="48%"/>
</p>

*图：同样在第 5 层（共 6 层）的两个注意力头，似乎参与指代消解。上：头 5 的完整注意力。下：仅从词 `its` 出发、注意力头 5 与 6 的注意力。注意对该词注意力非常尖锐。*

<p align="center">
  <img src="../../../arxiv/foundations/transformer/extracted/vis/attending_to_head_new.png" width="48%"/>
  <img src="../../../arxiv/foundations/transformer/extracted/vis/attending_to_head2_new.png" width="48%"/>
</p>

*图：许多注意力头表现出似乎与句子结构相关的行为。上图给出两个例子，分别来自编码器第 5 层（共 6 层）自注意力中的两个不同头。这些头明显学到了执行不同任务。*

---

*本译文依据 `arxiv/foundations/transformer/extracted/` 下 `ms.tex` 及其 `\input{}` 分章，按原文结构与表述翻译整理；插图见 `Figures/` 与 `vis/`。表格栏目与数值与原文一致。*
