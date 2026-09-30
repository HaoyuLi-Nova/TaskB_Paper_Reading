# 视觉表征对比学习的简单框架（A Simple Framework for Contrastive Learning of Visual Representations）

**作者：** Ting Chen、Simon Kornblith、Mohammad Norouzi、Geoffrey Hinton
**机构：** Google Research, Brain Team
**会议：** ICML 2020
**代码：** https://github.com/google-research/simclr
**原文 TeX：** [`arxiv/foundations/simclr/extracted/main.tex`](../../../arxiv/foundations/simclr/extracted/main.tex)

---

## 摘要

本文提出 **SimCLR**：一个用于视觉表征对比学习的**简单框架**。我们简化了近期提出的对比自监督学习算法，**不需要**特殊架构或 memory bank。为理解对比预测任务为何能学到有用表征，我们系统研究了框架的主要组件，并表明：

1. **数据增强的组合**对定义有效预测任务至关重要；
2. 在表征与对比损失之间引入**可学习的非线性变换**显著提升所学表征的质量；
3. 相对监督学习，对比学习更受益于**更大 batch size** 与**更多训练步数**。

综合这些发现，SimCLR 在 ImageNet 自监督与半监督学习上大幅超过此前方法。在自监督表征上训练的线性分类器达到 **76.5%** top-1 准确率，相对此前最优提升约 7%，匹配监督 ResNet-50；仅用 **1%** 标签微调时达到 **85.8%** top-5 准确率，以约少 100× 的标签超过 AlexNet。

---

## 1. 引言

无人工监督地学习有效视觉表征是一个长期存在的问题。主流方法大致分为两类：**生成式**或**判别式**。生成式方法学习在输入空间中生成或建模像素；然而，像素级生成计算昂贵，且对表征学习未必必要。判别式方法使用与监督学习类似的目标函数学习表征，但训练网络执行代理任务（pretext tasks），其输入与标签均来自无标注数据集。许多此类方法依赖启发式设计代理任务，可能限制所学表征的通用性。基于**潜空间对比学习**的判别方法近期表现出巨大潜力，并取得了最优结果。

![图 1：不同自监督方法所学表征上训练的线性分类器在 ImageNet 上的 Top-1 准确率（均在 ImageNet 上预训练）。灰色叉表示监督 ResNet-50；本文方法 SimCLR 以粗体标出。](../../../arxiv/foundations/simclr/extracted/figures/sota_figure.png)

本文提出一个用于视觉表征对比学习的简单框架，称为 **SimCLR**。SimCLR 不仅超过先前工作（图 1），而且更简单——既不需要特殊架构，也不需要 memory bank。

为理解何种因素使对比表征学习有效，我们系统研究框架的主要组件，并表明：

- 多种数据增强操作的**组合**对定义能产生有效表征的对比预测任务至关重要；此外，无监督对比学习比监督学习更受益于**更强**的数据增强。
- 在表征与对比损失之间引入**可学习的非线性变换**显著提升所学表征的质量。
- 使用对比交叉熵损失的表征学习受益于**归一化嵌入**与适当调节的**温度参数**。
- 相对其监督对应方法，对比学习更受益于**更大 batch size** 与**更长训练**；与监督学习类似，对比学习也受益于更深、更宽的网络。

我们综合这些发现，在 ImageNet ILSVRC-2012 上取得自监督与半监督学习的新最优结果。在线性评估协议下，SimCLR 达到 76.5% top-1 准确率，相对此前最优相对提升 7%。仅用 1% ImageNet 标签微调时，SimCLR 达到 85.8% top-5 准确率，相对提升 10%。在其他自然图像分类数据集上微调时，SimCLR 在 12 个数据集中的 10 个上达到或超过强监督基线。

---

## 2. 方法

### 2.1 对比学习框架

受近期对比学习算法启发（相关工作见第 7 节），SimCLR 通过在潜空间中用对比损失最大化同一数据样本不同增强视图之间的一致性来学习表征。如图 2 所示，该框架包含以下四个主要组件：

1. **随机数据增强模块：** 对任意给定数据样本随机变换，得到同一样本的两个相关视图，记为 $\tilde{\bm x}_i$ 与 $\tilde{\bm x}_j$，并将其视为正样本对。本文依次施加三种简单增强：**随机裁剪**后再 resize 回原始尺寸、**随机颜色扰动**、以及**随机高斯模糊**。如第 3 节所示，随机裁剪与颜色扰动的组合对取得良好性能至关重要。
2. **基编码器（base encoder）$f(\cdot)$：** 从增强样本提取表征向量。本框架允许各种网络架构选择而无约束。为简洁起见，我们采用常用的 ResNet，得到 $\bm h_i=f(\tilde{\bm x}_i)=\mathrm{ResNet}(\tilde{\bm x}_i)$，其中 $\bm h_i\in\mathbb{R}^d$ 为平均池化层之后的输出。
3. **投影头（projection head）$g(\cdot)$：** 将表征映射到施加对比损失的空间。我们使用带一层隐层的 MLP，得到 $\bm z_i=g(\bm h_i)=W^{(2)}\sigma(W^{(1)}\bm h_i)$，其中 $\sigma$ 为 ReLU 非线性。如第 4 节所示，我们发现在 $\bm z_i$ 上而非 $\bm h_i$ 上定义对比损失是有益的。
4. **对比损失函数：** 为对比预测任务定义。给定包含正对样本 $\tilde{\bm x}_i$ 与 $\tilde{\bm x}_j$ 的集合 $\{\tilde{\bm x}_k\}$，对比预测任务的目标是：对给定的 $\tilde{\bm x}_i$，从 $\{\tilde{\bm x}_k\}_{k\neq i}$ 中识别出 $\tilde{\bm x}_j$。

**图 2：** 视觉表征对比学习的简单框架。从同一增强族采样两个独立的数据增强算子（$t\sim\mathcal{T}$ 与 $t'\sim\mathcal{T}$），并施加到每个数据样本以得到两个相关视图。基编码器网络 $f(\cdot)$ 与投影头 $g(\cdot)$ 用对比损失训练以最大化一致性。训练结束后丢弃投影头 $g(\cdot)$，用编码器 $f(\cdot)$ 与表征 $\bm h$ 做下游任务。

```
x ──t∼𝒯──→ x̃_i ──f(·)──→ h_i ──g(·)──→ z_i ─┐
                                              ├─ Maximize agreement
x ──t'∼𝒯─→ x̃_j ──f(·)──→ h_j ──g(·)──→ z_j ─┘
```

我们随机采样大小为 $N$ 的 minibatch，并在由该 minibatch 导出的增强样本对上定义对比预测任务，从而得到 $2N$ 个数据点。我们**不显式采样负样本**。相反，给定一个正对，我们将 minibatch 内其余 $2(N-1)$ 个增强样本视为负样本。令 $\mathrm{sim}(\bm u,\bm v)=\bm u^\top\bm v/\lVert\bm u\rVert\lVert\bm v\rVert$ 表示 $\ell_2$ 归一化后 $\bm u$ 与 $\bm v$ 的点积（即余弦相似度）。则正对样本 $(i,j)$ 的损失函数定义为：

$$
\ell_{i,j}=-\log\frac{\exp(\mathrm{sim}(\bm z_i,\bm z_j)/\tau)}{\sum_{k=1}^{2N}\mathbbm{1}_{[k\neq i]}\exp(\mathrm{sim}(\bm z_i,\bm z_k)/\tau)}
$$

其中 $\mathbbm{1}_{[k\neq i]}\in\{0,1\}$ 为指示函数（当且仅当 $k\neq i$ 时为 1），$\tau$ 为温度参数。最终损失对 mini-batch 内所有正对 $(i,j)$ 与 $(j,i)$ 计算。该损失已用于先前工作；为方便起见，我们称其为 **NT-Xent**（normalized temperature-scaled cross entropy loss，归一化温度缩放交叉熵损失）。

算法 1 总结了所提方法。

**算法 1：SimCLR 主学习算法**

```
输入: batch size N, 常数 τ, 结构 f, g, 𝒯
for 采样的 minibatch {x_k}_{k=1}^N do
  for all k ∈ {1, …, N} do
    采样两个增强函数 t ∼ 𝒯, t' ∼ 𝒯
    # 第一个增强
    x̃_{2k−1} = t(x_k)
    h_{2k−1} = f(x̃_{2k−1})          # 表征
    z_{2k−1} = g(h_{2k−1})          # 投影
    # 第二个增强
    x̃_{2k} = t'(x_k)
    h_{2k} = f(x̃_{2k})              # 表征
    z_{2k} = g(h_{2k})              # 投影
  end for
  for all i ∈ {1, …, 2N} and j ∈ {1, …, 2N} do
    s_{i,j} = z_iᵀ z_j / (‖z_i‖ ‖z_j‖)   # 两两相似度
  end for
  定义 ℓ(i, j) 为
    ℓ(i, j) = −log [ exp(s_{i,j}/τ) / Σ_{k=1}^{2N} 𝟙_{[k≠i]} exp(s_{i,k}/τ) ]
  ℒ = (1/(2N)) Σ_{k=1}^N [ ℓ(2k−1, 2k) + ℓ(2k, 2k−1) ]
  更新网络 f 与 g 以最小化 ℒ
end for
返回编码器网络 f(·)，并丢弃 g(·)
```

### 2.2 大 Batch 训练

为保持简单，我们不用 memory bank 训练模型。相反，我们将训练 batch size $N$ 从 256 变化到 8192。batch size 为 8192 时，每个正对从两个增强视图共得到 16382 个负样本。使用标准 SGD/Momentum 与线性学习率缩放时，大 batch 训练可能不稳定。为稳定训练，我们对所有 batch size 使用 **LARS** 优化器。我们在 Cloud TPU 上训练模型，根据 batch size 使用 32 到 128 个核。（脚注：使用 128 个 TPU v3 核时，以 batch size 4096 训练 ResNet-50 共 100 epoch 约需 1.5 小时。）

**Global BN。** 标准 ResNet 使用批归一化（BN）。在数据并行的分布式训练中，BN 均值与方差通常按设备局部聚合。在我们的对比学习中，由于正对在同一设备上计算，模型可能利用局部信息泄漏来提高预测准确率，却不提升表征。我们通过在训练期间**在所有设备上聚合 BN 均值与方差**来解决该问题。其他做法包括跨设备打乱数据样本，或用 layer norm 替换 BN。

### 2.3 评估协议

此处给出实证研究的协议，旨在理解框架中的不同设计选择。

**数据集与指标。** 我们大多数关于无监督预训练（无标签学习编码器网络 $f$）的研究使用 ImageNet ILSVRC-2012 数据集。CIFAR-10 上的部分额外预训练实验见附录 B.9。我们也在多种数据集上测试预训练结果以进行迁移学习。为评估所学表征，我们遵循广泛使用的**线性评估协议**：在冻结的基网络之上训练线性分类器，并以测试准确率作为表征质量的代理。除线性评估外，我们也与半监督及迁移学习上的最优方法比较。

**默认设置。** 除非另有说明，数据增强使用随机裁剪与 resize（含随机翻转）、颜色扰动与高斯模糊（细节见附录 A）。我们使用 ResNet-50 作为基编码器网络，以及将表征投影到 128 维潜空间的 2 层 MLP 投影头。损失使用 NT-Xent，用 LARS 优化，学习率为 $4.8(=0.3\times\mathrm{BatchSize}/256)$，权重衰减为 $10^{-6}$。我们以 batch size 4096 训练 100 epoch。（脚注：虽在 100 epoch 未达最高性能，但已取得合理结果，从而允许公平且高效的消融。）此外，前 10 epoch 使用线性 warmup，之后用无重启的余弦衰减调度衰减学习率。

---

## 3. 对比表征学习中的数据增强

数据增强定义了对比预测任务。简单的随机裁剪（含 resize 与翻转）即可创建涵盖**全局→局部视图**与**相邻视图**预测的对比预测任务族，从而将预测任务与网络架构解耦。

<p align="center">
  <img src="../../../arxiv/foundations/simclr/extracted/figures/img_original.png" alt="Original" width="18%"/>
  <img src="../../../arxiv/foundations/simclr/extracted/figures/img_crop.png" alt="Crop" width="18%"/>
  <img src="../../../arxiv/foundations/simclr/extracted/figures/img_color.png" alt="Color" width="18%"/>
  <img src="../../../arxiv/foundations/simclr/extracted/figures/img_gblur.png" alt="Blur" width="18%"/>
  <img src="../../../arxiv/foundations/simclr/extracted/figures/img_sobel.png" alt="Sobel" width="18%"/>
</p>

**图 3：** 所研究的数据增强算子示意。每种增强均可按内部参数随机变换数据（例如旋转角度、噪声水平）。注意：我们**仅在消融中测试**这些算子；**用于训练我们模型的增强策略**仅包括**随机裁剪（含翻转与 resize）**、**颜色扰动**与**高斯模糊**。子图包括：Original；Crop and resize；Crop, resize (and flip)；Color distort. (drop)；Color distort. (jitter)；Rotate $\{90^\circ,180^\circ,270^\circ\}$；Cutout；Gaussian noise；Gaussian blur；Sobel filtering。

### 3.1 数据增强操作的组合对学好表征至关重要

![图 4：单独或组合数据增强下的线性评估（ImageNet top-1 准确率），增强仅施加于一支路。除最后一列外，对角线对应单一变换，非对角线对应两种变换的组合（依次施加）。最后一列为该行的平均值。](../../../arxiv/foundations/simclr/extracted/figures/heatmap_da_pairwise.png)

为理解数据增强对对比预测任务以及下游性能的影响，我们系统研究了单独变换及其组合的影响。我们考虑若干常见增强。一种增强操作是对数据的变换，按预设方式随机化，且变换本身并不显著改变数据的视觉外观。我们研究空间/外观变换，如裁剪（含翻转与 resize）、旋转、cutout 等；以及外观变换，如颜色扰动（包括颜色丢弃：颜色抖动与转换为灰度）、高斯模糊与 Sobel 滤波。图 3 可视化了这些变换在示例图像上的效果。

为分离**必须先裁剪**这一混杂因素（否则无法获得两视图），我们在消融研究中采用**非对称**数据增强设置：两支路都先应用随机裁剪，并将裁剪结果 resize 到同一分辨率，再**仅对一支路**施加目标变换。其余训练细节与第 2.3 节默认设置相同。

图 4 展示了单独增强与成对组合下线性评估的结果。我们观察到：

- **没有单一变换足以学习良好表征**，即便模型在对比预测任务上几乎总能完美识别正对。
- 当组合增强时，对比预测任务变得更难，但表征质量显著提升。
- 尤为突出的一种组合是**随机裁剪与颜色失真**：仅用随机裁剪时，多数图块共享相似颜色分布（见图 5），模型可利用颜色直方图区分图像，从而走捷径；与颜色失真组合后，该捷径被阻断，网络被迫学习可泛化的特征。

<p align="center">
  <img src="../../../arxiv/foundations/simclr/extracted/figures/hist_no_da.png" alt="Without color distortion" width="48%"/>
  <img src="../../../arxiv/foundations/simclr/extracted/figures/hist_da.png" alt="With color distortion" width="48%"/>
</p>

**图 5：** 两张不同图像（即两行）不同裁剪上像素强度（所有通道）的直方图。左：无颜色扰动；右：有颜色扰动。

### 3.2 对比学习需要比监督学习更强的数据增强

为进一步展示数据增强的重要性，我们调整颜色增强的强度，并与监督 ResNet-50 比较。表 1 显示：更强的颜色增强显著提升无监督任务上的线性评估性能。在此情况下，AutoAugment（一组高级增强）并不优于简单的裁剪 +（更强）颜色失真。当只用随机裁剪训练时，加入模糊也有帮助。

尽管无监督对比学习受益于颜色增强强度的提升，监督模型则并非如此——更强的颜色增强并不提升甚至损害其性能。当去除颜色（信息）失真并只施加裁剪与模糊时，无监督与监督模型之间的差距要小得多。因此，我们的实验表明：**无监督对比学习比监督学习更受益于（颜色）数据增强。**

**表 1：** 无监督 ResNet-50（线性评估）与监督 ResNet-50 在不同颜色失真强度及其他数据变换下的 top-1 准确率。强度 1（+Blur）为我们的默认数据增强策略。（脚注：监督模型训练 90 epoch；更长训练可使更强增强的性能提升约 0.5%。）

| Methods    |  1/8 |  1/4 |  1/2 |    1 |      1 (+Blur) | AutoAug |
| ---------- | ---: | ---: | ---: | ---: | -------------: | ------: |
| SimCLR     | 59.6 | 61.0 | 62.6 | 63.2 | **64.5** |    61.1 |
| Supervised | 77.0 | 76.7 | 76.5 | 75.7 |           75.4 |    77.1 |

---

## 4. 编码器与投影头的架构

我们研究编码器网络 $f(\cdot)$ 与投影头 $g(\cdot)$ 架构选择的影响。

### 4.1 无监督对比学习（更）受益于更大模型

图 6 显示：增加深度与宽度都能提升性能。与监督学习类似，参数越多，性能越好。同时值得注意的是，随着模型变大，监督模型与线性分类器在无监督模型上的差距缩小，表明**无监督学习比监督学习更受益于更大模型**。

![图 6：不同深度与宽度模型的线性评估。蓝点：本文训练 100 epoch 的模型；红星：本文训练 1000 epoch 的模型；绿叉：监督 ResNet（训练 90 epoch）。](../../../arxiv/foundations/simclr/extracted/figures/curve_params_wodense_top1.png)

### 4.2 非线性投影头提升其前一层的表征质量

我们随后研究包含投影头 $g(\bm h)$ 的重要性。图 7 展示了使用三种不同头架构时的线性评估结果：(1) 恒等映射（identity / 图中 **None**：训练时无投影头，对比损失直接加在表征上）；(2) 线性投影（若干先前方法所使用）；(3) 默认的非线性投影，带一个额外隐层（及 ReLU 激活）。

**由图 7 直接可读的结论（比较的是**训练时用哪种头**，评估时取的是投影前表征 $\bm h$，None 时即骨干输出本身）：** 非线性投影优于线性投影（约 +3%），并远优于无投影（>10%）。当使用投影头时，无论输出维度如何，结果都类似。

**原文紧接着给出的另一条观察（同一段散文陈述，图 7 并未画出对 $z$ 的线性评估柱）：** 即便使用非线性投影，投影头之前的层 $\bm h$ 仍远好于之后的层 $\bm z=g(\bm h)$（>10%），这表明**投影头之前的隐层是比其后一层更好的表征**。换言之：图中绿色 None（~50%）对应的是**训练时没有 $g$**；而**$h$ 优于 $z$**对应的是**训练时已有非线性 $g$ 的同一个模型里，评估时取 $h$ 还是取 $z$**——二者不是同一根柱子。

![图 7：不同投影头 $g(\cdot)$ 以及 $\bm z=g(\bm h)$ 不同输出维度下的线性评估。有投影头时，图中柱状对应的是投影前表征 $\bm h$（此处为 2048 维）的线性评估；None 表示训练时无投影头。图中未单独给出对 $z$ 的线性评估曲线。](../../../arxiv/foundations/simclr/extracted/figures/critic_dim.png)

我们猜想：使用非线性投影前表征之所以重要，是因为对比损失会导致信息损失。具体而言，$\bm z=g(\bm h)$ 被训练为对数据变换不变。因此，$g$ 可能移除对下游任务有用的信息，例如物体的颜色或朝向。通过利用非线性变换 $g(\cdot)$，更多信息可以在 $\bm h$ 中形成并保持。为验证该假设（信息在 $g$ 之后丢失），我们进行实验：分别用 $\bm h$ 或 $g(\bm h)$ 学习预测预训练期间施加的变换。此处设 $g(h)=W^{(2)}\sigma(W^{(1)}h)$，输入与输出维度相同（即 2048）。**表 3** 表明 $\bm h$ 包含关于所施加变换的远更多信息，而 $g(\bm h)$ 丢失了信息（这是原文对**为何 $h$ 优于 $z$**给出的显式实验依据；线性评估上 $h$ vs $z$ 的 >10% 差距则写在正文段落中，未另附专表）。进一步分析见附录 B.4。

**表 3：** 在不同表征上训练额外 MLP 以预测所施加变换的准确率。除裁剪与颜色增强外，对最后三行我们在预训练期间额外且独立地加入旋转（$\{0^\circ,90^\circ,180^\circ,270^\circ\}$ 之一）、高斯噪声与 Sobel 滤波变换。$\bm h$ 与 $g(\bm h)$ 维度相同，均为 2048。

| What to predict?        | Random guess | Representation$\bm h$ | Representation$g(\bm h)$ |
| ----------------------- | -----------: | ----------------------: | -------------------------: |
| Color vs grayscale      |           80 |                    99.3 |                       97.4 |
| Rotation                |           25 |                    67.6 |                       25.6 |
| Orig. vs corrupted      |           50 |                    99.5 |                       59.6 |
| Orig. vs Sobel filtered |           50 |                    96.6 |                       56.3 |

---

## 5. 损失函数与 Batch Size

### 5.1 带可调温度的归一化交叉熵损失优于其他选择

我们将 NT-Xent 损失与其他常用对比损失函数比较，例如 logistic 损失与 margin 损失。表 2 给出了目标函数以及损失函数输入的梯度。观察梯度可知：(1) $\ell_2$ 归一化（即余弦相似度）与温度可有效加权不同样本，合适的温度有助于模型从 hard negatives 学习；(2) 与交叉熵不同，其他目标函数不会按相对难度对负样本加权。因此，对这些损失函数必须应用 semi-hard negative mining：不是对所有损失项计算梯度，而是用 semi-hard 负样本项计算梯度（即落在损失 margin 内、距离最近、但比正样本更远的那些）。

**表 2：** 负损失函数及其梯度。所有输入向量（即 $\bm u,\bm v^+,\bm v^-$）均经 $\ell_2$ 归一化。NT-Xent 是 “Normalized Temperature-scaled Cross Entropy” 的缩写。不同损失函数对正负样本施加不同加权。

| Name           | Negative loss function                                                                | Gradient w.r.t.$\bm u$                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| NT-Xent        | $\bm u^T\bm v^+/\tau-\log\sum_{\bm v\in\{\bm v^+,\bm v^-\}}\exp(\bm u^T\bm v/\tau)$ | $(1-\frac{\exp(\bm u^T\bm v^+/\tau)}{Z(\bm u)})/\tau\,\bm v^+-\sum_{\bm v^-}\frac{\exp(\bm u^T\bm v^-/\tau)}{Z(\bm u)}/\tau\,\bm v^-$ |
| NT-Logistic    | $\log\sigma(\bm u^T\bm v^+/\tau)+\log\sigma(-\bm u^T\bm v^-/\tau)$                  | $(\sigma(-\bm u^T\bm v^+/\tau))/\tau\,\bm v^+-\sigma(\bm u^T\bm v^-/\tau)/\tau\,\bm v^-$                                              |
| Margin Triplet | $-\max(\bm u^T\bm v^--\bm u^T\bm v^++m,0)$                                          | $\bm v^+-\bm v^-$ if $\bm u^T\bm v^+-\bm u^T\bm v^-<m$ else $\bm 0$                                                               |

为公平比较，我们对所有损失函数使用相同的 $\ell_2$ 归一化，并调参后报告最佳结果。（脚注：细节见附录 B.10。为简单起见，我们仅考虑来自一侧增强视图的负样本。）表 4 表明：尽管（semi-hard）负样本挖掘有帮助，最佳结果仍远差于我们的默认 NT-Xent 损失。

**表 4：** 用不同损失函数训练的模型的线性评估（top-1）。“sh” 表示使用 semi-hard negative mining。

| Margin | NT-Logi. | Margin (sh) | NT-Logi.(sh) |        NT-Xent |
| -----: | -------: | ----------: | -----------: | -------------: |
|   50.9 |     51.6 |        57.5 |         57.9 | **63.9** |

我们接着测试默认 NT-Xent 损失中 $\ell_2$ 归一化（即余弦相似度 vs 点积）与温度 $\tau$ 的重要性。表 5 表明：没有归一化与适当的温度缩放时，性能显著更差。没有 $\ell_2$ 归一化时，对比任务准确率更高，但所得表征在线性评估下更差。

**表 5：** 对 NT-Xent 损失采用不同 $\ell_2$ 归一化与温度 $\tau$ 选择时训练模型的线性评估。对比分布覆盖 4096 个样本。

| $\ell_2$ norm? | $\tau$ | Entropy | Contrastive acc. |          Top 1 |
| :--------------: | -------: | ------: | ---------------: | -------------: |
|       Yes       |     0.05 |     1.0 |             90.5 |           59.7 |
|       Yes       |      0.1 |     4.5 |             87.8 | **64.4** |
|       Yes       |      0.5 |     8.2 |             68.2 |           60.7 |
|       Yes       |        1 |     8.3 |             59.1 |           58.0 |
|        No        |       10 |     0.5 |             91.7 |           57.2 |
|        No        |      100 |     0.5 |             92.1 |           57.0 |

### 5.2 对比学习（更）受益于更大 Batch Size 与更长训练

![图 8：用不同 batch size 与 epoch 数训练的模型（ResNet-50）的线性评估。每根柱为从零开始的单次运行。](../../../arxiv/foundations/simclr/extracted/figures/bar_bsstep_top1.png)

图 8 展示了在不同 epoch 数下训练时 batch size 的影响。我们发现：当训练 epoch 数较小时（例如 100 epoch），较大 batch size 相对较小者有显著优势。随着训练步数/epoch 增加，不同 batch size 之间的差距减小或消失（前提是 batch 被随机重采样）。与监督学习不同，在对比学习中，更大 batch size 提供更多负样本，从而促进收敛（即达到给定准确率所需的 epoch 与步数更少）。更长训练同样提供更多负样本，从而提升结果。附录 B.1 给出了更长训练步数的结果。

---

## 6. 与当时最优方法的比较

在本小节中，我们使用三种不同隐层宽度的 ResNet-50（宽度乘数为 $1\times$、$2\times$ 与 $4\times$）。为更好收敛，此处模型训练 1000 epoch。

### 线性评估

表 6 在线性评估设定下将我们的结果与先前方法比较。我们能够用标准网络取得远优于需要专门设计架构的先前方法的结果。我们用 ResNet-50 ($4\times$) 取得的最佳结果可匹配监督预训练的 ResNet-50。

**表 6：** 在不同自监督方法所学表征上训练的线性分类器的 ImageNet 准确率。

| Method                                 | Architecture            | Param (M) |          Top 1 |          Top 5 |
| -------------------------------------- | ----------------------- | --------: | -------------: | -------------: |
| *Methods using ResNet-50:*           |                         |           |                |                |
| Local Agg.                             | ResNet-50               |        24 |           60.2 |             — |
| MoCo                                   | ResNet-50               |        24 |           60.6 |             — |
| PIRL                                   | ResNet-50               |        24 |           63.6 |             — |
| CPC v2                                 | ResNet-50               |        24 |           63.8 |           85.3 |
| SimCLR (ours)                          | ResNet-50               |        24 | **69.3** | **89.0** |
| *Methods using other architectures:* |                         |           |                |                |
| Rotation                               | RevNet-50 ($4\times$) |        86 |           55.4 |             — |
| BigBiGAN                               | RevNet-50 ($4\times$) |        86 |           61.3 |           81.9 |
| AMDIM                                  | Custom-ResNet           |       626 |           68.1 |             — |
| CMC                                    | ResNet-50 ($2\times$) |       188 |           68.4 |           88.2 |
| MoCo                                   | ResNet-50 ($4\times$) |       375 |           68.6 |             — |
| CPC v2                                 | ResNet-161 (*)          |       305 |           71.5 |           90.1 |
| SimCLR (ours)                          | ResNet-50 ($2\times$) |        94 |           74.2 |           92.0 |
| SimCLR (ours)                          | ResNet-50 ($4\times$) |       375 | **76.5** | **93.2** |

### 半监督学习

我们按类别均衡方式采样 1% 或 10% 的带标签 ILSVRC-12 训练数据（每类分别约 12.8 与 128 张图像）。我们简单地在带标签数据上微调整个基网络，不使用正则化。表 7 展示了我们的结果与近期方法的比较。来自先前工作的监督基线因超参（包括增强）的密集搜索而很强。我们的方法在 1% 与 10% 标签上均显著超过当时最优。有趣的是，在**完整** ImageNet 上微调我们预训练的 ResNet-50（$2\times$、$4\times$）也显著优于从零训练（最多约 2%，见附录 B.2）。

**表 7：** 少标签训练模型的 ImageNet 准确率（Top-5）。

| Method                                          | Architecture            |       1% Top 5 |      10% Top 5 |
| ----------------------------------------------- | ----------------------- | -------------: | -------------: |
| Supervised baseline                             | ResNet-50               |           48.4 |           80.4 |
| *Methods using other label-propagation:*      |                         |                |                |
| Pseudo-label                                    | ResNet-50               |           51.6 |           82.4 |
| VAT+Entropy Min.                                | ResNet-50               |           47.0 |           83.4 |
| UDA (w. RandAug)                                | ResNet-50               |             — |           88.5 |
| FixMatch (w. RandAug)                           | ResNet-50               |             — |           89.1 |
| S4L (Rot+VAT+En. M.)                            | ResNet-50 ($4\times$) |             — |           91.2 |
| *Methods using representation learning only:* |                         |                |                |
| InstDisc                                        | ResNet-50               |           39.2 |           77.4 |
| BigBiGAN                                        | RevNet-50 ($4\times$) |           55.2 |           78.8 |
| PIRL                                            | ResNet-50               |           57.2 |           83.8 |
| CPC v2                                          | ResNet-161(*)           |           77.9 |           91.2 |
| SimCLR (ours)                                   | ResNet-50               |           75.5 |           87.8 |
| SimCLR (ours)                                   | ResNet-50 ($2\times$) |           83.0 |           91.2 |
| SimCLR (ours)                                   | ResNet-50 ($4\times$) | **85.8** | **92.6** |

### 迁移学习

我们在线性评估（固定特征提取器）与微调两种设定下，跨 12 个自然图像数据集评估迁移学习性能。对每个模型–数据集组合进行超参调优，并在验证集上选择最佳超参。表 8 展示 ResNet-50 ($4\times$) 模型的结果。微调时，我们的自监督模型在 5 个数据集上显著优于监督基线，而监督基线仅在 2 个数据集上更优（即 Pets 与 Flowers）。在其余 5 个数据集上，模型统计打平。完整实验细节以及标准 ResNet-50 架构的结果见附录 B.8。

**表 8：** 在 ImageNet 上预训练的 ResNet-50 ($4\times$) 模型上，我们的自监督方法与监督基线跨 12 个自然图像分类数据集的迁移学习性能比较。与最佳结果无显著差异（$p>0.05$，置换检验）的结果以粗体标出。

|                        |           Food |        CIFAR10 |       CIFAR100 |       Birdsnap |         SUN397 |           Cars |       Aircraft |        VOC2007 |            DTD |           Pets |    Caltech-101 |        Flowers |
| ---------------------- | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: |
| *Linear evaluation:* |                |                |                |                |                |                |                |                |                |                |                |                |
| SimCLR (ours)          | **76.9** | **95.3** |           80.2 |           48.4 | **65.9** |           60.0 |           61.2 | **84.2** | **78.9** |           89.2 | **93.9** | **95.0** |
| Supervised             |           75.2 | **95.7** | **81.2** | **56.4** |           64.9 | **68.8** | **63.8** |           83.8 | **78.7** | **92.3** | **94.1** |           94.2 |
| *Fine-tuned:*        |                |                |                |                |                |                |                |                |                |                |                |                |
| SimCLR (ours)          | **89.4** | **98.6** | **89.0** | **78.2** | **68.1** | **92.1** | **87.0** | **86.6** | **77.8** |           92.1 | **94.1** |           97.6 |
| Supervised             |           88.7 |           98.3 | **88.7** | **77.8** |           67.0 |           91.4 | **88.0** |           86.5 | **78.8** | **93.2** | **94.2** | **98.0** |
| Random init            |           88.3 |           96.0 |           81.9 | **77.0** |           53.7 |           91.3 |           84.8 |           69.4 |           64.1 |           82.7 |           72.5 |           92.5 |

---

## 7. 相关工作

使图像表征在小变换下彼此一致的思想可追溯至 Becker & Hinton (1992)。我们通过利用数据增强、网络架构与对比损失方面的近期进展加以扩展。类似的一致性思想（但针对**类标签预测**）已在半监督学习等其他语境中被探索。

**手工设计的代理任务。** 自监督学习的近期复兴始于人工设计的代理任务，例如相对图块预测、解拼图、着色与旋转预测。尽管用更大网络与更长训练可取得不错结果，这些代理任务依赖某种 ad-hoc 启发式，从而限制所学表征的通用性。

**对比视觉表征学习。** 可追溯至 Hadsell 等 (2006)，这些方法通过对比正对与负对来学习表征。沿此路线，Dosovitskiy 等提出将每个实例视为由特征向量表示的一个类（参数形式）。Wu 等提出用 memory bank 存储实例类表征向量，该方法被若干近期论文采用并扩展。其他工作探索用 batch 内样本做负采样，而非 memory bank。

近期文献试图将其方法的成功与潜表征之间互信息最大化联系起来。然而，尚不清楚对比方法的成功是由互信息决定，还是由对比损失的具体形式决定。

我们注意到：我们框架中几乎所有单独组件都曾出现在先前工作中，尽管具体实例化可能不同。我们框架相对先前工作的优势不能由任一单独设计选择解释，而由其**组合**解释。附录 C 给出了我们设计选择与先前工作的全面比较。

---

## 8. 结论

本文提出并实例化了一个用于对比视觉表征学习的简单框架。我们仔细研究其组件，并展示不同设计选择的影响。通过综合我们的发现，我们在自监督、半监督与迁移学习上相对先前方法取得显著提升。

相对标准的监督 ImageNet 训练，我们方法的主要差异在于：数据增强的选择、网络末端的非线性头，以及损失函数。这一简单框架的强度表明：尽管兴趣高涨，自监督学习仍被低估。

---

## 附录 A. 数据增强细节

默认预训练设置（用于训练最佳模型）采用：随机裁剪（含 resize 与随机翻转）、随机颜色扰动、随机高斯模糊。细节如下。

### A.1 随机裁剪并 resize 到 224×224

采用标准 Inception 风格随机裁剪：裁剪面积相对原图均匀采样于 $[0.08, 1.0]$，长宽比相对原图默认在 $[3/4, 4/3]$，再 resize 回原尺寸。TensorFlow 实现为 `slim.preprocessing.inception_preprocessing.distorted_bounding_box_crop`，PyTorch 为 `torchvision.transforms.RandomResizedCrop`。裁剪后总是以 50% 概率做随机水平翻转。翻转有帮助但非必需：从默认策略中去掉后，ResNet-50（100 epoch）线性评估 top-1 从 64.5% 降到 63.4%。

### A.2 颜色扰动

颜色扰动由 **color jittering** 与 **color dropping** 组成。更强的 jittering 通常更好，故设置强度参数 $s$。

**TensorFlow 伪代码：**

```python
import tensorflow as tf
def color_distortion(image, s=1.0):
    # image 取值范围 [0, 1]；s 为颜色扰动强度
    def color_jitter(x):
        # 也可每次打乱下列增强的顺序
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def color_drop(x):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1, 1, 3])
        return image

    # 以概率 p 随机应用变换
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image
```

**PyTorch 伪代码**（结果以 TensorFlow 为准，此处仅作参考）：

```python
from torchvision import transforms
def get_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort
```

### A.3 高斯模糊

属于默认策略。对 ResNet-50（100 epoch），加入模糊后 top-1 从 63.2% 提升到 64.5%。以 50% 概率用高斯核对图像模糊；随机采样 $\sigma\in[0.1, 2.0]$，核大小设为图像高/宽的 10%。

---

## 附录 B. 补充实验结果

### B.1 Batch Size 与训练步数

![图 B.1：不同 batch size 与 epoch 下 ResNet-50 线性评估（top-5）。每根柱为从零训练的单次运行。](../../../arxiv/foundations/simclr/extracted/figures/bar_bsstep_top5.png)

结论与 top-1 类似，但不同 batch / 训练步之间的差距略小。

正文图中对不同 batch 使用类似 Goyal 等的**线性学习率缩放**。虽线性缩放在 SGD/Momentum 上常用，但对 LARS 更宜用**平方根缩放**：$\mathrm{LearningRate}=0.075\times\sqrt{\mathrm{BatchSize}}$，而非线性情形的 $0.3\times\mathrm{BatchSize}/256$。在默认 batch 4096 时两种缩放学习率相同。

**表 B.1：** 不同 batch size 与训练 epoch 下的线性评估（top-1）。斜杠左侧为线性 LR 缩放，右侧为平方根 LR 缩放；若优出超过 0.5% 则加粗。平方根缩放对小 batch、较少 epoch 更有利（配合 LARS）。

| Batch size \ Epochs | 100                  | 200                  | 400                  | 800         |
| ------------------: | -------------------- | -------------------- | -------------------- | ----------- |
|                 256 | 57.5 /**62.8** | 61.9 /**64.3** | 64.7 /**65.7** | 66.6 / 66.5 |
|                 512 | 60.7 /**63.8** | 64.0 /**65.6** | 66.2 / 66.7          | 67.8 / 67.4 |
|                1024 | 62.8 /**64.3** | 65.3 /**66.1** | 67.2 / 67.2          | 68.5 / 68.3 |
|                2048 | 64.0 /**64.7** | 66.1 /**66.8** | 68.1 / 67.9          | 68.9 / 68.8 |
|                4096 | 64.6 / 64.5          | 66.5 / 66.8          | 68.2 / 68.0          | 68.9 / 69.1 |
|                8192 | 64.8 / 64.8          | 66.6 / 67.0          | 67.8 / 68.3          | 69.0 / 69.1 |

进一步用平方根缩放训练更大 batch（至 32K）与更长训练（至 3200 epoch）：

![图 B.2：不同 batch 与更长 epoch 下的线性评估 top-1（平方根学习率）。](../../../arxiv/foundations/simclr/extracted/figures/bar_bsstep_top1_sqrt.png)

**性能在 batch 8192 附近趋于饱和，而更长训练仍可显著提升。**

### B.2 更广的数据增强组合可进一步提升

将默认增强扩展为包含：(1) Sobel 滤波；(2) 额外颜色扰动（equalize、solarize）；(3) motion blur。线性评估下，ResNet-50（$1\times, 2\times, 4\times$）分别达到 **70.0（+0.7）、74.4（+0.2）、76.8（+0.3）**。

**表 B.2：** 用更广增强预训练的 SimCLR，在 1%、10%、100% ImageNet 上微调的分类准确率。参考：ResNet-50 ($4\times$) 从零训练（100% 标签）为 78.4% top-1 / 94.2% top-5。

| Architecture            | 1% Top1 | 1% Top5 | 10% Top1 | 10% Top5 |      100% Top1 |      100% Top5 |
| ----------------------- | ------: | ------: | -------: | -------: | -------------: | -------------: |
| ResNet-50               |    49.4 |    76.6 |     66.1 |     88.1 |           76.0 |           93.1 |
| ResNet-50 ($2\times$) |    59.4 |    83.7 |     71.8 |     91.2 |           79.1 |           94.8 |
| ResNet-50 ($4\times$) |    64.1 |    86.6 |     74.8 |     92.8 | **80.4** | **95.4** |

有趣的是：在完整 ImageNet 上微调时，ResNet ($4\times$) 达到 **80.4% / 95.4%**（无更广增强预训练时为 80.1% / 95.2%），显著优于用同一增强集从零训练（78.4% / 94.2%）。ResNet-50 ($2\times$) 微调也优于从零（77.8% / 93.9%）；标准 ResNet-50 微调无额外提升。

### B.3 监督模型更长训练的影响

在与无监督相同的增强集（随机裁剪、颜色扰动、50% 高斯模糊）下测试 ResNet-50 与 ResNet-50 ($4\times$)。

**表 B.3：** 监督模型在更长训练与不同增强下的 top-1。

| Model                   | Training epochs | Crop | +Color | +Color+Blur |
| ----------------------- | --------------: | ---: | -----: | ----------: |
| ResNet-50               |              90 | 76.5 |   75.6 |        75.3 |
| ResNet-50               |             500 | 76.2 |   76.5 |        76.7 |
| ResNet-50               |            1000 | 75.8 |   75.2 |        76.4 |
| ResNet-50 ($4\times$) |              90 | 78.4 |   78.9 |        78.7 |
| ResNet-50 ($4\times$) |             500 | 78.3 |   78.4 |        78.5 |
| ResNet-50 ($4\times$) |            1000 | 77.9 |   78.2 |        78.3 |

观察：ImageNet 上监督模型**更长训练无明显收益**。更强增强略提升 ResNet-50 ($4\times$)，对 ResNet-50 无帮助。使用更强增强时，ResNet-50 通常需要更长训练（如 500 epoch）才达最优；ResNet-50 ($4\times$) 不受益于更长训练。

### B.4 理解非线性投影头

<p align="center">
  <img src="../../../arxiv/foundations/simclr/extracted/figures/critic_linproj_eig.png" alt="均匀刻度" width="48%"/>
  <img src="../../../arxiv/foundations/simclr/extracted/figures/critic_linproj_eig_log.png" alt="对数刻度" width="48%"/>
</p>

**图 B.3：** 线性投影矩阵 $W\in\mathbb{R}^{2048\times2048}$（用于 $g(\bm h)=W\bm h$）的平方实特征值分布。左：均匀刻度；右：对数刻度。该矩阵仅有相对较少的大特征值，表明近似低秩。

<p align="center">
  <img src="../../../arxiv/foundations/simclr/extracted/figures/tsne_h1.png" alt="h" width="48%"/>
  <img src="../../../arxiv/foundations/simclr/extracted/figures/tsne_gh1.png" alt="z=g(h)" width="48%"/>
</p>

**图 B.4：** 验证集随机选取 10 类图像隐向量的 t-SNE。左：$\bm h$；右：$\bm z=g(\bm h)$。对最佳 ResNet-50（线性评估 top-1 69.3%），$\bm h$ 表示的类别分离明显优于 $\bm z$。

### B.5 通过微调做半监督学习

**微调流程：** Nesterov momentum，batch 4096，动量 0.9，学习率 0.8（$\mathrm{LearningRate}=0.05\times\mathrm{BatchSize}/256$），无 warmup。预处理仅随机裁剪（含随机左右翻转并 resize 到 224×224）。**不使用任何正则**（含权重衰减）。1% 标签微调 60 epoch，10% 微调 30 epoch。推理时将图像 resize 到 256×256，取中心 $224\times224$ 裁剪。

**表 B.4：** 少标签 ImageNet 的 top-1（正文表 7 为 top-5）。

| Method                                                          | Architecture            |       1% Top 1 |      10% Top 1 |
| --------------------------------------------------------------- | ----------------------- | -------------: | -------------: |
| Supervised baseline                                             | ResNet-50               |           25.4 |           56.4 |
| *Methods using label-propagation:*                            |                         |                |                |
| UDA (w. RandAug)                                                | ResNet-50               |             — |           68.8 |
| FixMatch (w. RandAug)                                           | ResNet-50               |             — |           71.5 |
| S4L (Rot+VAT+Ent. Min.)                                         | ResNet-50 ($4\times$) |             — |           73.2 |
| *Methods using self-supervised representation learning only:* |                         |                |                |
| CPC v2                                                          | ResNet-161(*)           |           52.7 |           73.1 |
| SimCLR (ours)                                                   | ResNet-50               |           48.3 |           65.6 |
| SimCLR (ours)                                                   | ResNet-50 ($2\times$) |           58.5 |           71.7 |
| SimCLR (ours)                                                   | ResNet-50 ($4\times$) | **63.0** | **74.4** |

### B.6 线性评估

流程类似微调，但学习率更大（1.6，即 $0.1\times\mathrm{BatchSize}/256$），训练 90 epoch。用 LARS 与预训练超参也可得到相近结果。另发现：在预训练时把线性分类器接在基编码器上（对分类器输入做 `stop_gradient`，防止标签信息影响编码器）并同时训练，可达到相近性能。

### B.7 线性评估与微调的相关性

![图 B.5：不同 epoch 训练模型在线性评估与微调下的 top-1（对应图 B.2 的设置）。](../../../arxiv/foundations/simclr/extracted/figures/epoch_linear_vs_ft.png)

二者几乎线性相关；在小比例标签上微调似乎更受益于更长预训练。

<p align="center">
  <img src="../../../arxiv/foundations/simclr/extracted/figures/arch_linear_vs_ft_1pct.png" alt="1%" width="45%"/>
  <img src="../../../arxiv/foundations/simclr/extracted/figures/arch_linear_vs_ft_10pct.png" alt="10%" width="45%"/>
</p>

**图 B.6：** 不同架构在线性评估与微调下的 top-1（左：1% 标签；右：10% 标签）。

### B.8 迁移学习

在两种设置下评估：线性评估（冻结 ImageNet 自监督表征，训练逻辑回归）与微调（允许所有权重更新）。大体遵循 Kornblith 等的做法，预处理略有不同。

#### 方法

**数据集：** Food-101、CIFAR-10/100、Birdsnap、SUN397、Stanford Cars、FGVC Aircraft、PASCAL VOC 2007 分类、DTD、Oxford-IIIT Pets、Caltech-101、Oxford 102 Flowers。指标按各数据集原文：多数报 top-1；Aircraft / Pets / Caltech-101 / Flowers 报 mean per-class accuracy；VOC 2007 报 11-point mAP。DTD 与 SUN397 仅报第一折；Caltech-101 无官方划分，每类随机选 30 张训练、其余测试。

用数据集验证集（或从训练集划出）选超参，再在全部训练+验证图像上重训，报告测试集准确率。

**线性分类器迁移：** 在冻结网络提取的特征上训练 $\ell_2$ 正则多项逻辑回归；L-BFGS 优化 softmax 交叉熵；无数据增强。预处理：短边 bicubic resize 到 224，再取 $224\times224$ 中心裁剪。$\ell_2$ 正则从 $10^{-6}$ 到 $10^5$ 的 45 个对数间隔值中选取。

**微调迁移：** 以预训练权重初始化，整网微调。batch 256，SGD + Nesterov（动量 0.9），训练 20,000 步。BN 统计动量为 $\max(1-10/s, 0.9)$（$s$ 为每 epoch 步数）。微调增强仅随机裁剪+翻转（无颜色增强或模糊）。测试时短边 resize 到 256，取 $224\times224$ 中心裁剪。学习率与权重衰减在对数网格上搜索（学习率 $10^{-4}$–$0.1$ 共 7 档；权重衰减 $10^{-6}$–$10^{-3}$ 共 7 档，另加无衰减）；报告的权重衰减值再除以学习率。

**随机初始化训练：** 流程同微调，但更长（40,000 步），超参网格改为学习率 $0.001$–$1.0$、权重衰减 $10^{-5}$–$10^{-1.5}$。该长度足以接近最大准确率。

在 Birdsnap 上各方法无显著差异；Food-101、Cars、Aircraft 上微调相对从零训练优势较小；其余 8 个数据集上预训练优势明显。

**监督基线：** 架构相同的 ResNet，用标准交叉熵在 ImageNet 上训练；增强与自监督相同（裁剪、强颜色、模糊），训练 1000 epoch。虽更强增强与更长训练对 ImageNet 本身无益，但在部分迁移数据集的线性评估上明显优于 90 epoch + 普通增强的监督基线。监督 ResNet-50 ImageNet top-1 为 76.3%（自监督 69.3%）；ResNet-50 ($4\times$) 为 78.3%（自监督 76.5%）。

**统计显著性：** 用置换检验。给定两模型预测，从零假设下随机交换各样本预测并计算准确率差，重复 100,000 次，看比观测差异更极端的比例。对 top-1 等价于精确 McNemar 检验；对 mean per-class accuracy 交换性假设仍成立；对 VOC 比较准确率而非 mAP。注意该过程不考虑训练 run-to-run 变异，只考虑有限评估样本带来的变异。

#### 标准 ResNet-50 结果

**表 B.5：** ImageNet 预训练标准 ResNet-50 在 12 个数据集上的迁移（另见表 8 的 $4\times$ 结果）。

|                        |           Food |        CIFAR10 |       CIFAR100 |       Birdsnap |         SUN397 |           Cars |       Aircraft |        VOC2007 |            DTD |           Pets |    Caltech-101 |        Flowers |
| ---------------------- | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: |
| *Linear evaluation:* |                |                |                |                |                |                |                |                |                |                |                |                |
| SimCLR (ours)          |           68.4 |           90.6 |           71.6 |           37.4 |           58.8 |           50.3 |           50.3 |           80.5 | **74.5** |           83.6 |           90.3 |           91.2 |
| Supervised             | **72.3** | **93.6** | **78.3** | **53.7** | **61.9** | **66.7** | **61.0** | **82.8** | **74.9** | **91.5** | **94.5** | **94.7** |
| *Fine-tuned:*        |                |                |                |                |                |                |                |                |                |                |                |                |
| SimCLR (ours)          | **88.2** | **97.7** | **85.9** | **75.9** |           63.5 |           91.3 | **88.1** |           84.1 | **73.2** |           89.2 |           92.1 |           97.0 |
| Supervised             | **88.3** | **97.5** | **86.4** | **75.8** | **64.3** | **92.1** |           86.0 | **85.0** | **74.6** | **92.1** | **93.3** | **97.6** |
| Random init            |           86.9 |           95.9 |           80.2 | **76.1** |           53.6 |           91.4 |           85.9 |           67.3 |           64.8 |           81.5 |           72.6 |           92.0 |

正文中 ResNet-50 ($4\times$) 上监督与自监督无明显优劣；但在更窄的标准 ResNet-50 上，监督仍明显占优：线性评估全部数据集、微调 12 中 10 个更优。这可能与 ImageNet 上准确率差距有关：自监督 ResNet 比监督低 6.8%（69.3% vs 76.3%），而 $4\times$ 仅低 1.8%（76.5% vs 78.3%）。

### B.9 CIFAR-10

目标不是刷 CIFAR-10 SOTA，而是进一步确认 ImageNet 上的观察。仍用 ResNet-50：因图像更小，将首层 $7\times7$ stride 2 卷积为 $3\times3$ stride 1，并去掉首个 max pooling。增强：Inception crop（翻转并 resize 到 32×32）+ 颜色扰动（强度 0.5），不加高斯模糊。预训练搜索学习率 $\{0.5,1.0,1.5\}$、温度 $\{0.1,0.5,1.0\}$、batch $\{256,512,1024,2048,4096\}$；其余设置同 ImageNet。

最佳模型（batch 1024）线性评估 **94.0%**，同架构同 batch 监督基线为 95.1%。此前 CIFAR-10 线性评估最佳自监督 AMDIM 为 91.2%，模型约大 25×。加入额外增强或更合适骨干还可提升。

![图 B.7：CIFAR-10 上不同 batch size 与 epoch 的线性评估。每柱为 3 次运行平均（学习率 0.5/1.0/1.5，温度 <span class=](../../../arxiv/foundations/simclr/extracted/figures/cifar10_bar_bsstep_top1.png)\tau=0.5\tau=0.5），误差棒为标准差。" />

结论与 ImageNet 一致；最大 batch 4096 在 CIFAR-10 上略有下降。

<p align="center">
  <img src="../../../arxiv/foundations/simclr/extracted/figures/cifar10_bar_bstau1.png" alt="≤300" width="45%"/>
  <img src="../../../arxiv/foundations/simclr/extracted/figures/cifar10_bar_bstau2.png" alt=">300" width="45%"/>
</p>

**图 B.8：** CIFAR-10 上三种温度与不同 batch 的线性评估。左：训练 epoch ≤300；右：>300。

收敛后（如 epoch >300），$\{0.1,0.5,1.0\}$ 中最优温度为 **0.5**，且与 batch 基本无关；但 $\tau=0.1$ 的表现随 batch 增大而改善，暗示最优温度可能略向 0.1 偏移。

### B.10 其他损失函数的调参

对 NT-Xent 最优的学习率未必适合其他损失。为保证公平，对 margin loss 与 logistic loss 也调参：学习率 $\{0.01,0.1,0.3,0.5,1.0\}$；margin loss 再调 margin $\{0,0.4,0.8,1.6\}$；logistic loss 再调温度 $\{0.1,0.2,0.5,1.0\}$。为简单起见，仅使用一侧增强视图的负样本（略损性能，但保证公平对比）。

---

## 附录 C. 与相关方法的进一步比较

正文已指出：SimCLR 多数单独组件曾出现在先前工作中，性能提升来自组合。下表给出高层设计对比；相对先前工作，本文选择通常更简单。

**表 C.1：** 各方法设计选择与训练设置（ImageNet 最佳结果）的高层对比。即使描述相同，具体公式与实现也可能不同。$^\#$ 样本被切成多 patch，有效增大 batch；$^*$ 使用 memory bank。

| Model  | Data Augmentation | Base Encoder                  | Projection Head | Loss                   | Batch Size  | Train Epochs |
| ------ | ----------------- | ----------------------------- | --------------- | ---------------------- | ----------- | -----------: |
| CPC v2 | Custom            | ResNet-161 (modified)         | PixelCNN        | Xent                   | 512$^\#$  |         ~200 |
| AMDIM  | Fast AutoAug.     | Custom ResNet                 | Non-linear MLP  | Xent w/ clip,reg       | 1008$^\#$ |          150 |
| CMC    | Fast AutoAug.     | ResNet-50 ($2\times$, L+ab) | Linear layer    | Xent w/$\ell_2,\tau$ | 156$^*$   |          280 |
| MoCo   | Crop+color        | ResNet-50 ($4\times$)       | Linear layer    | Xent w/$\ell_2,\tau$ | 256$^*$   |          200 |
| PIRL   | Crop+color        | ResNet-50 ($2\times$)       | Linear layer    | Xent w/$\ell_2,\tau$ | 1024$^*$  |          800 |
| SimCLR | Crop+color+blur   | ResNet-50 ($4\times$)       | Non-linear MLP  | Xent w/$\ell_2,\tau$ | 4096        |         1000 |

与近期对比表征学习方法的深入对比：

- **DIM/AMDIM：** 通过预测 ConvNet 中间层实现全局→局部 / 局部→邻域预测；对感受野施加强约束（大量 $3\times3$ 换成 $1\times1$）。本文用随机裁剪（含 resize）与两增强视图的最终表征做预测，将预测任务与编码器解耦，从而可用标准、更强的 ResNet。NT-Xent 用归一化与温度限制相似度范围，而他们用带正则的 tanh。本文增强策略更简单，他们最佳结果用 FastAutoAugment。
- **CPC v1/v2：** 用确定性切 patch + PixelCNN 上下文聚合定义预测任务；编码器只看远小于原图的 patch。本文解耦预测任务与架构，无需上下文聚合网络，编码器可看更广分辨率谱。损失用带归一化与温度的 NT-Xent，而非未归一化交叉熵；增强也更简单。
- **InstDisc / MoCo / PIRL：** 推广 Exemplar 思路并显式使用 memory bank。本文不用 memory bank，发现大 batch 下 batch 内负样本已足够；并使用非线性投影头，取投影前表征。增强类型相近（裁剪+颜色），但具体参数可能不同。
- **CMC：** 每个视图用独立网络；本文对所有随机增强视图共享单一网络。增强、投影头与损失也不同；用更大 batch 而非 memory bank。
- **Ye 等：** 最大化增强与未增强副本的相似度；本文对框架两支路**对称**施加增强。本文对基特征网络输出做非线性投影，并用投影前表征；Ye 等用线性投影后的最终隐向量。多加速器大 batch 训练时用 Global BN，避免会严重损害表征质量的捷径。
