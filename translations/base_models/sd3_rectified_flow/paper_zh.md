# 扩展整流流 Transformer 用于高分辨率图像合成（Scaling Rectified Flow Transformers for High-Resolution Image Synthesis）

**作者：** Patrick Esser$^{*}$、Sumith Kulal、Andreas Blattmann、Rahim Entezari、Jonas Müller、Harry Saini、Yam Levi、Dominik Lorenz、Axel Sauer、Frederic Boesel、Dustin Podell、Tim Dockhorn、Zion English、Kyle Lacey、Alex Goodwin、Yannik Marek、Robin Rombach$^{*}$
（$^{*}$前两位与最后一位作者贡献同等）
**机构：** Stability AI
**会议：** ICML 2024
**arXiv：** [2403.03206](https://arxiv.org/abs/2403.03206)
**原文 TeX：** [`arxiv/base_models/sd3_rectified_flow/extracted/main.tex`](../../../arxiv/base_models/sd3_rectified_flow/extracted/main.tex)

---

## 摘要

扩散模型通过反转数据朝向噪声的前向路径，从噪声中生成数据，已成为图像、视频等高维感知数据的强大生成建模技术。整流流（Rectified Flow）是近期提出的一种生成模型形式化方法，它在数据与噪声之间建立直线连接。尽管其具有更好的理论性质与概念简洁性，尚未被明确确立为标准实践。在本工作中，我们通过将噪声采样偏向感知相关尺度，改进了用于训练整流流模型的现有噪声采样技术。通过大规模研究，我们证明了该方法在高分辨率文本到图像合成上优于既有扩散形式化。此外，我们提出了一种用于文本到图像生成的新型 Transformer 架构：对两种模态使用独立权重，并在图像与文本 token 之间实现双向信息流，从而提升文本理解、排版与人类偏好评分。我们证明该架构遵循可预测的扩展趋势，且更低的验证损失与多种指标及人类评估所衡量的改进文本到图像合成相关。我们最大的模型优于当前最优模型；我们将公开实验数据、代码与模型权重。

![图：8B 整流流模型的高分辨率样本，展示其在排版、精确提示遵循、空间推理、细节关注以及多种风格下的高图像质量方面的能力。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/teaser.png)

---

## 1. 引言

扩散模型从噪声中生成数据。它们被训练来反转数据朝向随机噪声的前向路径，因此结合神经网络的近似与泛化性质，可用于生成训练数据中不存在、但遵循训练数据分布的新数据点。这一生成建模技术已被证明对图像等高维感知数据非常有效。近年来，扩散模型已成为从自然语言输入生成高分辨率图像与视频的事实标准方法，并展现出令人印象深刻的泛化能力。由于其迭代性质及相应的计算成本，以及推理阶段较长的采样时间，关于更高效训练与/或更快采样的形式化研究日益增多。

虽然指定从数据到噪声的前向路径可带来高效训练，但也引出了应选择哪条路径的问题。这一选择对采样可能有重要影响。例如，若前向过程未能从数据中完全去除噪声，可能导致训练与测试分布不一致，并产生诸如灰色图像样本等伪影。重要的是，前向过程的选择也影响所学反向过程，进而影响采样效率。弯曲路径需要许多积分步来模拟过程，而直线路径可用单步模拟且不易累积误差。由于每一步对应一次神经网络求值，这直接影响采样速度。

一种特定的前向路径选择是所谓的**整流流**（Rectified Flow），它在数据与噪声之间以直线连接。尽管该类模型具有更好的理论性质，尚未在实践上被明确确立。迄今，一些优势已在中小规模实验中得到经验验证，但大多限于类别条件模型。在本工作中，我们通过引入整流流模型中噪声尺度的重加权（类似于噪声预测扩散模型）来改变这一状况。通过大规模研究，我们将新形式化与现有扩散形式化比较，并展示其优势。

我们表明，广泛用于文本到图像合成的方法——将固定文本表示直接输入模型（例如通过交叉注意力）——并不理想；我们提出一种新架构，为图像与文本 token 引入可学习流，使二者之间实现双向信息流。我们将此与改进的整流流形式化结合，并研究其可扩展性。我们展示了验证损失的可预测扩展趋势，并表明更低的验证损失与改进的自动评估及人类评估强相关。

我们最大的模型在提示理解定量评估与人类偏好评分上，均优于 SDXL、SDXL-Turbo、Pixart-$\alpha$ 等开放最优模型，以及 DALL-E 3 等闭源模型。

本工作的核心贡献为：

**(i)** 我们对不同扩散模型与整流流形式化进行大规模、系统性研究，以识别最佳设置。为此，我们引入了用于整流流模型的新噪声采样器，其性能优于先前已知采样器。

**(ii)** 我们设计了一种用于文本到图像合成的新型可扩展架构，允许网络内文本与图像 token 流之间的双向混合。我们展示了其相对 UViT、DiT 等既有主干的优势。

**(iii)** 我们对该模型进行扩展研究，证明其遵循可预测的扩展趋势。我们表明更低的验证损失与 T2I-CompBench、GenEval 等指标及人类评分所评估的改进文本到图像性能强相关。

我们将公开结果、代码与模型权重。

---

## 2. 背景

### 扩散模型

扩散模型通过近似将数据变换为噪声的随机前向过程的反向 ODE 来生成数据。它们已成为图像与视频生成建模的标准方法。由于这些模型既可通过负对数似然的变分下界推导，也可通过分数匹配（score matching）推导，前向与反向过程、模型参数化、损失加权及 ODE 求解器的多种形式化导致了大量不同的训练目标与采样流程。近期 Kingma 与 Karras 等人的开创性工作提出了统一形式化，并为训练与推理引入了新的理论与实践洞见。然而，尽管有这些改进，常见 ODE 的轨迹仍涉及相当程度的曲率，需要更多求解器步数，因而使快速推理困难。为克服这一点，我们采用整流流模型，其形式化允许学习直线 ODE 轨迹。

### 整流流模型

整流流通过 ODE 在两个分布之间构造传输映射来进行生成建模。该方法与连续归一化流（CNF）及扩散模型有密切联系。相比 CNF，整流流与随机插值（Stochastic Interpolants）的优势在于训练时无需模拟 ODE。相比扩散模型，它们可产生比扩散模型相关概率流 ODE 更快模拟的 ODE。尽管如此，它们并不给出最优传输解，多项工作旨在进一步最小化轨迹曲率。已有工作证明了整流流形式化在类别条件图像合成、潜空间上采样中的可行性；也有工作将 reflow 过程应用于蒸馏预训练文本到图像模型。在此，我们关注整流流作为少步采样文本到图像合成的基础。我们对不同形式化与损失加权进行广泛比较，并提出了性能改进的整流流训练时间步调度。

### 扩展扩散模型

Transformer 架构在 NLP 与计算机视觉任务中以其扩展性质著称。对扩散模型而言，U-Net 架构一直是主流选择。尽管近期工作探索扩散 Transformer 主干，文本到图像扩散模型的扩展定律仍未被探索。

---

## 3. 流的免模拟训练

我们考虑通过常微分方程（ODE）定义从噪声分布 $p_1$ 的样本 $x_1$ 到数据分布 $p_0$ 的样本 $x_0$ 的映射的生成模型：

$$
dy_t = v_\Theta(y_t, t)\,dt
$$

其中速度 $v$ 由神经网络权重 $\Theta$ 参数化。Chen 等人的先前工作建议通过可微 ODE 求解器直接求解上式，但该过程计算昂贵，尤其对参数化 $v_\Theta(y_t, t)$ 的大型网络架构而言。更高效的替代方案是直接回归生成 $p_0$ 与 $p_1$ 之间概率路径的向量场 $u_t$。

为构造这样的 $u_t$，我们定义前向过程，对应 $p_0$ 与 $p_1=\mathcal{N}(0, 1)$ 之间的概率路径 $p_t$：

$$
z_t = a_t x_0 + b_t \epsilon\quad\text{其中}\;\epsilon \sim \mathcal{N}(0,I)
$$

对 $a_0 = 1, b_0 = 0, a_1 = 0$ 且 $b_1 = 1$，边缘分布

$$
p_t(z_t) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} p_t(z_t \vert \epsilon)
$$

与数据及噪声分布一致。

为表达 $z_t$、$x_0$ 与 $\epsilon$ 的关系，我们引入 $\psi_t$ 与 $u_t$：

$$
\begin{aligned}
\psi_t(\cdot | \epsilon) &: x_0 \mapsto a_t x_0 + b_t \epsilon \\
u_t(z| \epsilon) &\coloneqq \psi'_t(\psi_t^{-1}(z| \epsilon)  \vert \epsilon)
\end{aligned}
$$

由于 $z_t$ 可写为 ODE $z_t' = u_t(z_t |\epsilon)$、初值 $z_0=x_0$ 的解，$u_t(\cdot | \epsilon)$ 生成 $p_t(\cdot | \epsilon)$。值得注意的是，可利用条件向量场 $u_t(\cdot | \epsilon)$ 构造生成边缘概率路径 $p_t$ 的边缘向量场 $u_t$：

$$
u_t(z) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} u_t(z \vert \epsilon) \frac{p_t(z \vert \epsilon)}{p_t(z)}
$$

虽然直接用**流匹配**（Flow Matching）目标

$$
\mathcal{L}_{FM} =  \mathbb{E}_{t, p_t(z)} \| v_{\Theta}(z, t) - u_t(z) \|_2^2
$$

回归 $u_t$ 由于式中边缘化而不可行，**条件流匹配**（Conditional Flow Matching）

$$
\mathcal{L}_{CFM} =  \mathbb{E}_{t, p_t(z | \epsilon), p(\epsilon) }\| v_{\Theta}(z, t) - u_t(z | \epsilon)  \|_2^2
$$

提供了等价且可处理的目标。

将 $\psi_t'(x_0 \vert \epsilon) = a_t'x_0 + b_t' \epsilon$ 与 $\psi_t^{-1}(z\vert \epsilon) = \frac{z -b_t \epsilon}{a_t}$ 代入，得

$$
z_t' = u_t(z_t \vert \epsilon) = \frac{a_t'}{a_t} z_t - \epsilon b_t \left(\frac{a_t'}{a_t} - \frac{b_t'}{b_t}\right)
$$

考虑**信噪比** $\lambda_t := \log \frac{a_t^2}{b_t^2}$。由 $\lambda_t' = 2 \left(\frac{a_t'}{a_t} - \frac{b_t'}{b_t}\right)$，可重写为

$$
u_t(z_t \vert \epsilon) = \frac{a_t'}{a_t} z_t - \frac{b_t}{2} \lambda_t' \epsilon
$$

进而可将条件流匹配重参数化为噪声预测目标：

$$
\begin{aligned}
\mathcal{L}_{CFM} &= \mathbb{E}_{t, p_t(z | \epsilon), p(\epsilon) } \left\| v_{\Theta}(z, t) - \frac{a_t'}{a_t} z  + \frac{b_t}{2} \lambda_t' \epsilon \right\|_2^2 \\
&= \mathbb{E}_{t, p_t(z | \epsilon), p(\epsilon) } \left(-\frac{b_t}{2}\lambda_t' \right)^2  \| \epsilon_\Theta(z, t) - \epsilon \|_2^2
\end{aligned}
$$

其中 $\epsilon_\Theta \coloneqq \frac{-2}{\lambda_t' b_t} \left(v_\Theta - \frac{a_t'}{a_t} z\right)$。

注意，引入与时间相关的加权不改变上述目标的最优解。因此可导出多种加权损失，它们指向期望解但可能影响优化轨迹。为统一分析包括经典扩散形式化在内的不同方法，可将目标写为（遵循 Kingma 等人）：

$$
\mathcal{L}_w(x_0) = -\frac{1}{2} \mathbb{E}_{t\sim\mathcal{U}(t), \epsilon\sim \mathcal{N}(0, I)} \left[ w_t \lambda_t' \Vert \epsilon_\Theta(z_t, t) - \epsilon \Vert^2 \right]
$$

其中 $w_t = -\frac{1}{2} \lambda_t' b_t^2$ 对应 $\mathcal{L}_{CFM}$。

---

## 4. 流轨迹

本工作考虑上述形式化的不同变体，简述如下。

### 整流流（Rectified Flow）

整流流将前向过程定义为数据分布与标准正态分布之间的直线路径：

$$
z_t = (1-t) x_0 + t \epsilon
$$

并使用 $\mathcal{L}_{CFM}$，此时对应 $w_t^\text{RF} = \frac{t}{1-t}$。网络输出直接参数化速度 $v_\Theta$。

### EDM

EDM 使用前向过程

$$
z_t = x_0 + b_t \epsilon
$$

其中 $b_t = \exp{F_{\mathcal{N}}^{-1}(t \vert P_m, P_s^2)}$，$F_{\mathcal{N}}^{-1}$ 为均值 $P_m$、方差 $P_s^2$ 的正态分布分位函数。该选择导致

$$
\lambda_t \sim \mathcal{N}(-2P_m, (2P_s)^2)\quad\text{对}\;t\sim \mathcal{U}(0,1)
$$

网络通过 **F-预测**参数化，损失可写为 $\mathcal{L}_{w_t^\text{EDM}}$，其中

$$
w_t^\text{EDM} = \mathcal{N}(\lambda_t \vert -2P_m, (2P_s)^2)(e^{-\lambda_t}+0.5^2)
$$

### 余弦（Cosine）

Nichol 等人提出前向过程

$$
z_t = \cos\bigl(\frac{\pi}{2} t\bigr) x_0 + \sin\bigl(\frac{\pi}{2} t\bigr) \epsilon
$$

结合 $\epsilon$-参数化与损失，对应加权 $w_t = \operatorname{sech}(\lambda_t/2)$。结合 **v-预测**损失，加权为 $w_t = e^{-\lambda_t/2}$。

### （LDM-）Linear

LDM 使用 DDPM 调度的修改版。二者均为方差保持调度，即 $b_t=\sqrt{1-a_t^2}$，并在离散时间步 $t = 0, \dots, T-1$ 上通过扩散系数 $\beta_t$ 定义 $a_t = (\prod_{s=0}^t (1 - \beta_s))^{\frac{1}{2}}$。给定边界值 $\beta_0$ 与 $\beta_{T-1}$，DDPM 使用 $\beta_t = \beta_0 + \frac{t}{T-1} (\beta_{T-1} - \beta_0)$，LDM 使用 $\beta_t = \left( \sqrt{\beta_0} + \frac{t}{T-1} (\sqrt{\beta_{T-1}} - \sqrt{\beta_0}) \right)^2$。

### 4.1 面向 RF 模型的定制 SNR 采样器

RF 损失在 $[0, 1]$ 的所有时间步上均匀训练速度 $v_\Theta$。然而直观上，速度预测目标 $\epsilon - x_0$ 在 $[0, 1]$ 中间段的 $t$ 更困难：$t=0$ 时最优预测为 $p_1$ 的均值，$t=1$ 时为 $p_0$ 的均值。一般而言，将 $t$ 的分布从常用的均匀分布 $\mathcal{U}(t)$ 改为密度为 $\pi(t)$ 的分布，等价于加权损失 $\mathcal{L}_{w_t^\pi}$，其中

$$
w_t^\pi = \frac{t}{1-t}\pi(t)
$$

因此，我们通过更频繁采样中间时间步来赋予其更大权重。下面描述用于训练模型的时间步密度 $\pi(t)$。

![图：用于偏置训练时间步采样的 mode（左）与 logit-正态（右）分布。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/dists/modesampler.png)
![图：logit-正态采样器分布（右图与左图并列展示于原文）。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/dists/lnsampler.png)

#### Logit-正态采样

一种在中间步赋予更多权重的分布是 logit-正态分布。其密度为

$$
\pi_{\text{ln}}(t; m, s) = \frac{1}{s\sqrt{2\pi}} \frac{1}{t(1-t)}\exp\left(-\frac{(\text{logit}(t)-m)^2}{2s^2}\right)
$$

其中 $\text{logit}(t) = \log \frac{t}{1-t}$，$m$ 为位置参数，$s$ 为尺度参数。位置参数使我们可将训练时间步偏向数据 $p_0$（负 $m$）或噪声 $p_1$（正 $m$）。尺度参数控制分布宽度。

实践中，我们从正态分布 $u \sim \mathcal{N}(u; m, s)$ 采样随机变量 $u$，再通过标准 logistic 函数映射。

#### 带重尾的 Mode 采样

logit-正态密度在端点 $0$ 与 $1$ 处恒为零。为研究这是否对性能有不利影响，我们还使用在 $[0, 1]$ 上严格正密度的 time step 采样分布。对尺度参数 $s$，定义

$$
f_{\text{mode}}(u; s) = 1 - u - s\cdot\left(\cos^2\left(\frac{\pi}{2} u\right) - 1 + u\right)
$$

对 $-1 \leq s \leq \frac{2}{\pi -2}$，该函数单调，可用于从隐含密度 $\pi_{\text{mode}}(t; s) = \left| \frac{d}{dt} f_{\text{mode}}^{-1}(t) \right|$ 采样。尺度参数控制采样时更偏向中点（正 $s$）还是端点（负 $s$）。$s=0$ 时包含**均匀加权** $\pi_{\text{mode}}(t; s=0) = \mathcal{U}(t)$，这在先前整流流工作中被广泛使用。

#### CosMap

最后，我们在 RF 设定下也考虑第 4 节中的**余弦**调度。具体地，寻找映射 $f: u \mapsto f(u) = t,\; u \in [0, 1]$，使 log-SNR 与余弦调度匹配：$2 \log \frac{\cos(\frac{\pi}{2}u)}{\sin(\frac{\pi}{2}u)} = 2 \log \frac{1-f(u)}{f(u)}$。解得，对 $u \sim \mathcal{U}(u)$

$$
t = f(u) = 1 - \frac{1}{\tan(\frac{\pi}{2} u) + 1}
$$

密度为

$$
\pi_{\text{CosMap}}(t) = \left| \frac{d}{dt} f^{-1}(t) \right| = \frac{2}{\pi - 2\pi t + 2\pi t^2}
$$

---

## 5. 文本到图像架构

对图像的条件文本采样，模型需同时考虑文本与图像两种模态。我们使用预训练模型得到合适表示，然后描述扩散主干架构。整体概览见原文图（MM-DiT 架构示意，由 TikZ 绘制）。

整体设置遵循 LDM，在预训练自编码器潜空间中训练文本到图像模型。与图像编码为潜表示类似，我们也遵循先前方法，使用预训练、冻结的文本模型编码文本条件 $c$。细节见附录。

### 多模态扩散主干（MM-DiT）

架构建立在 DiT 之上。DiT 仅考虑类别条件图像生成，使用调制机制使网络同时条件于扩散过程时间步与类别标签。类似地，我们使用时间步 $t$ 与 $c_\text{vec}$ 的嵌入作为调制机制输入。然而，由于池化（pooled）文本表示仅保留文本输入的粗粒度信息，网络还需要序列表示 $c_\text{ctxt}$ 的信息。

我们构造由文本与图像输入嵌入组成的序列。具体地，对潜像素表示 $x \in \RR^{h \times w \times c}$ 的 $2\times 2$ patch 加位置编码并展平，得到长度 $\frac{1}{2} \cdot h \cdot \frac{1}{2} \cdot w$ 的 patch 编码序列。将该 patch 编码与文本编码 $c_\text{ctxt}$ 嵌入到共同维度后，拼接两序列。随后遵循 DiT，应用一系列调制注意力与 MLP。

由于文本与图像嵌入在概念上差异较大，我们对两种模态使用两套独立权重。这等价于每种模态各有一个独立 Transformer，但在注意力操作时将两模态序列拼接，使两种表示可在各自空间工作，同时考虑对方。

对扩展实验，我们以模型深度 $d$（注意力块数量）参数化模型规模：隐藏维度设为 $64\cdot d$（MLP 块中扩展为 $4\cdot64\cdot d$ 通道），注意力头数等于 $d$。

---

## 6. 实验

### 6.1 改进整流流

我们旨在理解式 (ODE) 所描述的免模拟归一化流训练中哪种方法最高效。为在不同方法间比较，我们控制优化算法、模型架构、数据集与采样器。此外，不同方法的损失不可比，也不一定与输出样本质量相关；因此需要允许方法间比较的评估指标。

我们在 ImageNet 与 CC12M 上训练模型，并在训练过程中用验证损失、CLIP 分数与 FID 评估训练权重与 EMA 权重（在不同采样器设置下：不同引导尺度与采样步数）。FID 按 Sauer 等人建议在 CLIP 特征上计算。所有指标在 COCO-2014 验证集上评估。训练与采样超参数细节见附录。

#### 结果

我们在两个数据集上各训练 61 种不同形式化，包括：

- $\epsilon$- 与 **v-预测**损失，linear（`eps/linear`、`v/linear`）与 cosine（`eps/cos`、`v/cos`）调度；
- RF 损失与 $\pi_{\text{mode}}(t; s)$（`rf/mode(s)`），$s$ 在 $-1$ 至 $1.75$ 间均匀取 7 个值，另加 $s=1.0$ 与 $s=0$（均匀时间步采样，`rf/mode`）；
- RF 损失与 $\pi_{\text{ln}}(t; m, s)$（`rf/lognorm(m, s)`），$(m, s)$ 在 $m\in[-1,1]$、$s\in[0.2,2.2]$ 的网格上取 30 组；
- RF 损失与 $\pi_{\text{CosMap}}(t)$（`rf/cosmap`）；
- EDM（`edm($P_m, P_s$)`），$P_m\in[-1.2,1.2]$、$P_s\in[0.6,1.8]$ 各 15 个值（$P_m, P_s=(-1.2, 1.2)$ 对应 Karras 等人默认参数）；
- 匹配 `rf` 与 `v/cos` 的 log-SNR 加权的 EDM（`edm/rf`、`edm/cos`）。

对每个 run，选取 EMA 权重下验证损失最小的步，收集 6 种采样器设置下（含/不含 EMA）的 CLIP 与 FID。

对 24 种采样器设置、EMA 权重与数据集组合，用非支配排序算法对形式化排名：反复计算在 CLIP 与 FID 上 Pareto 最优的变体，赋予当前迭代索引，移除后继续，直至全部排名。最后在 24 种控制设置上平均排名。

**表 1：变体全局排名**（在 EMA/非 EMA、两数据集、不同采样设置上平均的非支配排序）

| 变体 | 全部平均 | 5 步 | 50 步 |
|------|---------|------|-------|
| rf/lognorm(0.00, 1.00) | 1.54 | 1.25 | 1.50 |
| rf/lognorm(1.00, 0.60) | 2.08 | 3.50 | 2.00 |
| rf/lognorm(0.50, 0.60) | 2.71 | 8.50 | 1.00 |
| rf/mode(1.29) | 2.75 | 3.25 | 3.00 |
| rf/lognorm(0.50, 1.00) | 2.83 | 1.50 | 2.50 |
| eps/linear | 2.88 | 4.25 | 2.75 |
| rf/mode(1.75) | 3.33 | 2.75 | 2.75 |
| rf/cosmap | 4.13 | 3.75 | 4.00 |
| edm(0.00, 0.60) | 5.63 | 13.25 | 3.25 |
| rf | 5.67 | 6.50 | 5.75 |
| v/linear | 6.83 | 5.75 | 7.75 |
| edm(0.60, 1.20) | 9.00 | 13.00 | 9.00 |
| v/cos | 9.17 | 12.25 | 8.75 |
| edm/cos | 11.04 | 14.25 | 11.25 |
| edm/rf | 13.04 | 15.25 | 13.25 |
| edm(-1.20, 1.20) | 15.58 | 20.25 | 15.00 |

我们观察到 `rf/lognorm(0.00, 1.00)` 始终获得较好排名。它优于均匀时间步采样的整流流（`rf`），从而确认中间时间步更重要的假设。在所有变体中，**仅**修改时间步采样的整流流形式化优于先前使用的 LDM-Linear（`eps/linear`）。

部分变体在某些设置下表现好、其他设置差，例如 `rf/lognorm(0.50, 0.60)` 在 50 步采样时最佳（平均排名 8.5 在 5 步时较差）。表 2 展示代表性变体在 25 步下的指标。

**表 2：不同变体的 CLIP 与 FID（25 步采样）**

| 变体 | ImageNet CLIP | ImageNet FID | CC12M CLIP | CC12M FID |
|------|---------------|--------------|------------|-----------|
| rf | 0.247 | 49.70 | 0.217 | 94.90 |
| edm(-1.20, 1.20) | 0.236 | 63.12 | 0.200 | 116.60 |
| eps/linear | 0.245 | 48.42 | 0.222 | 90.34 |
| v/cos | 0.244 | 50.74 | 0.209 | 97.87 |
| v/linear | 0.246 | 51.68 | 0.217 | 100.76 |
| rf/lognorm(0.50, 0.60) | **0.256** | 80.41 | 0.233 | 120.84 |
| rf/mode(1.75) | 0.253 | **44.39** | 0.218 | 94.06 |
| rf/lognorm(1.00, 0.60) | 0.254 | 114.26 | **0.234** | 147.69 |
| rf/lognorm(-0.50, 1.00) | 0.248 | 45.64 | 0.219 | **89.70** |
| rf/lognorm(0.00, 1.00) | 0.250 | 45.78 | 0.224 | 89.91 |

![图：整流流在少步采样时更高效。25 步及以上时，仅 rf/lognorm(0.00, 1.00) 仍可与 eps/linear 竞争。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/fid_vs_steps_traj.png)

最后在定性上，不同形式化在减少采样步数时，整流流总体表现更好，且性能下降更少。

### 6.2 改进模态特定表示

找到可与 LDM-Linear、EDM 竞争甚至超越的整流流形式化后，我们将其应用于高分辨率文本到图像合成。最终性能不仅取决于训练形式化，还取决于神经网络参数化及所用图像与文本表示质量。以下各节描述如何改进这些组件，然后在 6.3 节扩展最终方法。

#### 6.2.1 改进自编码器

潜扩散模型通过在预训练自编码器潜空间中操作实现高效，该自编码器将 RGB 输入 $X\in \RR^{H\times W\times 3}$ 映射到更低维空间 $x=E(X) \in \RR^{h\times w \times d}$。自编码器重建质量为潜扩散训练后可达到图像质量的上界。与 Emu 类似，我们发现增加潜通道数 $d$ 可显著提升重建性能（表 3）。直观上，预测更高 $d$ 的潜变量是更难任务，更大容量模型应能表现更好，最终获得更高图像质量。图 FID 研究证实了该假设：$d=16$ 自编码器在样本 FID 上扩展性能更好。本文余下部分选择 $d=16$。

**表 3：改进自编码器——不同通道配置的重建指标（下采样因子 $f=8$）**

| 指标 | 4 通道 | 8 通道 | 16 通道 |
|------|--------|--------|---------|
| FID () | 2.41 | 1.56 | **1.06** |
| 感知相似度 () | 0.85 | 0.68 | **0.45** |
| SSIM ($\uparrow$) | 0.75 | 0.79 | **0.86** |
| PSNR ($\uparrow$) | 25.12 | 26.40 | **28.62** |

#### 6.2.2 改进文本标注（Caption）

Betker 等人证明，合成 caption 可大幅改进大规模训练的文本到图像模型。这是因为大规模图像数据集附带的人工 caption 往往过于简单，过度聚焦图像主体，通常省略背景、构图或显示文字等细节。我们遵循其方法，使用 CogVLM 为大规模图像数据集生成合成标注。由于合成 caption 可能使模型遗忘 VLM 知识库中不存在的概念，我们使用 50% 原始与 50% 合成 caption 的比例。

为评估该 caption 混合的效果，我们训练两个 $d=15$ 的 MM-DiT 模型各 250k 步，一个仅用原始 caption，另一个用 50/50 混合。GenEval 结果（表 4）表明，加入合成 caption 的模型明显优于仅用原始 caption 的模型。余下工作使用 50/50 混合。

**表 4：改进 caption——GenEval 成功率（%）**

| 任务 | 原始 caption | 50/50 混合 |
|------|-------------|-----------|
| 颜色归因 | 11.75 | 24.75 |
| 颜色 | 71.54 | 68.09 |
| 位置 | 6.50 | 18.00 |
| 计数 | 33.44 | 41.56 |
| 单物体 | 95.00 | 93.75 |
| 双物体 | 41.41 | 52.53 |
| **总分** | 43.27 | **49.78** |

#### 6.2.3 改进文本到图像主干

本节比较现有基于 Transformer 的扩散主干与我们提出的多模态 Transformer 主干 MM-DiT。MM-DiT 专为处理不同域（此处为文本与图像 token）而设计，使用（两套）不同可训练权重。我们在 6.1 节实验设置下，在 CC12M 上比较 DiT、CrossDiT（DiT 但对文本 token 交叉注意力而非序列拼接）与 MM-DiT。对 MM-DiT，我们比较两套与三套权重（后者分别处理 CLIP 与 T5 token）。DiT（按 5 节方式拼接文本与图像 token）可视为 MM-DiT 对所有模态共享一套权重的特例。我们还考虑 UViT 作为常用 U-Net 与 Transformer 变体之间的混合架构。

![图：各架构在 CC12M 上的训练动态（验证损失、CLIP、FID）。MM-DiT 在所有指标上表现更优。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/archs_squeezed/val_loss_level_avg.png)
![图：CLIP 与 FID 对比（与上图并列于原文）。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/archs_squeezed/clip_fid_sampler_default_ema_True.png)

收敛行为分析表明：Vanilla DiT 弱于 UViT；CrossDiT 优于 UViT，尽管 UViT 初期学习更快。MM-DiT 显著优于交叉注意力与 vanilla 变体。三套权重相对两套仅小幅提升（但参数与 VRAM 增加），余下工作选择两套。

### 6.3 大规模训练

扩展前，我们过滤并预编码数据以确保安全、高效预训练。随后，关于扩散形式化、架构与数据的先前考虑在本节汇总，将模型扩展至 80 亿参数。

#### 6.3.1 数据预处理

**预训练缓解措施**

训练数据显著影响生成模型能力，数据过滤可有效约束不期望能力。大规模训练前，我们按以下类别过滤数据：(i) 色情内容：用 NSFW 检测模型过滤；(ii) 美学：移除预测低分图像；(iii) 复述（Regurgitation）：用基于聚类的去重方法移除感知与语义重复；见附录。

**预计算图像与文本嵌入**

模型使用多个预训练、冻结网络的输出作为输入（自编码器潜变量与文本编码器表示）。由于训练期间这些输出恒定，我们对整个数据集预计算一次。详见附录。

#### 6.3.2 高分辨率微调

**QK 归一化**

一般而言，所有模型在 $256^2$ 像素低分辨率图像上预训练，随后在高分辨率混合宽高比上微调。我们发现，迁移到高分辨率时，混合精度训练可能不稳定且损失发散。可切换全精度训练以缓解该问题，但相较混合精度约慢 $2\times$。更高效替代方案来自判别式 ViT 文献：Dehghani 等人观察到大型 ViT 训练发散是因为注意力熵不可控地增长；他们在注意力操作前对 Q、K 归一化以避免此问题。我们遵循该方法，在 MM-DiT 两流中使用带可学习尺度的 RMSNorm。

![图：QK 归一化的效果。归一化 Q、K 嵌入可防止注意力 logit 增长不稳定（左），该不稳定会导致注意力熵坍缩（右）。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/qk_norm/02_max_attn_logit_qk.png)
![图：注意力熵（与左图并列）。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/qk_norm/02_attn_entropy_qk.png)

额外归一化防止注意力 logit 增长不稳定，确认 Dehghani 与 Wortsman 等人的发现，并在 AdamW 优化器 $\epsilon=10^{-15}$ 下使 bf16 混合精度训练可行。该技术也可用于预训练时未使用 qk 归一化的模型：模型快速适应额外归一化层并更稳定训练。需指出，虽一般有助于稳定大型模型训练，但并非普适配方，可能需依具体训练设置调整。

**可变宽高比的位置编码**

在固定 $256\times256$ 分辨率训练后，我们旨在提高分辨率并支持灵活宽高比推理。因使用 2D 位置频率嵌入，需依分辨率调整。多宽高比设定下，直接插值嵌入（如 ViT）不能正确反映边长。我们改用扩展与插值位置网格的组合，再频率嵌入。

对目标分辨率 $S^2$ 像素，使用分桶（bucket）采样使每个 batch 为同质尺寸 $H \times W$（$H \cdot W \approx S^2$）。对最大/最小训练宽高比，得到潜空间（8 倍下采样后再 2 倍 patch）中最大宽 $w_\text{max}$、高 $h_\text{max}$ 及 $s=S/16$。构造垂直位置网格 $((p-\frac{h_\text{max}-s}{2})\cdot\frac{256}{S})_{p=0}^{{h_\text{max}-1}$ 及对应水平网格，嵌入前从所得 2D 网格中心裁剪。

**分辨率相关的时间步调度平移**

直观上，更高分辨率有更多像素，需要更多噪声才能破坏其信号。设分辨率为 $n=H\cdot W$ 像素。考虑"常数"图像，即每像素值为 $c$。前向过程产生 $z_t = (1-t)c \mathbbm{1} + t \epsilon$，其中 $\mathbbm{1}$ 与 $\epsilon \in \mathbb{R}^n$。$z_t$ 提供 $n$ 个随机变量 $Y=(1-t)c + t \eta$（$c,\eta\in\mathbb{R}$，$\eta$ 标准正态）的观测。故 $\mathbb{E}(Y)=(1-t)c$，$\sigma(Y)=t$，可通过 $c=\frac{1}{1-t}\mathbb{E}(Y)$ 恢复 $c$，$c$ 与其样本估计 $\hat{c} = \frac{1}{1-t}\sum_{i=1}^n z_{t,i}$ 的误差标准差为 $\sigma(t, n)=\frac{t}{1-t}\sqrt{\frac{1}{n}}$。若已知 $z_0$ 在像素上为常数，$\sigma(t, n)$ 表示关于 $z_0$ 的不确定度。例如，宽高加倍使任意 $0<t<1$ 的不确定度减半。可将分辨率 $n$ 上的时间步 $t_n$ 映射到分辨率 $m$ 上的 $t_m$，使 $\sigma(t_n, n)=\sigma(t_m, m)$。解得

$$
t_m = \frac{\sqrt{\frac{m}{n}}t_n}{1+(\sqrt{\frac{m}{n}}-1)t_n}
$$

![图：高分辨率时间步平移。右上：应用平移函数的人类质量偏好。下：$512^2$ 模型在 $\sqrt{m/n}=1.0$（上）与 $3.0$（下）下训练与采样。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/timeshift_v1.png)

人类偏好研究表明，平移系数大于 $1.5$ 的样本更受偏好，更高平移值之间差异较小。后续实验在 $1024\times 1024$ 训练与采样中使用 $\alpha=3.0$。注意上式意味着 log-SNR 平移 $\log \frac{n}{m}$，类似 Hoogeboom 等人。

在 $1024 \times 1024$ 平移训练后，我们使用直接偏好优化（DPO）对齐模型（见附录）。

#### 6.3.3 结果

![图：扩展的定量效应。验证损失随模型规模与训练步数平滑下降；与 GenEval、人类偏好、T2I-CompBench 等指标强相关。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/scale_val_squeeze/00_coco_val_loss_train-step.png)

我们在 $256^2$ 分辨率、batch 4096、预编码数据上训练不同参数规模的 MM-DiT 各 500k 步，每 50k 步在 COCO 上报告验证损失。为降低验证损失噪声，在 $t\in(0,1)$ 等距采样损失 level 并分别计算，再除 $t=1$ 外平均。

类似地，我们对视频进行 MM-DiT 初步扩展研究：从预训练图像权重出发，额外使用 2× 时间 patch；按 Align-your-Latents 将时间轴折叠到 batch 轴；每层注意力后在视觉流重排表示并在空间注意力后、最终 FFN 前加入全时空 token 注意力。视频模型在 16 帧、$256^2$ 像素、batch 512 上训练 140k 步，每 5k 步在 Kinetics 上报告验证损失。

图像与视频域均观察到：增大模型规模与训练步数时验证损失平滑下降。验证损失与 CompBench、GenEval 及人类偏好高度相关，支持其作为简单通用的性能度量。结果对图像与视频模型均未显示饱和。

应用 6.3.2 节方法并提高训练图像分辨率后，最大模型在多数 GenEval 类别领先，总分超越 DALL-E 3（表 5）。

**表 5：GenEval 对比（节选）**

| 模型 | Overall | Single | Two | Counting | Colors | Position | Color Attr. |
|------|---------|--------|-----|----------|--------|----------|-------------|
| SDXL | 0.55 | 0.98 | 0.74 | 0.39 | 0.85 | 0.15 | 0.23 |
| DALL-E 3 | 0.67 | 0.96 | 0.87 | 0.47 | 0.83 | **0.43** | 0.45 |
| Ours (d=38), $512^2$ | 0.68 | 0.98 | 0.84 | 0.66 | 0.74 | 0.40 | 0.43 |
| Ours (d=38), $512^2$ w/DPO | 0.71 | 0.98 | 0.89 | **0.73** | 0.83 | 0.34 | 0.47 |
| Ours (d=38), $1024^2$ w/DPO | **0.74** | **0.99** | **0.94** | 0.72 | **0.89** | 0.33 | **0.60** |

![图：与当前闭源/开源 SOTA 文本到图像模型的人类偏好评估（Parti-prompts：视觉质量、提示遵循、排版）。](../../../arxiv/base_models/sd3_rectified_flow/extracted/img/baseline_comp.png)

$d=38$ 模型在 Parti-prompts 的**视觉美学**、**提示遵循**与**排版生成**上优于当前专有与开源 SOTA。人类评估问题为：

- **提示遵循：** 哪张图像更**代表**上述文本并**忠实**遵循？
- **视觉美学：** 给定提示，哪张图像**质量更高**、**美学上更 pleasing**？
- **排版：** 哪张图像更准确地显示描述中的文字？拼写更准确优先，忽略其他方面。

**表 6：模型规模对采样效率的影响——相对 CLIP 分数下降（%，相对 50 步）**

| depth | 5/50 步 | 10/50 步 | 20/50 步 | 路径长度 |
|-------|---------|----------|----------|----------|
| 15 | 4.30 | 0.86 | 0.21 | 191.13 |
| 30 | 3.59 | 0.70 | 0.24 | 187.96 |
| 38 | 2.71 | 0.14 | 0.08 | 185.96 |

更大模型不仅性能更好，达到峰值性能所需步数更少——我们归因于更强鲁棒性与更好拟合整流流直路径目标，从而得到更短的路径长度。

**灵活文本编码器**

使用多个文本编码器的主要动机是提升整体性能；我们现表明该选择还增加 MM-DiT 整流流在推理时的灵活性。训练时使用三个文本编码器，各自以 46.3% 的概率丢弃（dropout）。故推理时可使用任意子集，在性能与内存效率间权衡——对需大量 VRAM 的 T5-XXL（47 亿参数）尤其重要。有趣的是，仅用两个 CLIP 编码器、T5 嵌入置零时性能下降有限。仅对含高度详细场景描述或较长书面文字的复杂提示，三编码器才有显著增益。人类评估中，移除 T5 对美学无影响（50% 胜率），对提示遵循影响小（46%），对文字生成贡献更显著（38% 胜率）。

---

## 7. 结论

本工作对文本到图像合成中的整流流模型进行了扩展分析。我们提出了改进整流流训练的时间步采样，优于先前潜扩散训练的扩散形式化，并保留整流流在少步采样机制下的有利性质。我们还展示了考虑文本到图像任务多模态性质的 Transformer 基 **MM-DiT** 架构的优势。最后，我们将该组合扩展至 80 亿参数与 $5\times 10^{22}$ 训练 FLOPs，表明验证损失改进与现有文本到图像基准及人类偏好评估相关。生成建模改进与可扩展多模态架构的结合，使性能可与当前最优专有模型竞争。扩展趋势未显示饱和，使我们对未来继续改进模型保持乐观。

---

## 更广泛影响

本文旨在推进机器学习领域，尤其是图像合成。我们工作有许多潜在社会影响，此处无特别需要强调者。对扩散模型一般影响的深入讨论，请读者参阅 Po 等人。

---

## 附录

### A. 流匹配细节

#### A.1 免模拟流训练细节

遵循 Lipman 等人，为说明 $u_t(z)$ 生成 $p_t$，连续性方程提供必要充分条件：

$$
\frac{d}{dt} p_t(x) + \nabla \cdot [p_t(x) v_t(x)] = 0 \leftrightarrow \text{$v_t$ 生成概率密度路径 $p_t$}
$$

因此只需证明

$$
\begin{aligned}
- \nabla \cdot [u_t(z) p_t(z)] &= - \nabla \cdot \left[\mathbb{E}_{\epsilon} u_t(z \vert \epsilon) \frac{p_t(z \vert \epsilon)}{p_t(z)}  p_t(z) \right] \\
&= \mathbb{E}_{\epsilon} - \nabla \cdot  [u_t(z \vert \epsilon) p_t(z \vert \epsilon) ] \\
&= \mathbb{E}_{\epsilon}  \frac{d}{dt} p_t(z \vert \epsilon) = \frac{d}{dt} p_t(z)
\end{aligned}
$$

其中对 $u_t(z \vert \epsilon)$ 使用连续性方程，因 $u_t(z \vert \epsilon)$ 生成 $p_t(z \vert \epsilon)$。

目标 $\mathcal{L}_{FM} \leftrightharpoons \mathcal{L}_{CFM}$ 的等价性来自：

$$
\begin{aligned}
\mathcal{L}_{FM}(\Theta) &= \mathbb{E}_{t, p_t(z)}\| v_{\Theta}(z, t) - u_t(z)  \|_2^2 \\
&= \mathbb{E}_{t, p_t(z)}\| v_{\Theta}(z, t) \|_2^2  - 2 \mathbb{E}_{t, p_t(z)} \langle v_{\Theta}(z, t),  u_t(z) \rangle + c \\
&=\mathbb{E}_{t, p_t(z)}\| v_{\Theta}(z, t) \|_2^2  - 2 \mathbb{E}_{t, p_t(z | \epsilon),  p(\epsilon) } \langle v_{\Theta}(z, t), u_t(z | \epsilon ) \rangle + c \\
&= \mathbb{E}_{t, p_t(z | \epsilon), p(\epsilon) }\| v_{\Theta}(z, t) - u_t(z | \epsilon)  \|_2^2 + c'  = \mathcal{L}_{CFM}(\Theta) + c'
\end{aligned}
$$

其中 $c, c'$ 不依赖 $\Theta$。

### B. 图像与文本表示细节

**潜图像表示**

遵循 LDM，使用预训练自编码器将 RGB 图像 $X \in \RR^{H\times W\times 3}$ 表示为更小潜空间 $x=E(X) \in \RR^{h\times w \times d}$。空间下采样因子为 8，$h=\frac{H}{8}$、$w=\frac{W}{8}$，在 6.2.1 节实验不同 $d$。始终在潜空间应用前向过程，采样表示 $x$ 后通过解码器 $D$ 解码回像素空间 $X = D(x)$。按 LDM 在训练数据子集上全局计算均值与标准差归一化潜变量。

**文本表示**

类似地，使用预训练、冻结文本模型编码文本条件 $c$。所有实验组合 CLIP 与 encoder-decoder 文本模型：用 CLIP L/14 与 OpenCLIP bigG/14 的文本编码器，拼接池化输出（768 与 1280 维）得 $c_\text{vec} \in \RR^{2048}$；通道维拼接倒数第二层隐藏表示得 $c_\text{ctxt}^{\text{CLIP}} \in \RR^{77\times2048}$；T5-v1.1-XXL 编码器 final hidden 为 $c_\text{ctxt}^{\text{T5}}\in\RR^{77\times4096}$；将 CLIP 上下文沿通道零填充至 4096 维，再与 T5 沿序列轴拼接，得 $c_\text{ctxt}\in\RR^{154\times4096}$。$c_\text{vec}$ 与 $c_\text{ctxt}$ 以第 5 节所述两种方式使用。

### C. 6.1 节实验预备

**数据集**

ImageNet：为类别添加 ``a photo of a \<class name\>'' 形式标注。CC12M：更贴近真实场景的文本到图像数据集。

**优化**

全局 batch 1024，AdamW 学习率 $10^{-4}$，1000 步线性 warmup。混合精度训练，EMA 每 100 batch 更新，衰减 0.99。Classifier-free guidance：三个文本编码器输出各自以 46.4% 概率置零。

**评估**

在 COCO-2014 验证集上用 CLIP、FID 与验证损失。损失在 $[0,1]$ 八个等距 $t$ 值上分层评估。每种采样器生成 1000 样本（不同引导尺度与步数），用 CLIP L/14 计算 CLIP 分数与 FID。采样始终用 Euler 离散化，六种设置：50 步 + guidance 1.0/2.5/5.0，以及 5/10/25 步 + guidance 5.0。

### D. 直接偏好优化（DPO）

DPO 是用偏好数据微调 LLM 的技术，近期已适配于文本到图像扩散模型偏好微调。我们对 2B 与 8B 基座模型应用 Wallace 等人的方法：对所有线性层引入 rank 128 的 LoRA，2B/8B 分别微调 4k/2k 迭代。人类偏好研究表明基座模型可有效对齐人类偏好。DPO 微调通常产生美学质量更高、拼写更准确的样本。

### E. 指令式图像编辑微调

常见做法是在 channel 维拼接输入图像潜变量与扩散目标噪声潜变量后输入 U-Net。我们同样拼接后再 patch，证明该方法适用于 MM-DiT。在类似 InstructPix2Pix 的编辑任务及 inpainting、分割、上色、去模糊、ControlNet 等任务上微调 2B 基座。2B Edit 模型可操纵给定图像中的文字，尽管训练数据未含文字操纵任务；在相同数据上训练的 SDXL 编辑模型未能复现类似结果。

### F. 大规模文本到图像训练的数据预处理

**预计算嵌入**

优势：(i) 训练时 GPU 无需加载编码器，降低内存；(ii) 首 epoch 后跳过编码前向，节省时间（表 7）。

**表 7：预编码冻结输入网络的关键数据**

| 模型 | 内存 [GB] | 前向 [ms] | 存储 [kB/样本] | 训练步增量 [%] |
|------|-----------|-----------|----------------|----------------|
| VAE (Enc) | 0.14 | 2.45 | 65.5 | 13.8 |
| CLIP-L | 0.49 | 0.45 | 121.3 | 2.6 |
| CLIP-G | 2.78 | 2.77 | 202.2 | 15.6 |
| T5 | 19.05 | 17.46 | 630.7 | 98.3 |

劣势：无法每 epoch 随机增强；预计算时用 square-center crop。高分辨率微调先 resize/crop 到 aspect ratio bucket 再预计算。文本编码器稠密输出存储大；语言模型嵌入以半精度保存，实践中未观察到性能下降。

**防止图像记忆**

生成模型记忆训练样本可导致多种问题。我们扫描数据集移除重复样本。去重使用 SSCD 作为 backbone，Autofaiss + FAISS 聚类（$N=16000$ 簇）。SSCD 阈值 0.5 时，按 Carlini 等人方法评估，记忆样本约减少 $5\times$。

---

*译本说明：章节编号与原文 main.tex 正文结构对齐；背景（第 2 节）在原文附录中给出，此处置于正文前以便阅读。公式保留 LaTeX；表格由原文 TeX 转为 Markdown。*
