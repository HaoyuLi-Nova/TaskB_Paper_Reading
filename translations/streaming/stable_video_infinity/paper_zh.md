# Stable Video Infinity：利用错误回收生成无限长度视频

**作者：** Wuyang Li、Wentao Pan、Po-Chien Luan、Yang Gao、Alexandre Alahi

**机构：** VITA@EPFL

**出处：** ICLR 2026 Oral（arXiv:2510.09212）

**arXiv：** [2510.09212](https://arxiv.org/abs/2510.09212)

**项目页：** https://stable-video-infinity.github.io/homepage/

**原文 TeX：** [`arxiv/streaming/stable_video_infinity/extracted/arxiv.tex`](../../../arxiv/streaming/stable_video_infinity/extracted/arxiv.tex)

---

## 摘要

我们提出 **Stable Video Infinity（SVI）**，一种能够生成无限长度视频的方法，同时保持较高的时间一致性、合理的场景转换以及可控的流式故事线。现有长视频方法通常通过手工设计的 anti-drifting 机制（例如修改 noise scheduler、进行 frame anchoring）来**缓解累积误差**，但仍受限于单一 prompt 外推，因而容易生成场景同质、动作重复的视频。

我们发现，根本挑战并不只是误差累积，而是训练假设与测试时自回归现实之间存在关键差异：训练时模型看到的是干净数据，而测试时模型必须以自己生成的、带有误差的输出作为条件。为弥合这一 hypothesis gap，SVI 引入新的 **Error-Recycling Fine-Tuning（ERFT，错误回收微调）**：将 Diffusion Transformer（DiT）自己生成的误差回收为监督 prompt，促使 DiT **主动识别并修正自身错误**。这通过闭环回收完成：注入、收集并将错误存入 bank，从而在带误差反馈上自回归学习。

具体而言：(i) 注入 DiT 历史生成中的错误，干预干净输入，从而在 flow matching 中模拟误差累积轨迹；(ii) 通过单步双向积分高效近似预测，并利用 residual 计算误差；(iii) 将错误按照离散 timestep 动态存入 replay memory，再为新输入重新采样。SVI 可以在不增加推理成本的情况下，把视频从数秒扩展到无限时长，并兼容 audio、skeleton、text stream 等多种条件。我们在 consistent、creative 和 conditional 三类 benchmark 上进行评估，验证了其通用性和 SOTA 表现。

---

## 1. 引言

> *Failure is simply the opportunity to begin again, this time more intelligently.*
>
> —— Henry Ford

随着模型和数据规模扩大，video Diffusion Transformer（DiT）（Wang et al., 2025；Kong et al., 2024；Liu et al., 2024；Hong et al., 2022）在生成逼真且时间连贯的视频方面取得了很大进展，并支持开放域内容创作。尽管如此，该社区仍受限于视频长度，通常约为 5 秒（Wang et al., 2025）。这主要是因为 **error accumulation（误差累积，也称 drifting）**：自回归地以前一段生成结果为条件时，预测误差会随时间复合，导致图像保真度、运动稳定性和语义可控性逐步下降（图 1a）。

在此背景下，现有方案大致分为三类：**(i) noise modification**，增强或修改 noise schedule，减少对历史帧的依赖（Chen et al., 2024；Ruhe et al., 2024）；**(ii) frame anchoring**，用无误差的 reference image 作为 anchor，降低对带误差帧的依赖（Henschel et al., 2025）；**(iii) improved sampling**，例如 masked-noise guidance（Song et al.）和 anti-drifting sampling（Zhang et al., 2025）。

![图 1. 比较 (a) video generative DiT、(b) restoration DiT 与 (c) 本文 SVI：第 1 行是方案，第 2 行是训练–测试 hypothesis gap，第 3 行是结果。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/intro.png)

然而，现有方法主要是在**缓解**而非**修正**累积误差，因而有两个关键局限：**长度受限**（一般约 10 秒，最多约 1 分钟），以及带重复运动的 **scene homogeneity bias**。实践上，多数方法本质上是在单一 prompt 控制下外推原片段，而不是生成可由 prompt stream 故事线轻松控制的真正长视频。因此，当前方案难以满足许多创意需求，例如需要频繁而合理切换场景的短片拍摄，或小时级在线演示。

为此，我们希望治本而非治标：从根本上修正累积误差本身，而不是仅仅减轻其影响。观察误差造成的伪影（见图 6），我们经验上发现它们与图像 restoration 社区中常见的 blur、color shift 等退化高度一致。鉴于 DiT 在底层视觉中的 SOTA 地位，这些退化对容量更大的视频 DiT（例如 14B）本不应困难。令人惊讶的是实践中恰恰相反：为什么这些强大模型对这类误差如此敏感，以至于迅速崩溃？

我们揭示，根本挑战在于**训练与测试之间的 hypothesis gap**。训练 generative DiT 时（图 1a），flow matching 的历史轨迹被假设为无误差；测试时这一假设很容易被打破，因为模型自回归地使用带预测误差的先前生成结果（第 3 节给出数学说明）。相对地，restoration DiT（图 1b）在训练和测试时都以带误差输入为条件，因此具有错误鲁棒性。为弥合这一差距，我们采取新视角：**把模型自己生成的错误回收为监督 prompt，通过错误反馈训练 DiT 自回归地修正自身错误。**

本文提出 **Stable Video Infinity（SVI）**，能够沿长故事线生成时间连贯、视觉合理的无限长度视频。如图 1c，SVI 采用新颖的 **Error-Recycling Fine-Tuning**：把 DiT 自生成误差重新用作监督信号，使模型通过自回归错误反馈迭代精炼输出。具体地，我们 (i) 向干净输入注入历史错误以模拟退化，(ii) 用双向单步积分近似预测并计算误差，(iii) 在离散 timestep 上从 replay memory 动态保存并选择性重采样错误。由此可高效释放视频 DiT 中的 restoration 能力，在生成中主动纠错。此外，SVI 相对先前工作还有若干涌现优势。**数据轻量**：LoRA 微调只需小规模数据；**高效**：零额外推理成本；**通用**：支持开放域控制信号，例如 audio 与 skeleton（图 7c）。

贡献有五点。(1) SVI 通过主动纠错，把视频长度限制从数秒扩展到无限。(2) 我们系统分析长视频生成中的训练–测试 hypothesis gap，并理论形式化两类误差。(3) 为弥合该差距，提出 error-recycling fine-tuning：动态计算、保存并选择性注入错误到干净输入，预测 error-recycled velocity。(4) 将 SVI 扩展为面向不同应用的模型族，例如 talking 与 dancing（图 7c）。(5) 提出覆盖短/长、一致/创意设定的综合 benchmark，对齐多样化真实用户需求。

---

## 2. 相关工作

**开放域视频生成。** 随着模型（Hu et al., 2024）与数据（Weissenborn et al., 2020）规模扩大，商业级视频生成模型（Liu et al., 2024；Video World Simulators, 2024；Yang et al., 2024；Blattmann et al., 2023；Ho et al., 2022；Lin et al., 2024）——例如 Wan（Wang et al., 2025）与 Hunyuan（Chen et al., 2025）——在高质量短视频上取得显著进展。在此基础上，社区围绕多样化目标发展二次创作，引入面向任务的控制（audio、skeleton 等），生成 talking（Kong et al., 2025；Chen et al., 2025）、dancing（Wang et al., 2025）、navigation（Agarwal et al., 2025；Hassan et al., 2025）、gaming（Yu et al., 2025；Che et al., 2024）等内容。尽管进展可观，短时长仍是开放挑战，限制实际应用。

**长视频生成。** 由于误差累积，长视频仍是开放问题，现有解法有三条路线。(i) Modified scheduler：提升 ODE solver 对误差的鲁棒性。部分工作通过 noise rescheduling 扩展视频（Qiu et al., 2024；Hoeg et al., 2024）；另一些工作修改并增强 noise schedule 以降低对过去帧的依赖（Chen et al., 2024；Ruhe et al., 2024；Song et al.）。(ii) Frame anchoring：用干净图像作为一致参考，包括专门的 anchor 设计（Henschel et al., 2025；Weng et al., 2024）以及基于规划的优化（Video World Simulators, 2024；Zhao et al., 2024；Yang et al., 2024）。(iii) Error-robust architecture：通过 bidirectional distillation（Yin et al., 2025；Huang et al., 2025）、distributed generation（Tan et al., 2024）、anti-drifting sampling（Zhang et al., 2025；Gu et al., 2025）、改进 attention（Kodaira et al., 2025；Lu et al., 2024）和 mixture of context（Cai et al., 2025）提升自回归长程一致性。与这些工作不同，我们回收错误并鼓励 DiT 修正自己产生的错误。同期工作见附录 A.3。

![图 2. 训练–测试 hypothesis gap。(a) 训练假设历史轨迹与中间状态无误差，测试时被两类误差轻易打破。(b) 回归特性导致的 predictive error 影响轨迹终点 $X_{\mathrm{vid}}$。(c) 含误差图像造成的 conditional error 也影响起点 $\tilde{X}_{\mathrm{noi}}^{\mathrm{img}}$。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/pre_error.png)

---

## 3. 预备知识与动机

### 3.1 长视频训练中的无误差假设

**记号。** 下文用 hat $\hat{(\cdot)}$ 表示模型预测变量，tilde $\tilde{(\cdot)}$ 表示注入误差的变量，不带上标 ${(\cdot)}$ 表示干净（无误差）变量。

**训练。** Flow matching 使 DiT 能够连续时间生成。为深入训练–测试 hypothesis gap，我们从无误差 flow matching（图 2a）的图像到视频训练出发：目标是学习从联合 noise / reference-image 分布 $X_{\mathrm{noi}}^{\mathrm{img}}$ 到 video latent $X_{\mathrm{vid}}$ 的 ODE。训练时，假设**无误差 video latent** $X_{\mathrm{vid}}$、噪声 $X_{\mathrm{noi}}\sim\mathcal{N}(0,I)$、timestep $t\in[0,1]$、**无误差 reference image** $X_{\mathrm{img}}$ 以及可选多模态条件 $C$，训练目标为：

$$
\mathcal{L}=\mathbb{E}_{X_{\mathrm{noi}},X_{\mathrm{vid}},X_{\mathrm{img}},C,t}\big|u(X_t,X_{\mathrm{img}},C,t;\theta)-V_t\big|^2,
\tag{1}
$$

其中 $X_t=t\cdot X_{\mathrm{vid}}+(1-t)\cdot X_{\mathrm{noi}}$ 为中间状态，$\hat{V}_t=u(X_t,X_{\mathrm{img}},C,t;\theta)$ 为模型 $\theta$ 预测的 velocity，$V_t=\frac{dX_t}{dt}=X_{\mathrm{vid}}-X_{\mathrm{noi}}$ 为真值。

**测试。** 生成的 video latent 在 $t=1$ 处得到 $X_1=X_{\mathrm{vid}}$，从噪声 $X_0$ 出发共 $N_{\mathrm{test}}$ 步采样。实践中，将单位区间离散为 $0=t_0<t_1<\cdots<t_{N_{\mathrm{test}}}=1$，并用数值 ODE solver（Esser et al., 2024）逐步积分：

$$
X_{t_{k+1}}=X_{t_k}+(t_{k+1}-t_k)\cdot u(X_{t_k},X_{\mathrm{img}},t_k;\theta),
\tag{2}
$$

其中 $k<N_{\mathrm{test}}-1$，$X_{t_{k+1}}$ 为下一步生成。

### 3.2 误差污染的推理

测试阶段会出现两类由无误差训练假设导致的误差。

**Single-clip predictive error（单片段预测误差）。** 式 (1) 的训练假设 $X_t$ 来自干净 latent $X_{\mathrm{vid}}$ 与正确历史轨迹。推理中（图 2b）该假设很容易被打破，因为 $\tilde{X}_t$ 来自带固有误差的预测轨迹。由于 MSE 回归特性，预测 velocity $\hat{V}_t=u(\tilde{X}_t,X_{\mathrm{img}},C,t;\theta)$ 与真值 $V_t=u(X_t,X_{\mathrm{img}},C,t;\theta)$ 之间会持续存在差异 $E_t$。该偏移在每一步采样中逐渐累积，因为后续步骤会复用先前积分得到的 velocity。因此，对独立生成而言，逐步微小误差沿式 (2) 的 ODE 轨迹积分，产生偏移后的预测 latent $\hat{X}_{\mathrm{vid}}$，定义为 **single-clip sampling error**：

$$
E=\hat{X}_{\mathrm{vid}}-X_{\mathrm{vid}}.
$$

概念上，$E$ 通常较小，具有对抗攻击性质（Goodfellow et al., 2014），在短视频生成中感知退化可忽略。

**Cross-clip conditional error（跨片段条件误差）。** 如图 2c，自回归生成后续片段时，模型使用来自 $\hat{X}_{\mathrm{vid}}$ 的**含误差**帧 $\tilde{X}_{\mathrm{img}}$（图 2b），而不是训练式 (1) 中使用的**干净** $X_{\mathrm{img}}$，从而使轨迹起点从 $X_{\mathrm{vid}}^{\mathrm{img}}$ 偏移到 $\tilde{X}_{\mathrm{vid}}^{\mathrm{img}}$。由于式 (1) 在**干净输入**上优化，这些**被误差污染的样本** $\tilde{X}_{\mathrm{vid}}^{\mathrm{img}}$ 相对干净训练数据是分布外的，会严重干扰 DiT 预测 $\hat{V}_t=u(\tilde{X}_t,\tilde{X}_{\mathrm{img}},C,t;\theta)$。由于其与训练中期望 velocity $V_t=u(X_t,X_{\mathrm{img}},C,t;\theta)$ 差距很大，该误差会造成更偏的 $X_{\mathrm{vid}}$ 预测。我们将这一累积误差定义为 **cross-clip conditional error**：

$$
E=\hat{X}_{\mathrm{vid}}-X_{\mathrm{vid}},
$$

其中 $\hat{X}_{\mathrm{vid}}$ 由带误差的 $\tilde{X}_{\mathrm{img}}$ 积分 $\hat{V}_t$ 得到。

**误差累积与放大。** 在自回归跨片段条件化中，两类误差会**累积并互相强化**：predictive error 使生成的 video latent 漂移，放大轨迹起点误差，进而进一步增大 predictive error。该反馈环可迅速导致生成视频灾难性退化。

### 3.3 弥合训练–测试差距

综上，长视频生成的本质挑战是无误差训练与含误差推理之间的 hypothesis gap。更有害的是，DiT 在自回归条件下倾向于**累积并放大这些误差**，而不是修正它们。为弥合该差距，我们用 **Error-Recycling Fine-Tuning（ERFT）** 打破无误差训练假设，以稳定长生成中的 DiT：把 DiT 自生成误差回收为监督 prompt，促使模型从自回归错误反馈中修正自身错误。数学上，错误回收目标如下。

> 给定随机 timestep $t$ 上的含误差输入与干净输入，**ERFT** 旨在分别预测 **error-recycled velocity**：$V_t^{\mathrm{rcy}}=u(\tilde{X}_t,\tilde{X}_{\mathrm{img}},C,t;\theta)$ 与 $V_t^{\mathrm{rcy}}=u(X_t,X_{\mathrm{img}},C,t;\theta)$，以稳定自回归生成中的 DiT。该 velocity **始终指向干净 latent** $X_{\mathrm{vid}}$，无论当前状态 $\tilde{X}_t$ 以及 $t$ 之前的历史轨迹是否正确。

我们通过闭环微调实现：注入 DiT 产生的误差 $E$ 以模拟退化（第 4.1 节），计算并保存误差（第 4.2 节），动态存入 bank 并为新输入重采样（第 4.3 节），最终优化 error-recycled velocity $V_t^{\mathrm{rcy}}$（第 4.4 节）。考虑到对偶性，我们双向使用 noise error $E_{\mathrm{noi}}$ 与 latent error $E_{\mathrm{vid}}$，以保持理论完整。

---

## 4. Stable Video Infinity 方法

### 4.1 Error-Recycling Fine-Tuning

给定干净视频片段 $\{I_i\}_{i=1}^{T_{\mathrm{vid}}}$ 与 reference image $I_i$，用 3D VAE 提取 video latent $X_{\mathrm{vid}}$ 与 image latent $X_{\mathrm{img}}$（通常带 padding），二者均 $\in\mathbb{R}^{C\times T\times H\times W}$。然后随机采样噪声 $X_{\mathrm{noi}}\in\mathbb{R}^{C\times T\times H\times W}\sim\mathcal{N}(0,I)$ 以及训练 timestep $t\in\mathcal{T}_{\mathrm{tra}}$。

**误差注入。** 与假设干净输入的现有工作不同，我们旨在模拟推理中出现的误差累积退化。给定干净输入 $X_{\mathrm{vid}},X_{\mathrm{noi}},X_{\mathrm{img}}$，我们相应地设计三类误差：$E_{\mathrm{vid}}$、$E_{\mathrm{noi}}$、$E_{\mathrm{img}}$。这些误差从 memory bank $\mathcal{B}_{\mathrm{vid}}$、$\mathcal{B}_{\mathrm{noi}}$ 中重采样（见后续小节），再按概率注入干净输入：

$$
\tilde{X}_{\mathrm{vid}}=X_{\mathrm{vid}}+\mathbb{I}_{\mathrm{vid}}\cdot E_{\mathrm{vid}},\quad
\tilde{X}_{\mathrm{noi}}=X_{\mathrm{noi}}+\mathbb{I}_{\mathrm{noi}}\cdot E_{\mathrm{noi}},\quad
\tilde{X}_{\mathrm{img}}=X_{\mathrm{img}}+\mathbb{I}_{\mathrm{img}}\cdot E_{\mathrm{img}}.
\tag{3}
$$

其中 $\mathbb{I}_{\ast}=1$（以概率 $p_{\ast}$）否则为 $0$，控制误差注入概率。该设计用于模拟推理中任意 timestep 上误差累积的随机性与复杂性。为保留在已纠正误差下的生成能力，我们以概率 $p=0.5$ 使用无误差输入。送入 DiT 的最终输入记为 $\tilde{X}_t=\mathrm{Concat}(\tilde{X}_t,\tilde{X}_{\mathrm{img}})$，其中 $\tilde{X}_t=t\tilde{X}_{\mathrm{vid}}+(1-t)\tilde{X}_{\mathrm{noi}}$ 为带误差的 noisy video latent。误差注入从根本上打破式 (1) 中的无误差假设，用于弥合训练–测试差距。

![图 3. Stable Video Infinity。我们在闭环中 (a) 向干净 latent 注入误差以打破无误差假设，(b) 用单步积分近似预测并计算双向误差，(c) 从 memory 动态存取并重采样误差到干净输入。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/overall.png)

**控制注入与 velocity 预测。** 为满足开放域应用，我们将 SVI 扩展为额外控制 $C=\{C_{\mathrm{vis}},C_{\mathrm{emb}}\}$，见图 7c。**(a) $C_{\mathrm{vis}}$ 为 visual condition**，保证视频的空间级控制，例如 skeleton，通过逐元素相加注入 tokenized input。$C_{\mathrm{vis}}$ 可精确控制空间构图，服务于舞蹈动画等任务。**(b) $C_{\mathrm{emb}}$ 为 embedding condition**，用于无空间约束的多模态控制，例如 talking 动画中的 text 与 audio。$C_{\mathrm{emb}}$ 通过 DiT block 中的特定 cross-attention 注入。因此，带误差输入 $\tilde{X}_t$ 被 tokenize（可选叠加 $C_{\mathrm{vis}}$），再与可选 $C_{\mathrm{emb}}$ 一起送入 DiT，预测 velocity：

$$
\hat{V}_t=u(\tilde{X}_t,\tilde{X}_{\mathrm{img}},C,t;\theta).
$$

### 4.2 Bidirectional Error Curation

给定 velocity $\hat{V}_t$，我们用双向单步积分近似含误差预测，以高效整理误差，避免完整求解 ODE 的高昂成本。

**预测近似。** 为处理累积误差的复杂性，我们按图 4 中不同误差注入情形计算误差。对齐第 3.3 节的主目标，我们定义真值 **error-recycled velocity** $V_t^{\mathrm{rcy}}$（绿色单箭头）**指向无误差 latent** $X_{\mathrm{vid}}$，与历史轨迹和当前状态无关。然后，用带误差 noisy latent $\tilde{X}_t$ 与预测 velocity $\hat{V}_t$（红色单箭头），分别通过前向与后向单步积分近似 video latent $\hat{X}_{\mathrm{vid}}$ 与条件噪声 $\hat{X}_{\mathrm{noi}}^{\mathrm{img}}$（红色虚线）：

$$
X_{\mathrm{vid}}=\tilde{X}_t+\int_t^1 V_s\,ds,\qquad
X_{\mathrm{noi}}^{\mathrm{img}}=\tilde{X}_t-\int_0^t V_s\,ds.
$$

类似地，对 $\hat{V}_t$ 做积分得到 error-recycled latent 与 noise：

$$
X_{\mathrm{vid}}^{\mathrm{rcy}}=\tilde{X}_t+\int_t^1 V_s^{\mathrm{rcy}}\,ds,\qquad
X_{\mathrm{noi}}^{\mathrm{rcy}}=\tilde{X}_t-\int_0^t V_s^{\mathrm{rcy}}\,ds.
$$

**误差计算。** 用近似预测与 error-recycled 真值，考察图 4 中各注入情形。首先给出适用于所有情形的统一形式：

$$
E_{\mathrm{vid}}=\hat{X}_{\mathrm{vid}}-X_{\mathrm{vid}}^{\mathrm{rcy}},\quad
E_{\mathrm{noi}}=\hat{X}_{\mathrm{noi}}^{\mathrm{img}}-X_{\mathrm{noi}}^{\mathrm{rcy}},\quad
E_{\mathrm{img}}=\mathrm{Unif}_T(E_{\mathrm{vid}}),
\tag{4}
$$

其中 $\mathrm{Unif}(\cdot)$ 为在 latent 空间时间轴 $T$ 上的均匀采样。下面通过各真实情形说明该统一整理与第 3.3 节一致。

**(a) 无注入误差。** 可模拟初始 **single-clip predictive error**，即预测 velocity $\hat{V}_t$ 随时可能偏移。此时用 residual 定义 latent 与 noise 误差：$E_{\mathrm{vid}}=\hat{X}_{\mathrm{vid}}-X_{\mathrm{vid}}$，$E_{\mathrm{noi}}=\hat{X}_{\mathrm{noi}}^{\mathrm{img}}-X_{\mathrm{noi}}^{\mathrm{img}}$，且有 $X_{\mathrm{vid}}^{\mathrm{rcy}}=X_{\mathrm{vid}}$、$X_{\mathrm{noi}}^{\mathrm{rcy}}=X_{\mathrm{noi}}^{\mathrm{img}}$。

**(b) 注入起点误差。** 可模拟 **cross-clip conditional error**，即误差使起点从 $X_{\mathrm{noi}}^{\mathrm{img}}$ 偏移到 $\tilde{X}_{\mathrm{noi}}^{\mathrm{img}}$。此时可干预干净的 $X_{\mathrm{img}}$ 或 $X_{\mathrm{noi}}$ 以模拟误差影响。借助 error-recycled velocity，双向误差为 $E_{\mathrm{vid}}=\hat{X}_{\mathrm{vid}}-X_{\mathrm{vid}}$、$E_{\mathrm{noi}}=\hat{X}_{\mathrm{noi}}^{\mathrm{img}}-\tilde{X}_{\mathrm{noi}}^{\mathrm{img}}$，且有 $X_{\mathrm{vid}}^{\mathrm{rcy}}=X_{\mathrm{vid}}$、$X_{\mathrm{noi}}^{\mathrm{rcy}}=\tilde{X}_{\mathrm{noi}}^{\mathrm{img}}$。

![图 4. 误差计算。不同情形下，latent error $E_{\mathrm{vid}}$ 与 noise error $E_{\mathrm{noi}}$ 分别由前向（上）与后向（下）单步积分计算。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/error_cal.png)

**(c) 注入终点误差。** 可从累积视角同时模拟两类误差：先前积分指向退化生成 $\tilde{X}_{\mathrm{vid}}$。稳定后的 DiT 被鼓励核验、并在必要时把错误历史轨迹与错误中间状态 $\tilde{X}_t$ 纠正回干净方向（远离退化 latent $\tilde{X}_{\mathrm{vid}}$）。对齐 error-recycled velocity，误差可写为 $E_{\mathrm{vid}}=\hat{X}_{\mathrm{vid}}-X_{\mathrm{vid}}$、$E_{\mathrm{noi}}=\hat{X}_{\mathrm{noi}}^{\mathrm{img}}-X_{\mathrm{noi}}^{\mathrm{rcy}}$，此时仅有 $X_{\mathrm{vid}}^{\mathrm{rcy}}=X_{\mathrm{vid}}$。

### 4.3 Error Replay Memory

**误差更新。** 我们将计算出的 $E_{\mathrm{vid}}$ 与 $E_{\mathrm{noi}}$ 按 timestep 动态存入两个 replay memory $\mathcal{B}_{\mathrm{vid}}$ 与 $\mathcal{B}_{\mathrm{noi}}$。为更好缩小训练–测试差距，先将训练 timestep $\mathcal{T}_{\mathrm{tra}}=\{t_i\}_{i=1}^{N_{\mathrm{tra}}}$（通常 $N_{\mathrm{tra}}=1000$）与测试阶段使用的 timestep $\mathcal{T}_{\mathrm{test}}=\{t_n\}_{n=1}^{N_{\mathrm{test}}}$（通常 $N_{\mathrm{test}}=50$，见式 (2)）对齐并离散化。具体地，给定训练 timestep $t\in\mathcal{T}_{\mathrm{tra}}$，检索 $\mathcal{T}_{\mathrm{test}}$ 中最近的网格 $t_n$，把每个误差 $E_{\ast}$ 存入 bank $\mathcal{B}_{\ast}=\{B_{\ast,n}\}_{n=1}^{N_{\mathrm{test}}}$ 的对应位置 $B_{\ast,n}$，其中 $\ast=\{\mathrm{vid},\mathrm{noi}\}$。此处 timestep 充当指向特定存储位置的指针。

考虑到每 GPU 样本有限导致 bank 更新缓慢，我们设计 warmup：受 Federated Learning（McMahan et al., 2017）启发，用跨机器 gather 保存误差。为节省显存，设定每个位置保存误差数的上界 $|B_{\ast,n}|=Z=500$（附录定量实验）。当特定 bank $B_{\ast,n}$ 已满时，用 $L_2$ 距离衡量新误差 $E_{\ast}$ 与历史误差，替换最相似者，以保持误差多样性。

**误差采样。** 考虑到各误差项的具体角色，我们依据 flow-matching 轨迹中各输入项的性质设计选择性采样。对齐误差存入方式，先将从 $\mathcal{T}_{\mathrm{tra}}$ 采样的训练 timestep $t$ 离散到最近的测试 timestep $t_n\in\mathcal{T}_{\mathrm{test}}$。然后，对式 (3) 中的 $X_{\mathrm{vid}}$、$X_{\mathrm{noi}}$、$X_{\mathrm{img}}$，重采样误差为：

$$
E_{\mathrm{vid}}=\mathrm{Unif}(\mathcal{B}_{\mathrm{vid},n}),\quad
E_{\mathrm{noi}}=\mathrm{Unif}(\mathcal{B}_{\mathrm{noi},n}),\quad
E_{\mathrm{img}}=\mathrm{Unif}_T(\mathcal{B}_{\mathrm{vid}}).
\tag{5}
$$

其中 $\mathrm{Unif}(\mathcal{B}_{\ast,n})$ 在 timestep $t_n$ 对应的 memory bank $B_{\ast,n}\in\mathcal{B}_{\ast}$ 上均匀采样；$\mathrm{Unif}_T$ 在两个维度上进行：noise scheduler 的全部 timestep 以及视频的时间轴。各策略理由如下。

**(a) Video latent error** $E_{\mathrm{vid}}$ 从 timestep 对齐的 bank $\mathcal{B}_{\mathrm{vid},n}$ 均匀采样，因为逐步误差主要依赖轨迹中当前 timestep $t_n$。我们也经验发现退化类型与采样步高度相关。

**(b) Noise error** $E_{\mathrm{noi}}$ 遵循 video latent，从同一 timestep 的 $B_{\mathrm{noi},n}$ 均匀采样，考虑到噪声（起点）与 latent（终点）的对偶性。

**(c) Image latent error** $E_{\mathrm{img}}$ 从 video bank 采样，以对齐跨片段自回归，即生成帧作为下一段的 reference image。与逐步误差不同，reference image 由全部 timestep 积分得到，会累积整条轨迹上的误差。为模拟这种复杂性，我们跨 timestep 采样 $E_{\mathrm{img}}$，独立于当前 $t_n$，因为误差可能在任意步出现并累积。

### 4.4 优化

为训练 Stable Video Infinity，我们希望从式 (3) 得到的**含误差输入** $\tilde{X}_{\mathrm{vid}},\tilde{X}_{\mathrm{noi}},\tilde{X}_{\mathrm{img}}$ 预测指向干净 latent $X_{\mathrm{vid}}$ 的 **error-recycled velocity** $V_t^{\mathrm{rcy}}=X_{\mathrm{vid}}-\tilde{X}_{\mathrm{noi}}$。这与第 3.3 节弥合训练–测试差距的目标一致：

$$
\mathcal{L}_{\mathrm{SVI}}=\mathbb{E}_{\tilde{X}_{\mathrm{vid}},\tilde{X}_{\mathrm{noi}},\tilde{X}_{\mathrm{img}},C,t}\big|u(\tilde{X}_t,\tilde{X}_{\mathrm{img}},C,t;\theta)-V_t^{\mathrm{rcy}}\big|^2,
\tag{6}
$$

其中 $\tilde{X}_t=t\tilde{X}_{\mathrm{vid}}+(1-t)\tilde{X}_{\mathrm{noi}}$ 为注入误差后的 noisy latent。为保留用户灵活性，我们只训练 LoRA。错误回收微调能主动纠正轨迹，释放 DiT 的 restoration 能力。总体而言，呼应文首 Henry Ford 的引语，可在长视频生成语境下改写为：*“累积误差不过是通过回收错误重新开始的机会，这一次更稳定——Stable Video Infinity。”*

---

## 5. 实验

**表 1. 多样化设定下的通用视频生成。** 加粗为最高，下划线为次高。指标细节见 Huang et al. (2024)（VBench++）。

| Models | Generated Scenes | Subject Consistency | Background Consistency | Aesthetic Quality | Imaging Quality | Dynamic Degree | Motion Smoothness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Consistent Video Generation（单一 text prompt，无场景转换）** ||||||||
| Wan 2.1 | Single | 87.03% | 92.45% | 56.40% | 65.70% | 12.68% | 98.51% |
| StreamingT2V | Single | 84.79% | 89.27% | 56.81% | 66.41% | **57.04%** | 99.00% |
| HistoryGuidance | Single | 83.77% | 90.90% | 40.42% | 55.48% | 4.93% | *99.38%* |
| FramePack | Single | *93.08%* | *94.72%* | *63.57%* | *66.72%* | 7.75% | **99.57%** |
| SVI-Shot (Ours) | Single | **98.13%** | **98.19%** | **63.84%** | **71.88%** | *17.61%* | 98.93% |
| **Ultra-Long Consistent Video Generation（单一 text prompt，无场景转换）** ||||||||
| Wan 2.1 | Single | *80.00%* | *87.27%* | *56.19%* | *65.37%* | 14.29% | 98.74% |
| StreamingT2V | Single | 66.32% | 77.62% | 40.49% | 55.18% | **85.71%** | 95.60% |
| HistoryGuidance | Single | 64.84% | 80.51% | 29.84% | 50.41% | 7.14% | *99.42%* |
| FramePack | Single | 79.37% | 86.64% | 55.66% | 57.61% | 0.00% | **99.63%** |
| SVI-Shot (Ours) | Single | **97.50%** | **97.89%** | **65.75%** | **71.54%** | *21.43%* | 98.81% |
| **Creative Video Generation（带场景转换的 text prompt stream）** ||||||||
| Wan 2.1 | Multiple | 81.44% | 89.81% | 51.33% | 53.09% | **61.97%** | 98.57% |
| SVI-Film (Ours) | Multiple | 84.27% | 90.68% | 57.02% | 62.44% | *57.04%* | 99.12% |
| StreamingT2V | Single | 81.01% | 88.47% | 52.20% | 58.05% | **61.97%** | 98.96% |
| HistoryGuidance | Single | 84.12% | 91.21% | 38.40% | 52.31% | 7.75% | *99.39%* |
| FramePack | Single | *85.62%* | *91.22%* | **59.41%** | *59.44%* | 9.15% | **99.49%** |
| SVI-Shot (Ours) | Single | **93.52%** | **95.86%** | *58.07%* | **62.81%** | 55.63% | 98.42% |
| **Ultra-Long Creative Video Generation（带场景转换的 text prompt stream）** ||||||||
| Wan 2.1 | Multiple | 67.85% | 83.45% | 46.68% | 43.36% | 57.14% | 98.56% |
| SVI-Film (Ours) | Multiple | 70.90% | *84.30%* | *55.09%* | *57.73%* | *64.29%* | 98.89% |
| StreamingT2V | Single | 68.65% | 82.00% | 44.69% | 55.20% | **78.57%** | 96.95% |
| HistoryGuidance | Single | 62.58% | 81.97% | 28.66% | 47.68% | 7.14% | *99.36%* |
| FramePack | Single | *70.95%* | 83.46% | *52.39%* | 53.72% | 0.00% | **99.48%** |
| SVI-Shot (Ours) | Single | **91.96%** | **95.04%** | **63.31%** | **65.25%** | *64.29%* | 97.97% |

表中斜体表示原文下划线（次高）。Creative 设定中，Wan 2.1 与 SVI-Film 生成 Multiple 场景；其余长视频方法仍为 Single 场景外推。

**表 2. 音频条件长 talking。**

| Models | Sync-C $\uparrow$ | Sync-D  | FVD  |
| --- | --- | --- | --- |
| Wan 2.1 | 0.21 | 12.86 | 934 |
| MultiTalk | 1.26 | 9.57 | 520 |
| SVI-Talk (Ours) | **6.12** | **8.74** | **390** |

**表 3. 骨架条件长 dancing。**

| Models | PSNR $\uparrow$ | SSIM $\uparrow$ | FVD  |
| --- | --- | --- | --- |
| Wan 2.1 | 12.12 | 0.33 | 4099 |
| UniAnimate-DiT | 18.97 | 0.69 | 337 |
| SVI-Dance (Ours) | **20.01** | **0.71** | **299** |

![图 5. 关于视频长度的稳定性比较。SVI 更稳定，没有明显下降。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/sen.png)

**表 4. 各误差项消融。**

| Method | Sub. Cons. | Back. Cons. | Aest. Qual. | Img. Qual. |
| --- | --- | --- | --- | --- |
| Wan 2.1 | 66.73% | 82.83% | 43.95% | 42.31% |
| SVI w/o $E_{\mathrm{img}}$ | 73.82% | 84.21% | 49.58% | 57.63% |
| SVI w/o $E_{\mathrm{noi}}$ | 94.22% | 94.87% | 59.80% | 69.90% |
| SVI w/o $E_{\mathrm{vid}}$ | 93.56% | 95.01% | 58.99% | **71.50%** |
| SVI full | **94.69%** | **95.39%** | **61.88%** | 71.22% |

![图 6. 误差修正比较。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/error.png)

**Benchmark 设定。** 我们建立 consistent、creative 与 conditional 三类 benchmark，用于图像与文本到视频生成（各有两个变体），以满足多样化工业需求。**(a) Consistent Video Generation** 旨在用不变 text prompt 在同一场景内生成 **50 秒** 与 **250 秒**（ultra-long）视频。**(b) Creative Video Generation** 面向 vlogger（例如 TikTok）需求，强调带合理场景转换的叙事。我们通过 MLLM 开发自动引擎（附录 A.1）生成 prompt stream，覆盖 **50 秒** 与 **250 秒**（ultra-long）。**(c) Multimodal Conditional Generation** 衡量与额外条件的兼容性。我们评估 **300 秒** audio-guided talking 与 **50 秒** skeleton-guided dancing。**指标。** 全局视频质量使用 VBench++（Huang et al., 2024）的 6 项核心指标。特定条件生成额外使用 Sync-C、Sync-D、FVD、PSNR 与 SSIM。

**实现。** 我们部署一组仅有少量修改的 SVI 变体，仅用 300–6k 条短视频训练（附录 C）。**(a) SVI-Shot** 对齐先前长视频生成，关注同质场景。训练时用随机图像 padding 作为 anchor 得到 image latent，推理时替换为 reference image。**(b) SVI-Film** 支持由基于故事线的 prompt stream 控制的端到端长拍摄。使用五个 motion frames，并把 image latent 的 padding 帧替换为零。**(c) SVI-Talk** 面向音频条件说话人视频，在 SVI-Shot 上注入来自 Kong et al. (2025) 的 audio-image cross-attention。**(d) SVI-Dance** 用于骨架引导的舞蹈视频，按 Wang et al. (2025) 编码 skeleton 并注入 input tokens。$E_{\mathrm{img}}$、$E_{\mathrm{vid}}$、$E_{\mathrm{noi}}$ 的注入概率分别为 0.9、0.9 与 0.01。

### 5.1 开放域长视频生成

我们与 SOTA 方法比较：StreamingT2V（Henschel et al., 2025）、HistoryGuidance（Song et al.）、FramePack（Zhang et al., 2025），并报告 Wan 2.1（Wang et al., 2025）。条件生成分别与 MultiTalk（Kong et al., 2025）和 UniAnimate-DiT（Wang et al., 2025）比较音频引导 talking 与骨架引导 dancing。

**Consistent Video Generation。** 表 1 上半部分中，SVI-Shot 在多数核心指标上最好。注意：该设定下异常大的 dynamic degree 往往表示不可控的运动退化。相较 FramePack，我们在一致性上分别提升 5.05% 与 3.37%，在 image quality 上提升 5.16%。多数方法扩展到 ultra-long 时大幅下降，例如 Wan 2.1 与 FramePack 的 subject consistency 分别下降 7.03% 与 13.71%。相比之下，SVI 仅下降可忽略的 0.63%，同时保持满意的动态程度。

**Creative Video Generation。** 表 1 下半部分比较由基于故事线的 prompt stream 引导、带频繁场景转换的长视频。注意现有长视频工作普遍失败：它们无法用 prompt stream 生成拍摄级场景转换（见图 8）。SVI 在一致性、质量与满意的 dynamic degree 上取得最佳，优势显著。扩展到 ultra-long 时该优势仍然保持，显示 SVI 的稳定性。

**Multimodal Conditional Generation。** 表 2 与表 3 验证第 4.1 节所述两类典型条件：$C_{\mathrm{emb}}$ 的 audio-guided talking 与 $C_{\mathrm{vis}}$ 的 skeleton-guided dancing。可见 SVI 能轻松适配特定领域并在长视频上增强 SOTA，验证其开放域生成的通用性。

### 5.2 进一步分析

**对视频长度的稳定性。** 图 5 比较最新长视频方法在一致性与质量上的鲁棒性。与所有现有工作呈下降趋势不同，SVI 能保持稳健且较高的一致性与质量。这一性质证明其可生成任意长度，首次从根本上修正误差并打破时间限制。

**消融实验。** 表 4 用 1 个 epoch 的概念验证微调分析各误差，揭示两条关键信息。**(a)** 去掉 reference image 上的误差 $E_{\mathrm{img}}$ 会使所有指标显著下降。这表明干预轨迹起点（图 4b）以模拟误差累积是主要因素，直接支撑其在解决训练–测试 hypothesis gap 中的核心角色。**(b)** 向 video latent $E_{\mathrm{vid}}$ 或 noise $E_{\mathrm{noi}}$ 注入误差提供辅助收益，相对主因素 $E_{\mathrm{img}}$ 为间接作用。更多消融见附录 B。

**误差可视化。** 图 6 可视化解码后的 $E_{\mathrm{vid}}$ 与 $E_{\mathrm{noi}}$，并比较预测 $\hat{E}_{\mathrm{vid}}$，有两点观察。**(a)** 视频生成器（Wan 2.1）对其自身误差敏感，导致预测退化。SVI 的 error-recycling fine-tuning 可解决该问题，获得对自误差的鲁棒性。**(b)** 注入误差能很好模拟 drifting（Wan 2.1 的 $X_{\mathrm{vid}}$），证明错误回收在弥合训练–测试差距中的关键作用。

![图 7. 与各领域最佳方法的定性比较（视频见附录 D）。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/show.png)

### 5.3 定性比较

图 7a 比较由基于故事线的 prompt stream 引导的 **Creative Video Generation**。现有工作无法完成场景转换，并出现严重质量退化。SVI 则实现平滑场景转换，保持高视觉保真与 text-prompt 遵循，为端到端拍摄铺路。图 7b 比较不变 prompt 下的 **Consistent Video Generation**：现有方法出现 color shift、motion drift 以及静态图像退化；SVI 生成时间连贯、一致性与动态均合理的视频。图 7c 比较 talking 与 dancing 视频上的 **Multimodal Conditional Generation**：即使未针对这些领域专门设计，SVI 也能轻松处理长生成中的 drifting，证明有效性与可迁移性。更多视频见附录 D，包括跨域适应性（Tom and Jerry）。

---

## 6. 结论

我们处理长视频生成的核心挑战——训练–测试 hypothesis gap，它导致两类累积误差（第 3 节）。为弥合该差距，我们提出 Stable Video Infinity，通过主动修正自生成误差打破时间限制：采用新颖的 Error-Recycling Fine-Tuning（第 4 节），从错误反馈中自回归学习。与无误差训练假设不同，SVI 有意把历史误差注入干净输入，并学习预测 error-recycled velocity：用双向单步积分计算误差，存入 replay memory，并为新输入选择性重采样。在三类 benchmark 上，SVI 在长视频、超长视频与条件视频生成上超过 SOTA 方法。

---

## 致谢

本工作作为 Swiss AI Initiative 的一部分，由瑞士国家超算中心（CSCS）在 Alps 上以项目 ID a144 支持。感谢 Valentin Gerard、Tomasz Stanczyk、Megh Shukla 与 Xiaoyuan Liu 的讨论。

---

## 附录 A. Benchmark 设定

### A.1 自动 Prompt Stream 引擎

本文用故事线驱动的 text prompt stream 研究创意生成。由于高质量数据缺乏且标注费力，这在社区中仍是开放问题。为解决该问题并生成充足测试数据，我们提出一套**全自动系统**，用于轻松的端到端短片生产，简化创意视频生成与评估。用户只需给出高层主体说明（例如 “dog” 与 “street”）。系统随后自动检索并下载相关图像，并用 Multimodal Large Language Model（MLLM）为每个视频片段生成对齐故事线的 text prompt stream。这些图像–prompt 对再送入 Stable Video Infinity，生成无限长度、叙事驱动的高质量短视频。

![图 8. 所提出端到端自动管线概览：能从用户给定关键词生成无限短片。该引擎用于按特定故事线生成我们创意视频 benchmark 的 prompt stream。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/film.png)

完整流程见图 8。该管线系统地把高层关键词转化为结构化的图像与 prompt 序列对，消除费力的人工标注与 prompt 工程。步骤如下。

1. **基于关键词的图像检索（Keyword $\rightarrow$ Image）**：从用户定义关键词（例如 “dog” 与 “street”）开始，送入自动下载脚本，从在线资源检索多样化相关图像。用户也可跳过此步，直接使用自定义图像而非自动检索。

2. **自动 Prompt Stream 生成（Image $\rightarrow$ Storyline）**：每张检索图像由 Qwen2.5（Yang et al., 2024）自动标注。关键在于：模块不是生成单一静态描述，而是产出时间连贯的 $L$ 条不同 prompt 序列。我们称之为 “prompt stream”，用于描述由输入图像静态场景出发的合理动态演化或叙事。例如，给定休息中的狗，prompt stream 可能描述狗醒来、竖起耳朵、然后摇尾巴。流长度 $L$ 可控。**在我们的 benchmark 中，创意生成设 $L=10$（一条长镜头中 10 个顺序视频片段），ultra-long 设定设 $L=50$。** 用户也可跳过此步，直接提供自己的 prompt stream 与故事线。

3. **输入准备与视频合成（Storyline $\rightarrow$ Short Film）**：与 prompt 生成并行，最初下载的图像经过标准归一化后送入 SVI。归一化图像与对应生成的 prompt stream $\{\mathrm{prompt}\,1,\mathrm{prompt}\,2,\ldots,\mathrm{prompt}\,L\}$ 构成完整输入。**SVI 将为 prompt stream 中每条 prompt 迭代生成一个视频片段，以上一段生成的末帧为条件。** 模型随后根据 prompt stream 的顺序指令，对输入图像做动画合成。

该自动管线是本研究的核心：它能快速生成大量多样化测试用例用于定性评估，并为基准测试模型“解释静态内容并按动态文本引导做动画”的能力提供可扩展框架。

### A.2 Benchmark 数据集

借助自动 prompt-stream 引擎，我们能高效构建覆盖创意与一致视频生成的高质量测试数据。为更好镜像真实使用，约三分之二数据经该引擎从网络自动采集，其余三分之一来自真实用户（遵循 Zhang et al., 2025），以兼顾规模、多样性与真实性。作者将开源完整代码与全部 benchmark 数据集，以推动长视频生成与评估。

对通用视频生成，我们在默认 50 秒设定下组装 152 个样本，在 250 秒 ultra-long 设定下组装 14 个样本，并在相同条件下评估所有方法以保证公平。在 consistent 赛道中，**每张输入图像在两种时长下都配对单一、稳定的 text prompt**，强调时间连贯与身份保持。相对地，creative 赛道引入叙事动态：我们生成 **默认设定 50 条、ultra-long 设定 100 条文本描述的 prompt stream**，以支持丰富的、故事线驱动的场景转换与事件推进。该设定中，每个生成视频片段（5 秒，16 FPS）遵循一条独特 prompt。最后，对多模态条件生成，我们在所有方法上评估 10 个 ultra-long 样本，考察模型在延长时长上把视觉输入与演化的多模态引导（例如 audio 与 skeleton）对齐并融合的能力。

这些设定共同构成**全面、可扩展且严格的 benchmark 套件**，同时施压保真度与创意性，支持受控消融与跨方法比较，并反映开放域长视频创作的实际需求。这也突破先前长视频评测（Zhang et al., 2025；Henschel et al., 2025；Song et al.）只关注单一同质场景与重复运动的局限；该问题已被所提出的 SVI-Shot 充分解决，如表 1 所示。

### A.3 对工业的更广实践影响

**拍摄与娱乐。** 当代短视频制作通常需要大量人工设计场景转换、撰写故事线并拼接片段，使真正的单镜头叙事视频不现实。不同地，**SVI 使端到端、单镜头拍片变得可行且可及**。用户只需给出高层意图与简短文本描述；系统随后自主产出无限长度的单镜头视频，节奏可控、转场视觉合理，无需人工干预。

**机器人世界模型。** 现有面向机器人（Li et al., 2025）与仿真（例如 Cosmos，Agarwal et al., 2025）的 world model（Wang et al., 2023）受限于**短视频视野与有限训练多样性，难以模拟长时间、复杂场景或稀有分布外边角情况**。**SVI 在长时长、可控且语义一致的场景合成上展示出强潜力，尤其在导航领域。**

**世界生成、游戏与 Spatial AI。** 我们观察到所提出方法在跨场景上具有强的**长期几何与身份一致性**，可催化大规模世界生成与 spatial AI。管线在保持结构连贯的同时提供可控、叙事条件化的场景演化，有利于 3D-aware 视频合成、持续场景建模，以及在持久环境上推理的交互智能体研究（Che et al., 2024；Yu et al., 2025）。

### A.4 对学术界的更广方法影响

**我们的中心目标是弥合自回归生成中的训练–测试 hypothesis gap。该差距在包括 LLM 与 MLLM 在内的现代生成范式中普遍存在。** 训练时模型通常见到干净、无误差输入；测试时自回归生成却把每个新 token（或帧）以前面可能含误差的输出为条件。因此该 hypothesis gap 也是 LLM/MLLM 训练中的挑战。我们的错误回收微调让模型接触自身不完美的 rollout，并学会从错误中恢复，从而使优化与测试时过程对齐。

**该原则可推广到视频之外的广泛生成设定，包括 LLM 与自回归图像生成模型。** 在长视频生成中，使内容失稳的漂移与复合伪影可视为一种 **visual hallucination**。这与 LLM 与 MLLM 语境中的开放问题 **linguistic hallucination** 性质相似，即当生成上下文过长时 LLM 倾向于给出幻觉词。基于这一洞察，错误回收微调有潜力通过闭合训练–测试差距减轻该效应：模型学习在分布偏移下稳定、重新锚定并修复内容的鲁棒策略。

### A.5 同期相关工作

近期有许多基于 Self-Forcing（Huang et al., 2025）应对长视频挑战的同期工作。LongLive（Yang et al., 2025）采用因果、帧级 AR 设计，并整合 KV-recache 机制，用新 prompt 刷新缓存状态以实现平滑场景切换。Self-Forcing++（Cui et al., 2025）利用教师模型的丰富知识，用从自生成长视频中采样的片段指导学生模型。LoViC（Jiang et al., 2025）使用富有表达力的自编码器，把视频与文本联合压缩到统一潜表示，并采用受 Q-Former 启发的单 query-token 设计。

相较这些工作，SVI 有若干独特优势：(1) SVI 支持图像到视频生成，使 talking-head 与舞蹈合成等更广的长视频应用成为可能；(2) SVI 对场景转换提供灵活控制，以适应不同应用场景；(3) SVI 对生成视频没有固有长度限制；(4) SVI 可轻易适配任意视频生成器。

### A.6 局限性与未来工作

**规模扩大。** 受时间限制，模型在小数据集上训练，未经充分扩大。我们观察到：当测试时图像风格偏离训练分布时，相邻片段可能出现 color shift。可能原因是模型错误地把测试时低层风格当作误差并“纠正”。我们计划扩大数据并多样化风格以纠正这种“误解”，采用领域均衡采样，并引入风格保持损失或参考风格条件化以减少此类伪影。此外，课程式扩大、混合/高分辨率训练与更强增强应能进一步改善对风格偏移的鲁棒性。

**实时与交互生成。** 当前模型基于 Wan 2.1，以并行而非流式方式生成帧，给实时部署带来挑战。近期工作（例如 CausVid，Yin et al., 2025；Self-Forcing，Huang et al., 2025）已开始探索流式生成。由于本方法只训练轻量 LoRA adapter，可无缝接入实时管线。未来工作中，我们也计划追求实时、无限视野视频生成，并纳入交互控制信号（例如实时 prompt 更新、类摇杆轨迹引导与事件触发）以支持响应式编辑与操控。

**身份一致性。** 在 SVI-Film 中，我们通过以五个 motion frames 为条件维持跨片段运动连续性。然而，没有显式长期记忆时，当主角离开画面，可能发生 identity drift 或 swapping。虽然 SVI-Shot/Talk/Dance 已通过 anchor 帧实现身份控制，它们目前尚未扩展到带场景转换的创意生成。我们打算开发端到端拍摄管线，结合持久身份嵌入、跨镜头特征缓存与场景感知 anchor，以加强复杂转场中的主体一致性，并将提出更先进的锚定与记忆策略。

### A.7 LLM 使用说明

本工作中，LLM 仅用于写作润色与语法检查，严格遵循 ICLR 指南。LLM 不参与构想、方法及其他敏感部分。定性比较中，我们用 LLM 辅助附带的 Python 脚本任务（例如抽帧、片段拼接及相关工具）以及 README 文档。

### A.8 伦理考量

本工作使用与评测的模型及对应训练数据均来自公开可得数据集。我们不使用专有、受限或敏感数据。对检索得到的测试数据，我们检查了宽松许可。本研究不涉及人类受试者、生物识别信息或生物数据，因此不引发相关人体研究风险。关于说话人视频，我们认识到 deepfake 与欺诈等潜在滥用风险。为减轻风险，未来开源发布将包含合规约束与护栏，以劝阻恶意使用并支持伦理部署。

---

## 附录 B. 定量实验

**表 5. 探索朴素视频延长方法。** 异常高的最佳值用强调标出。

| Models | Generated Scenes | Subject Consistency | Background Consistency | Aesthetic Quality | Imaging Quality | Dynamic Degree | Motion Smoothness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Copy Clips | Single | 98.48% | 98.60% | 67.99% | 71.93% | 7.14% | 98.93% |
| Ping-Pong Clips | Single | 98.51% | 98.59% | 67.92% | 71.92% | 7.14% | 99.06% |
| Copy Reference Img | Single | 100.00% | 100.00% | 68.55% | 73.05% | 0.00% | 99.84% |
| Wan 2.1 | Single | 80.00% | 87.27% | 56.19% | 65.37% | 14.29% | 98.74% |
| StreamingT2V | Single | 66.32% | 77.62% | 40.49% | 55.18% | 85.71% | 95.60% |
| HistoryGuidance | Single | 64.84% | 80.51% | 29.84% | 50.41% | 7.14% | 99.42% |
| FramePack | Single | 79.37% | 86.64% | 55.66% | 57.61% | 0.00% | 99.63% |
| SVI-Shot (Ours) | Single | 97.50% | 97.89% | 65.75% | 71.54% | 21.43% | 98.81% |

**表 6. 自生成误差与手工图像增强误差的比较。**

| Error | Sub. Cons. | Back. Cons. | Aest. Qual. | Img. Qual. |
| --- | --- | --- | --- | --- |
| Self Only | **69.34%** | 83.14% | **52.83%** | **56.97%** |
| Handcraft Only | 69.21% | **83.65%** | 49.62% | 45.17% |
| Self+Handcraft | 69.24% | 83.48% | 48.51% | 39.50% |

**表 7. 通过修改 LoRA alpha 分析错误回收强度。**

| $\alpha$ | Sub. Cons. | Back. Cons. | Aest. Qual. | Img. Qual. |
| --- | --- | --- | --- | --- |
| 0.2 | 92.16% | 93.68% | 58.52% | 77.49% |
| 0.4 | 97.49% | 96.73% | 59.86% | 77.32% |
| 0.8 | **99.59%** | **99.37%** | 59.69% | 78.11% |
| 1.0 | 98.54% | 97.97% | **59.88%** | **78.12%** |

**探索欺骗指标的朴素方法。** 表 5 进一步研究三类欺骗指标的设计：**(a) Copy Clips**，朴素地把首段生成视频复制 50 次；**(b) Ping-Pong Clips**，以乒乓方式把首段复制 50 次；**(c) Copy Reference Image**，仅朴素地把 reference image 复制 $50\times 81$ 次。可见这些朴素方法能成功欺骗并攻击部分指标，例如 consistency 与 quality，显示现有视频生成评测的局限。相应地，这些指标上异常高的方法往往在对应项上异常低，例如 0.00% 的 dynamic degree。**因此，一个有价值的信息是：必须综合所有指标一起评测，以防止指标欺骗。** 我们的 SVI 在全部指标间给出满意折中，显示其有效性。

**自生成误差与朴素图像增强的比较。** 表 6 对 reference image 施加手工退化（随机 color shift、blur、sharpness），并与我们的自生成误差比较。我们观察到：朴素图像增强不仅无助，还会大幅损害图像质量。而且，把自生成误差与手工误差组合会造成严重冲突，导致进一步下降。这些结果表明：累积的、模型诱导的误差具有难以用手工增强模仿的独特特性，说明从模型自身误差中学习是必要的。

**错误回收强度分析。** 表 7 通过在测试时改变 LoRA 权重 $\alpha$ 逐步调节错误回收强度，较低值意味着较弱效果。相较 $\alpha=1$，从 0.8 降到 0.2 时所有指标一致下降，表明削弱纠错能力时会出现更灾难性的误差。因此，这证明主动纠错的必要角色。

**表 8. 误差 bank 大小 $Z$ 的消融。**

| $Z$ | Sub. Cons. | Back. Cons. | Aest. Qual. | Img. Qual. |
| --- | --- | --- | --- | --- |
| 1 | 67.82% | 82.90% | 51.55% | 52.96% |
| 10 | 67.41% | 81.79% | 52.12% | 54.29% |
| 100 | 68.51% | 82.97% | 51.03% | 55.30% |
| 500 | **69.34%** | 83.14% | **52.83%** | **56.97%** |
| 1000 | 69.14% | **83.83%** | 51.36% | 55.06% |
| 2000 | 69.02% | 82.98% | 51.39% | 54.55% |

**误差 bank 大小分析。** 表 8 评估改变误差 bank 大小 $Z$ 对多项指标的影响。过小的 bank（例如 $Z=1$ 或 $Z=10$）限制误差多样性，导致 Subject Consistency、Background Consistency、Aesthetic Quality 与 Image Quality 次优。随着 $Z$ 增大，各指标一致改善。但超过 $Z=500$ 后性能饱和，多数指标无显著增益或略降。我们选择 $Z=500$ 取得满意性能，有效平衡误差多样性与模型能力。

**表 9. 训练与测试使用的详细超参数。**

| Parameter | Value | Description |
| --- | --- | --- |
| Learning rate | 2.0e-05 | Adam 优化器学习率 |
| Max epochs | 10 | 最大训练 epoch |
| Gradient clipping | 1.00 | 梯度范数裁剪阈值 |
| Gradient accumulation | 1 | 梯度累积步数 |
| Training strategy | deepspeed_stage_2 | 分布式训练 |
| Data workers | 1 | 数据加载 worker 数 |
| Gradient checkpointing | Yes | 显存优化 |
| Checkpointing offload | Yes | CPU gradient checkpointing |
| LoRA rank | 128 | LoRA rank 维 |
| LoRA alpha | 128 | LoRA 缩放参数 |
| LoRA init | kaiming | LoRA 权重初始化 |
| Architecture | lora | 训练架构类型 |
| LoRA position | q,k,v,o,ffn.0,ffn.2 | LoRA 目标模块 |
| Frame height | 480 | 视频帧高（像素） |
| Frame width | 832 | 视频帧宽（像素） |
| Tiled processing | Yes | 分块推理以节省显存 |
| Tile height | 34 | 处理 tile 高 |
| Tile width | 34 | 处理 tile 宽 |
| Video frames | 81 | 每个样本的视频帧数 |
| error-recycling tuning | Yes | 启用错误回收微调 |
| Warmup iterations | 20 | 跨节点收集误差的迭代数 |
| Noise error $p_{\mathrm{noi}}$ | 0.01 | 噪声误差注入概率 |
| Latent error $p_{\mathrm{vid}}$ | 0.9 | Latent 误差注入概率 |
| Image error $p_{\mathrm{img}}$ | 0.9 | 图像误差注入概率 |
| Clean input $p$ | 0.5 | 完全不注入误差的概率 |
| Timestep grids | 50 | 误差缓冲的离散 timestep 网格 |
| Maximum error $Z$ | 500 | 每个 memory 网格最多保存的误差数 |
| Motion frames | 5 | 运动参考帧数 |
| Motion probability | 0.95 | 使用 motion frame 的概率 |

---

## 附录 C. 实现细节

实验在大规模 GH200 集群上进行。SVI 训练超参数见表 9。我们基于 Wan2.1-I2V-14B-480P 实现 SVI，仅微调 LoRA 以保留灵活性，即用户可轻松把 SVI 注入私有模型。**全部模型 / 源码 / benchmark 数据集均已公开。**

**训练数据。** 所提出方法数据效率显著，仅用小规模公开数据微调。所有设定下 SVI 只训练 10 个 epoch。对创意与一致视频生成，SVI-Shot 与 SVI-Film 用含 6K 视频的 MixKit Dataset（Lin et al., 2024）训练。我们也用 UltraVideo（Xue et al., 2025）探索扩大能力。对音频引导 talking，使用 Hallo 3（Cui et al., 2025）的随机子集，含 5,000 个视频片段。对骨架条件舞蹈，用 TikTok（Jafarian et al., 2021）做错误回收微调，其中 LoRA 从 Wang et al. (2025) 预训练。

---

## 附录 D. 额外定性比较

所提出 SVI 能在文本流引导下生成时间连贯的短片，见图 9 至图 13，展示其端到端叙事与创意内容创作潜力。除基础生成外，方法还支持多模态控制。如图 14 与图 15，SVI 通过 visual 与 embedding 两类控制实现稳健的长程视频合成，可精确操控角色运动与面部表情。

![图 9. 飞机降落故事的定性结果。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/Vis_Creative_Short_V2/airplane_creative_short.png)

![图 10. 猫故事的定性结果。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/Vis_Creative_Short_V2/cat_creative_short.png)

![图 11. 摩托车故事的定性结果。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/Vis_Creative_Short_V2/motorcycle_creative_short.png)

![图 12. 动物园故事的定性结果。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/Vis_Creative_Short_V2/zoo_creative_short.png)

![图 13. 婴儿故事的定性结果。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/Vis_Creative_Short_V2/baby_creative_short.png)

![图 14. 舞蹈的定性结果。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/dance_034.png)

![图 15. 说话脸的定性结果。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/talkingface_vis.png)

![图 16. 《猫和老鼠》片段的定性结果。](../../../arxiv/streaming/stable_video_infinity/extracted/Figure/tom_vis.png)
