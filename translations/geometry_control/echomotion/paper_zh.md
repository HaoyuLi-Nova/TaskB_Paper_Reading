# EchoMotion：经由 Dual-Modality Diffusion Transformer 的统一人体视频与运动生成

**作者：** Yuxiao Yang$^{1,2}$、Hualian Sheng$^{2}$、Sijia Cai$^{2,*}$、Jing Lin$^{3}$、Jiahao Wang$^{4}$、Bing Deng$^{2}$、Junzhe Lu$^{1}$、Haoqian Wang$^{1,\dagger}$、Jieping Ye$^{2,\dagger}$

**机构：** $^{1}$清华大学；$^{2}$阿里巴巴集团；$^{3}$南洋理工大学；$^{4}$西安交通大学

**贡献说明：** $^{*}$项目负责人（Project leader）；$^{\dagger}$通讯作者

**出处：** ICLR 2026（arXiv:2512.18814）

**arXiv：** [2512.18814](https://arxiv.org/abs/2512.18814)

**项目页：** https://yuxiaoyang23.github.io/EchoMotion-webpage/

**代码：** https://github.com/D2I-ai/EchoMotion

**原文 TeX：** [`arxiv/geometry_control/echomotion/extracted/main.tex`](../../../arxiv/geometry_control/echomotion/extracted/main.tex)

---

## 摘要

视频生成模型已取得显著进展，但仍难以合成复杂人体运动，根源在于人体关节具有高自由度。这一局限来自仅以像素为目标的训练目标所固有的约束：模型被偏向外观保真，而牺牲对底层运动学原理的学习。为此，我们提出 **EchoMotion**，该框架旨在对外观与人体运动的联合分布建模，从而提升复杂人体动作视频的生成质量。EchoMotion 将 DiT（Diffusion Transformer）扩展为双分支架构，联合处理由不同模态拼接而成的 token。进一步地，我们提出 **MVS-RoPE**（Motion-Video Synchronized RoPE），为视频 token 与运动 token 提供统一的 3D 位置编码。通过为双模态潜序列提供同步坐标系，MVS-RoPE 建立有利于两模态时间对齐的归纳偏置。我们还提出 **Motion-Video Two-Stage Training Strategy**。该策略使模型既能联合生成复杂人体动作视频及其对应运动序列，也能完成多样的跨模态条件生成任务。为训练具备上述能力的模型，我们构建 *HuMoVe*：约 80,000 条高质量、以人为中心的视频–运动对大规模数据集。结果表明，显式表示人体运动与外观互补，显著提升以人为中心视频生成的连贯性与合理性。项目页：https://yuxiaoyang23.github.io/EchoMotion-webpage/ 。

---

![图 1. EchoMotion 能力概览：（a）提升以人为中心视频合成中的解剖完整性；（b）实现视频与运动之间的双向控制。通过在统一的双分支 Diffusion Transformer 中处理视觉序列与运动序列，模型学习人体外观与运动学的联合分布。](../../../arxiv/geometry_control/echomotion/extracted/figures/teaser_.png)

---

## 1. 引言

近年来，视频生成模型取得显著进展，尤其受扩散模型（DDPM、score-based 扩散、DiT、Latent Diffusion Models）与 VLM caption 模型（LLaVA、Qwen2-VL、Gemini）快速演进的推动。现有视频生成模型（CogVideo、Wan、HunyuanVideo、Open-Sora）在视觉保真与时间一致性上已有可观结果。即便如此，即使是 state-of-the-art 生成器，在合成复杂人体运动时仍面临显著挑战。所得视频常出现严重解剖伪影与不自然的关节运动，如图 1(a) 所示。这一缺陷主要源于仅做像素回归的训练目标所固有的局限：它们往往优先视觉保真，而忽视支配人体关节运动的底层运动学原理。具体而言，扩散模型常用的像素级重建损失由静态外观与背景细节主导，而非时间上的动力学。对人体主体而言，这一缺陷尤为突出：人体自由度高，即使细微的运动学误差也会显得极不自然。

已有研究（VideoJAM、VBench）指出，即使是训练数据中充分出现的基本运动类型，人体视频合成问题依然存在。这一发现呼应如下洞见：解剖合理性的缺失并不仅仅是数据规模问题，而指向细粒度运动学动态建模上的固有困难。为应对该问题，部分工作把生成条件化在显式结构引导上，例如 2D keypoints 先验（VACE、Follow-Your-Pose）或 3D pose 先验（RealisDance 等）。该路径虽能提供直接控制，却带来两点重要局限。第一，它造成对控制信号的依赖，而真实应用中这些信号往往不可得。第二，即便使用人体 pose 这类 3D 先验，它们通常被投影到 2D 图像平面以与视频帧对齐。投影过程不可避免地丢弃关键 3D 几何信息，导致对底层身体结构的理解减弱，并可能引入运动不一致。

受上述发现启发，我们提出 **EchoMotion**：原生地对视频与人体运动模态的联合分布建模。它采用 dual-modality diffusion transformer，以双分支架构统一处理来自不同模态的 token。与先前 MMDiT 架构仅聚焦于对视频输入去噪不同，EchoMotion 进一步显式地对参数化运动做联合去噪。得益于这种显式运动建模，本方法能显著减少运动伪影，并促进复杂人体运动视频的生成。

与常见的以运动视频为条件的做法（RealisDance、Animate Anyone）相比，我们的参数化运动表示不仅更 token-efficient，还保留原生 3D 结构信息，更有利于模型学习运动学模式。具体地，视觉 token 与运动 token 沿序列维拼接，并由所提出的 Motion-Video Synchronized RoPE（MVS-RoPE）机制处理。该机制对两模态施加 token-wise 位置嵌入，强制精确的时间对应，同时避免视频流与运动流之间的空间位置碰撞。为保证模型高效收敛并充分挖掘多任务能力，我们设计 **Motion-Video Two-Stage Training Strategy**。该策略采用两阶段训练配方：先做 motion-only 训练，再做 motion-video multi-task training。第二阶段进一步构造三种范式——joint generation、motion-to-video generation 与 video-to-motion generation——以高效捕捉跨模态交互。此外，我们引入新的高质量、以人为中心的视频数据集，命名为 *HuMoVe*。该数据集约含 80,000 个视频片段，每段都具有清晰、可辨的人体运动。对每个视频，我们提供细粒度文本描述，分别详述：(1) 主体外观与服饰；(2) 背景语境；(3) 所执行动作的精确描述。为支持多模态联合训练，我们还为每个视频片段提取并包含对应的 SMPL 运动参数，提供丰富的结构引导。

实验表明，通过对视频与运动的联合分布做整体建模，EchoMotion 在时间连贯性与结构完整性上显著增强人体视频合成。全面评估验证其优于 state-of-the-art 基线，运动伪影大幅减少，物理合理性保持更好。此外，这一统一方法天然支持多样的跨模态可控生成，该能力经大量定性与定量评估得到确认。

---

## 2. 相关工作

### 2.1 视频扩散模型

扩散模型已成为视觉合成的事实标准，规模从高保真图像生成（Latent Diffusion Models、SD3 / MMDiT、DiT）扩展到视频（Stable Video Diffusion、Open-Sora、Wan）与 3D 资产（Wonder3D、Structured 3D、Wonder3D++、SyncDreamer、Hunyuan3D）。核心架构通常是在 VAE 潜空间中运行的 Diffusion Transformer（DiT），最初为静态图像设计。通过把 2D attention 扩展为 3D full attention（Wan、HunyuanVideo），或插入额外的 temporal attention 层（Stable Video Diffusion、Seedance），扩散模型被适配到时间连贯的视频生成。在条件化方面，MMDiT 不采用常规 cross-attention 注入文本提示，而是用另一路径：以分开的权重处理视觉 token 与文本 token，再将它们拼接后送入 attention。该设计促进两模态之间的双向信息流动。为进一步提升时间连贯性，VideoJAM 通过在外观之外预测光流，把显式运动先验引入视频模型。我们的工作不聚焦于光流这类稠密、低层、短时运动，而是把 SMPL 参数作为高层、结构化的人体运动学来利用。

### 2.2 条件人体视频生成

扩散模型的近期进展显著提升了人体视频生成质量。在预训练扩散模型之上，DisCo 与 Follow-Your-Pose 等方法把 ControlNet 架构适配为用 2D 人体 keypoints 引导生成。另一些工作如 MagicAnimate 与 Animate Anyone 采用专用 pose guidance 模块，编码 2D pose 序列并作为条件注入。另一支研究用渲染的 3D 人体模型做条件。例如 Champ 用渲染的 SMPL 模型作为视频生成的引导帧。沿此方向，RealisDance 沿通道维拼接多种视觉 pose 表示——例如 HaMeR 与 DWPose 的输出，以及渲染的 SMPL 模型——来引导生成。类似地，Human4DiT 以渲染的 SMPL 模型与相机姿态为条件，生成自由视角人体视频。这些方法虽强，却共享一个根本的架构局限：它们都是严格的条件生成器。相比之下，我们引入统一架构，把人体运动参数与视频当作耦合模态，从而同时支持联合生成与跨模态补全。

---

## 3. 方法

本文提出 EchoMotion：由输入文本提示生成视频及其对应运动序列的系统。我们首先提出 **Dual-Modality Diffusion Transformer**，辅以参数化运动表示与 **Motion-Video Synchronized RoPE**，以有效建模这两种不同模态之间的复杂交互（第 3.1 节）。然后，我们定制 **Multi-Modal Two-Stage Training Strategy**，促进该多模态系统内部的相互促进与补全（第 3.2 节）。最后，为支持本研究及更广泛社区，我们构建并发布 ***HuMoVe*** 数据集：大规模成对的视频、3D 人体运动参数与文本数据（第 3.3 节）。

### 3.1 面向视觉–运动联合生成的扩散

**架构。** 面向高保真视频生成，我们选择 Wan 作为骨干，因其性能更优。与只建模视频分布 $p(x|y)$——可能优先外观保真而牺牲运动原理——的做法不同，我们建模人体运动与视频的联合分布 $p(x,m|y)$。其中 $y$ 为给定文本提示，$x$ 与 $m$ 分别表示视频与人体运动参数。这种统一建模使模型能有效学习视觉动态与运动动态。如图 2 所示，Dual-Modality Diffusion Transformer 首先处理输入视频，由 SMPL 的 pose 与 shape 参数得到参数化人体运动表示。这些运动 token 与视觉 token 拼接，形成 **统一的多模态上下文序列**，再送入一系列 Dual-Modality Diffusion Transformer block。在每个 block 内，所提出的 MVS-RoPE 编码每个 token 在统一多模态上下文中的精确位置。这保证在联合 self-attention 过程中，模态内与跨模态信息都能被正确交换。

![图 2. EchoMotion 概览。（a）用于视频–运动联合建模的 dual-modality DiT block。（b）MVS-RoPE，作为双模态 token 序列的同步坐标系。](../../../arxiv/geometry_control/echomotion/extracted/figures/pipeline_v2.png)

**参数化人体运动表示。** 给定运动中的人物视频，我们用 SMPL 模型参数化人体姿态与体型。它通过低维 pose 与 shape 参数捕捉关节化人体构型，因而在涉及 human mesh recovery 的计算机视觉任务中被广泛采用。对单帧，SMPL 便于提取人体表示。具体地，它提供定义整体体型的 shape 参数 $\beta \in \mathbb{R}^{10}$、捕捉人体关节角的 pose 参数 $\theta \in \mathbb{R}^{24 \times 6}$、全局身体朝向 $\gamma \in \mathbb{R}^{6}$，以及人体根关节位置 $v \in \mathbb{R}^{3}$。遵循 DART，我们进一步用 3D 关节位置 $\eta \in \mathbb{R}^{24 \times 3}$ 表示各个人体关节。

为构造融合人体运动模态的统一表示，我们设计 multi-head projector，在参数空间与潜空间之间建立双向映射。对每一帧，我们把运动参数分为三组：$\{v,\eta\}$ 表示 3D 位置，$\{\theta,\gamma\}$ 表示 6D 旋转，$\beta$ 表示人体体型。三个独立 MLP 再把这些参数集合投影到目标 transformer 隐维，每帧生成 $51$ 个运动 token。类似地，另三个 MLP 把生成的运动 token 映回原始参数空间以供重建。关键的是，为更好建模快速变化的运动模式，我们保留这些运动 token 的时间结构。这与视觉 token 通常施加的时间维下采样形成对比；这些精炼且紧凑的运动 token 以仅增加极少计算开销的代价，保留了关键的时间运动信息。

**Dual-Modality Diffusion Transformer Block。** 该 block 处理并整合多模态信息。它首先把视频模态与人体运动模态的嵌入分别通过模态专用投影，实现为两套不同的可学习矩阵。投影后的特征再沿序列维拼接，形成：

$$
Q_{mm}, K_{mm}, V_{mm} = [Q_v; Q_m],\ [K_v; K_m],\ [V_v; V_m],
\tag{1}
$$

其中 $[\cdot;\cdot]$ 表示序列级拼接。随后施加联合 self-attention 层，以捕捉两模态内部及之间的依赖与相关。self-attention 之后，attended 特征被拆开，各模态特征再分别经独立的 cross-attention 层（与文本信息交互）与 FFN 处理。该架构使模态之间以及与文本引导之间都能进行细致交互。在 self-attention 层内，我们提出专用的多模态位置嵌入，为特征注入精确位置信息。这使模型能有效推理模态内与跨模态的 token 关系，表述如下。

**Motion-Video Synchronized RoPE。** 现有 MMDiT 架构通常通过文本编码器固有的位置 ID，或采用 M-RoPE（把相对位置编码扩展到统一序列中同时处理文本与视觉模态）来纳入位置信息。然而，这两种做法都未直接考虑人体运动与视频之间固有的时间对齐。因此，它们并不直接适用于表示参数化运动 token 的位置信息，需要专门设计以准确捕捉这一关键的运动位置信息。

如图 2(b) 所示，我们的 MVS-RoPE 旨在处理由视频潜 token 与运动潜 token 组成的多模态上下文序列所具有的独特时空性质。在空间上，我们划分坐标空间：视频 token 占据基座 $(h,w)$ 区域，而运动 token 被当作“对角扩展”，通过偏移其空间索引实现。这保证模态可区分。在时间上，考虑到运动 token 与视觉 token 之间的直接时间对应（视频 VAE 有 4 倍时间压缩），我们采用缩放索引方案。视频 token 以时间 $t$ 索引，对应的粗粒度运动 token 被赋予缩放索引 $t/4$。这把时间同步直接嵌入位置编码。经 MVS-RoPE 处理后的特征为：

$$
\hat{\bm{f}}_{t,h,w}^{v} = \mathrm{MVS\text{-}RoPE}(\bm{f}_{t,h,w}^{v}, t, h, w) = \mathcal{R}(t, h, w) \cdot \bm{f}_{t,h,w}^{v},
\tag{2}
$$

$$
\hat{\bm{f}}_{t,i}^{m} = \mathrm{MVS\text{-}RoPE}(\bm{f}_{t,i}^{m}, t, i) = \mathcal{R}\!\left(\tfrac{1}{4}t,\ H+i,\ W+i\right) \cdot \bm{f}_{t,i}^{m}.
\tag{3}
$$

其中 $t$ 为时间索引，$i$ 为运动 token 索引，$(h,w)$ 为视觉 token 的空间索引。$H$ 与 $W$ 表示视觉 token 的空间范围。函数 $\mathcal{R}(\cdot)$ 表示 RoPE 编码，根据输入的时间与空间索引施加旋转。

该设计保证：(1) **保留预训练知识**：视频 token 接收与预训练时完全相同的 3D RoPE，确保已学到的有价值表示不被扰动；(2) **强制正确的时间对齐**：$1/4$ 缩放显式编码多速率关系，消解歧义，确保视频与运动在时间上完全同步；(3) **保证模态可区分性**：空间对角扩展防止“位置碰撞”，使模型易于区分视频 token 与人体运动 token。

![图 3. Motion-Video Two-Stage Training Strategy 概览。Phase 1：模型在仅运动数据上预训练。Phase 2：在成对运动–视频数据上做多任务训练，把 “motion-with-video”“motion-to-video” 与 “video-to-motion” 当作三个同时学习的不同任务。](../../../arxiv/geometry_control/echomotion/extracted/figures/training_stages_v3.png)

### 3.2 Motion-Video Two-Stage Training Strategy

由于视觉 token 与运动 token 的特征表示差异大，需要分阶段训练，才能使模型有效处理并对齐这两种模态。为实现该对齐并保证稳定学习，我们提出从预训练 DiT 参数初始化的两阶段策略：

- **Phase 1: Motion-only Pretraining。** 运动分支在仅运动数据集上独立训练，视频分支冻结并停用（省略其输入）。该阶段聚焦于生成运动序列。
- **Phase 2: Motion-video Multi-task Training。** 随后，模型在运动–视频成对数据集上训练，两分支均解冻并激活，从而能同时生成视觉序列与运动序列。

仅做运动预训练的理由，是让运动分支先在自身域上收敛，避免被计算上占主导的视频分支淹没。一旦稳定，该架构即可自然扩展到视频与运动的联合生成，以及双向跨模态控制。因此在 Phase 2，我们用运动–视频成对数据训练，并聚焦于三种互补的交互范式。

如图 3 所示，每种范式被随机采样：(1) **Joint training**：同时生成视频序列与运动序列。(2) **Motion-to-video training**：运动序列作为视频生成的条件输入。(3) **Video-to-motion training**：视频序列用于条件化运动生成。当某一模态作为条件输入时，其特征被保留，不在前向扩散过程中注入噪声。此外，一个轻量 MLP 把任务嵌入投影到潜空间。该任务提示再加到 latents 上，以引导条件 token 预测。在此框架范式下，模型在推理时自然同时具备纯文本引导的运动–视频生成，以及跨模态条件生成。

**In-Context Classifier-Free Guidance（ICCFG）。** 我们的 ICCFG 实现针对 Phase 2 的三种训练范式采用不同的条件策略，与标准 CFG 不同。在 Phase 2，我们施加范式专用的条件丢弃策略。对 joint generation，文本条件被随机丢弃。对 motion-to-video generation，文本条件与运动条件都被随机丢弃。对 video-to-motion generation，文本条件始终丢弃，而视频条件被随机丢弃。相应地，为在推理时利用该能力，记 $\bm{u}_{\theta}(\cdot)$ 为扩散模型。对 Joint Generation 模式：

$$
\bm{o}^{v}_{t},\ \bm{o}^{m}_{t} = \bm{u}_{\theta}(x_{t}, m_{t}, \emptyset) + \omega_{1}\bigl(\bm{u}_{\theta}(x_{t}, m_{t}, y) - \bm{u}_{\theta}(x_{t}, m_{t}, \emptyset)\bigr),
\tag{4}
$$

其中 $\bm{o}^{v}_{t}$ 与 $\bm{o}^{m}_{t}$ 分别表示时间步 $t$ 的视频预测与运动预测。$\emptyset$ 表示缺少某一条件，$\omega_{1}$ 为文本条件的 guidance scale。对 Motion-to-Video Generation 模式，输出可写为：

$$
\bm{o}_{v}^{t} = \bm{u}_{\theta}(x_{t}, \emptyset, \emptyset) + \omega_{1}\bigl(\bm{u}_{\theta}(x_{t}, m_{t}, y) - \bm{u}_{\theta}(x_{t}, m_{t}, \emptyset)\bigr) + \omega_{2}\bigl(\bm{u}_{\theta}(x_{t}, m_{t}, \emptyset) - \bm{u}_{\theta}(x_{t}, \emptyset, \emptyset)\bigr),
\tag{5}
$$

其中 $\omega_{2}$ 为运动条件的 guidance scale。对 Video-to-Motion Generation 模式：

$$
\bm{o}^{m}_{t} = \bm{u}_{\theta}(\emptyset, m_{t}, \emptyset) + \omega_{2}\bigl(\bm{u}_{\theta}(x_{t}, m_{t}, \emptyset) - \bm{u}_{\theta}(\emptyset, m_{t}, \emptyset)\bigr).
\tag{6}
$$

这使模型能利用各模态提供的丰富上下文，同时有选择地启用特定生成能力。

![图 4. *HuMoVe* 数据集概览。（a）数据集构成的 Voronoi treemap。（b）文本 caption 的词云。（c）样本帧及其配对的 3D mesh 重建。](../../../arxiv/geometry_control/echomotion/extracted/figures/dataset_-1.png)

### 3.3 *HuMoVe* 数据集

当前开源资源在联合视频–运动生成上存在显著缺口。先前工作的数据集（AMASS、HumanML3D、Go 等）往往不适用，通常是因为它们模态单一（缺少视觉上下文），或视频质量低、背景与人物冗余。为弥补这一局限，我们构建 *HuMoVe*：大规模、高质量、场景与人物多样的数据集。*HuMoVe* 中每个视频都配有描述性文本 caption 以及对应的 3D SMPL 运动参数，形成紧密对齐的多模态语料，供先进生成建模使用。

我们实现完整的数据处理管线：从多种来源自动收集，再对审美质量与主体焦点做严格过滤，最后进行精确的 3D human mesh recovery 与标注，得到约 80,000 条最终数据。管线的详细分解见附录 A.4。所得 *HuMoVe* 数据集极为多样。如图 4(a) 的 Voronoi treemap 所示，它覆盖 9 个大类（以不同颜色组表示）与 38 个子类（各大类内的深浅变化），从细微日常动作到动态复杂表演。完整类别划分见附录 A.4。这一审慎整理为开发具有运动学意识的生成模型提供了干净、全面且具有挑战性的基础。

---

## 4. 实验

### 4.1 实现细节

我们在开源基座模型的两个变体 Wan2.1-1.3B 与 Wan2.2-5B 上做实验，以验证本方法有效性。初始的 motion-only 训练阶段利用由所提出 *HuMoVe* 数据集与大规模 HumanML3D 数据集组成的复合运动数据；HumanML3D 是公开仓库，含超过 14,616 条 SMPL 格式人体运动。该阶段训练 15k steps。随后，motion-video multi-task training 阶段在我们收集的运动–视频成对数据上再训练 12k steps。这些 step 数由经验确定：在 held-out 验证集上同时监控损失收敛与定性质量。Wan2.2-5B+EchoMotion 模型的训练量为 4000 A100 GPU hours。整个训练过程在 32 张 NVIDIA A100 GPU 上进行，约 4 天。

**表 1.** 与基线模型在视频生成上的对比。同时报告 human evaluation 与 automatic metrics。**加粗**为最优，<u>下划线</u>为次优。

| Method | Human Anatomy | Motion Smoothness | Dynamic Degree | Aesthetic Quality | Video Quality | Prompt Following | Posture Plausibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CogVideoX-2B | 61.7 | 97.0 | 49.4 | 51.6 | 55.3 | 52.1 | 53.6 |
| Wan-1.3B | 78.1 | 98.2 | 60.6 | **60.1** | 68.2 | 70.3 | 64.0 |
| Video Tuning(Wan-1.3B) | 77.4 | 98.3 | 61.6 | 59.7 | 69.3 | 73.2 | 65.5 |
| EchoMotion(Wan-1.3B) | 79.6 | 98.9 | 61.9 | <u>60.0</u> | 71.3 | 73.2 | 66.1 |
| CogVideoX1.5-5B | 65.3 | 98.5 | 54.4 | 53.2 | 62.5 | 60.4 | 59.4 |
| Wan-5B | 83.0 | <u>98.9</u> | 62.2 | 58.3 | <u>72.8</u> | 78.9 | 68.9 |
| Video Tuning(Wan-5B) | <u>83.1</u> | 98.7 | <u>63.1</u> | 57.9 | 72.3 | <u>79.6</u> | <u>70.2</u> |
| EchoMotion(Wan-5B) | **85.1** | **99.3** | **64.0** | 58.3 | **81.0** | **81.5** | **81.6** |

表中左侧四列为 Auto Metrics，右侧三列为 Human Eval。

### 4.2 文本到视频生成

**评估协议。** 为做全面评估，我们构建新 benchmark，覆盖从日常活动到极限运动的广泛人体运动谱系。该 benchmark 包含多样提示（每类 30 条），覆盖：体操与田径中精确、高动量的动作；舞蹈中流畅、富有表现力的运动；球类与格斗运动中的反应性与交互场景；以及日常生活中的自然手势。定量分析使用 VBench 与 VBench-2.0 的自动指标，并辅以 human user studies 收集数值评分。这些定量发现再由定性可视化补充，以说明本方法的表现。

**定量结果。** 我们在 1.3B 与 5B 两个尺度上，相对 video-only 开源基线评估本方法。表 1 汇总的结果表明，联合建模在运动保真上带来实质增益。在自动指标上，EchoMotion 相对基线显著提升 Motion Smoothness 与 Anatomical Consistency。值得注意的是，这一提升并未降低 Aesthetic Quality 分数。这些结果得到 human evaluation 的有力印证：对以人为中心的提示，参与者为 EchoMotion 在 Video Quality、Motion Plausibility 与 Prompt-Following 上给出显著更高的评分。

![图 5. 与 5B 基线的定性对比。本模型（EchoMotion，右）生成解剖正确、语义连贯的人体运动，消解基线（左）中的严重伪影与组合失败。](../../../arxiv/geometry_control/echomotion/extracted/figures/qualitative_.png)

**定性结果。** 图 5 突出 Video-only 训练在复杂、以人为中心提示上的关键局限。Wan2.2-5B 基线虽能生成尚可的审美效果，却持续违反运动学约束，产生严重解剖伪影（例如缠结的体操运动员、变形的滑板者）。此外，基线难以处理语义组合性，无法执行多步骤训练序列。相比之下，通过联合建模视频与人体运动，本模型生成具有合理解剖的主体，并成功遵循组合式指令。在所有挑战案例中，本模型都生成连贯且物理合理的运动，表明联合分布建模带来更优的人体运动学内部表示。

除优于基线外，本方法还展现出显著的多样性与复杂的多模态生成能力。图 6 展示其生成广度：从高尔夫挥杆到单板滑雪空中技巧，在多样场景中成功合成大量高质量、运动学上合理的动作。联合建模的有效性还直接由图 7 说明：模型从同一提示同时生成视频及其对应的、时间对齐的 SMPL 序列。这种共生成表明，模型输出并非单纯的像素级合成，而是从根本上以人体运动学的内部表示为条件。

![图 6. EchoMotion 的 text-to-video 结果，在多样以人为中心的场景中同时展现强 prompt 对齐与高运动学合理性。更多示例与视频结果见附录 A.6 与补充材料。](../../../arxiv/geometry_control/echomotion/extracted/figures/generations_.png)

![图 7. EchoMotion 联合生成 SMPL 运动序列（左）与视频（右），表明已学到联合分布。](../../../arxiv/geometry_control/echomotion/extracted/figures/joint_gen_.png)

### 4.3 跨模态补全

模型的统一设计带来强跨模态能力，如图 9 所示。通过把任务组织为模态补全，EchoMotion 双向运作：(a) 它能合成精确跟随给定运动序列的高保真视频（motion-to-video）；(b) 它能从输入视频恢复底层 SMPL 运动（video-to-motion，即 inverse kinematics）。用单一模型同时完成生成与 inverse kinematics，凸显联合建模的显著优势。跨模态补全任务的进一步定量评估见附录 A.7.2 与 A.7.3。

![图 8. MVS-RoPE 的效果。](../../../arxiv/geometry_control/echomotion/extracted/figures/cross_modal_attn-1.png)

![图 9. EchoMotion 的跨模态补全。**(a)** 由运动与文本做 Motion-to-Video 合成。**(b)** Video-to-Motion 恢复（inverse kinematics）。](../../../arxiv/geometry_control/echomotion/extracted/figures/multi_modal_fill_.png)

### 4.4 消融实验

本节通过消融验证框架的关键组件。更多消融详见附录 A.8。

**Joint Modeling vs. Video-Only Modeling。** 我们将联合训练与仅在本数据集视频数据上微调的基线（Video Tuning）对比。如表 1 所示，Video-only tuning 仅带来边际改进，且未能提升关键运动学指标；而 EchoMotion 在 Human Anatomy 上、尤其在 Posture Plausibility 上取得实质增益。该结果确认我们的核心假设：高质量人体运动合成的关键，在于训练时对外观与运动学做联合建模；仅仅增加更多以人为中心的视频数据，收益有限。

**MVS-RoPE 设计。** 为验证 MVS-RoPE 对齐视频与运动模态的设计，我们在图 8 中可视化 self-attention score。关键的是，运动时间序列长度为视频时间序列的四倍，要求模型学习非平凡的 4:1 时间映射。在带 MVS-RoPE 的模型 (a) 中，attention map 完美反映这一点：video-to-motion attention 形成清晰的浅对角，而 Motion-to-Video attention 形成对应的陡对角。这种不对称结构是模型已学到正确时间对齐的直接证据。相比之下，无 MVS-RoPE 的基线 (b) 完全失败：attention 分散，所需对角结构缺失，表明其无法同步两模态。

---

## 5. 结论

我们提出 EchoMotion：在 Dual-Modality DiT 内联合建模外观与运动学，从而生成人体视频的创新框架。本方法用基于 SMPL 的参数化表示刻画人体运动，将其 token 与视频 token 拼接到统一的多模态序列中。为在该序列内实现有效信息交换，我们引入 MVS-RoPE：一种新的位置嵌入，在多模态联合 self-attention 期间强制视频与运动模态的时间对齐。此外，Motion-Video Two-Stage Training Strategy 赋予模型多样的双向能力：可控的 motion-to-video 与 video-to-motion 生成。该框架的开发得到 *HuMoVe* 的支持：我们引入的新大规模数据集，约含 80,000 条高质量视频–运动对。

**局限性。** 当前框架限于单人生成。扩展到多人场景在架构上可行（拼接 SMPL token），但需要构建带逐人标注的新大规模数据集。鉴于这是显著的资源投入，本工作优先完善单人生成模型，把多人生成留作未来有前景的方向。

---

## 附录 A

### A.1 LLM 使用说明

我们在若干方面使用大语言模型（LLMs）辅助本研究。核心智力贡献——包括研究问题的形式化、EchoMotion 架构设计、MVS-RoPE 机制，以及混合多模态 in-context learning 策略——完全由人类作者构思与发展。LLMs 的角色仅限于以下辅助任务：

1. **写作与文稿润色：** 我们使用通用 LLMs（如 OpenAI 的 ChatGPT）做语法修正、文风改进，并提升文稿清晰度与可读性。科学叙述、结构与全部主张仍为作者原创。

2. **数据整理支持：** 为辅助收集 *HuMoVe* 数据集，我们用 LLM 生成与复杂人体运动相关的广泛关键词与检索查询。这有助于从公有领域来源系统识别相关视频内容。

3. **自动视频标注：** *HuMoVe* 数据集标注过程的重要部分由 Qwen-VL-Narrator 协助。该模型负责为每个视频片段生成细粒度文本描述的初稿，覆盖 (1) 主体外观与服饰，(2) 背景语境，以及 (3) 动作描述。

### A.2 预备知识

**Diffusion Transformers（DiT）。** Diffusion Transformers（DiTs）用纯 transformer 架构替代扩散模型中常规的 U-Net 骨干，展现出更优性能与可扩展性。DiT 在潜空间上运行，由三个主要阶段组成：

- 输入编码器，把带噪潜变量 “patchify” 成 token 序列。
- 一系列处理这些 token 的 transformer block。这些 block 通过 adaptive layer normalization（adaLN）以扩散时间步 $t$ 及其他上下文（如文本嵌入）为条件。
- 最终解码器，把输出 token “unpatchify” 回预测的噪声图。

**表 2.** 大类到对应细粒度类别的映射。

| Major Category | Fine-grained Categories |
| --- | --- |
| Urban & Outdoor | Archery, Cycling, Equestrian, Highlining, Motocross, Parkour, Rock Climbing, Skateboarding, Street Workout |
| Water Sports | Kayaking, Kitesurfing, Paddleboarding, Surfing, Swimming, Wakeboarding, Windsurfing |
| Athletics & Fitness | Athletics, Olympics |
| Ball Sports | Basketball, Golf, Soccer, Tennis, Volleyball |
| Combat Sports | Boxing, Judo, MMA, Taekwondo, Wrestling, Wushu |
| Dance & Artistic | Artistic Roller Skating, Breakdancing, Dancing |
| Gymnastics & Acrobatic | Gymnastics, Trampoline |
| Ice & Snow Sports | Figure Skating, Skiing & Snowboarding |
| General & Others | Daily Activity, Other |

**Flow Matching。** 训练时，视频 latents $x_i$ 按时间步 $t \in [0,1]$ 被随机噪声 $x_0 \sim \mathcal{N}(0,I)$ 扰动，即

$$
x_{t} = t\, x_{1} + (1-t)\, x_{0}.
\tag{7}
$$

模型被训练为预测速度 $v_t = x_1 - x_0$，通过最小化模型预测与 $v_t$ 的 MSE 损失：

$$
\mathcal{L} = \mathbb{E}_{x_0, x_1, y, t}\bigl\| u(x_t, y, t; \theta) - v_t \bigr\|^2,
\tag{8}
$$

其中 $y$ 为输入文本描述，$\theta$ 为模型权重，$u(x_t, y, t; \theta)$ 为 DiT 模型的预测。

**Rotary Position Embedding（RoPE）。** Rotary Position Embedding 通过对 query 与 key 向量施加位置相关的旋转来编码绝对位置信息，近期 Transformer 通常用它在 self-attention 层内注入位置信息。给定序列中带 3D 位置索引 $(t,h,w)$ 的 $d$ 维视觉嵌入 $x_{t,h,w}$，3D RoPE 按维施加：

$$
\tilde{x}_{t,h,w} = R^{\Theta}_{t,h,w}\, x_{t,h,w},
\tag{9}
$$

其中 $R^{\Theta}_{t,h,w}$ 为由给定 3D 位置索引与预定义基频 $\Theta$ 决定的 3D 旋转矩阵。

### A.3 视频与人体运动的联合分布

我们提出：与其只建模视频分布，人体运动视频的分布 $p_{\theta}(z)$ 可由视频及其对应人体运动参数的联合分布来建模。具体地，给定文本提示 $y$，分布 $p(z|y)$ 可写为

$$
p(z|y) = p_{\theta}(x, m \mid y),
\tag{10}
$$

其中 $x$ 与 $m$ 分别表示视频与人体运动参数。模型 $f$ 的目标是学习联合分布 $p_{\theta}(x, m \mid y)$。推理时，我们通过对视频潜变量与人体运动潜变量联合迭代去噪，生成视频及对应人体运动参数：

$$
x,\ m = f(y).
\tag{11}
$$

回顾式 (10)，可得到

$$
p_{\theta}(x \mid m, y) = \frac{p_{\theta}(x \mid y)}{p_{\theta}(x, m \mid y)},
\tag{12}
$$

其中 $p_{\theta}(x \mid m, y)$ 对应给定人体运动时的视频条件分布。因此，从 $p_{\theta}(x \mid m, y)$ 采样视频，等价于生成与给定人体运动 $m$ 及文本提示 $y$ 匹配的视频，即可控视频生成。类似地，给定视频，我们可通过从运动条件分布采样来恢复视频中的人体运动参数：

$$
p_{\theta}(m \mid x, y) = \frac{p_{\theta}(m \mid y=\emptyset)}{p_{\theta}(x, m \mid y=\emptyset)}.
\tag{13}
$$

### A.4 *HuMoVe* 数据集细节

**细粒度类别。** 表 2 给出用于组织数据集的两级分类方案。我们把具体、细粒度的活动（如 Skateboarding、Swimming、Boxing）归入更广的主题大类（如 Urban & Outdoor、Water Sports、Combat Sports）。该分类为分析数据集多样性、以及按不同运动类型评估模型表现提供结构化框架。

**原始数据收集。** 为确保原始数据覆盖广泛人体运动，收集过程采用结构化的自上而下路径。我们首先手工定义九个运动大类，如表 2 所示（例如 Urban & Outdoor、Water Sports、Combat Sports）。这些高层类别再作为输入提供给 Large Language Model（LLM），由其生成各类别下多样的具体检索关键词。该任务所用提示示例如下。所得关键词随后用于从开源数据集、电影与互联网等多种来源收集原始数据池。所用提示见 Listing 1。

**Listing 1.** 用于经 LLM 把大类扩展为具体检索关键词的 prompt。

```text
You are a data sourcing expert for computer vision research. Your task is to expand high-level categories of human motion into specific, diverse, and fine-grained search keywords suitable for video platforms.

For each major category provided, generate a list of 20-30 search terms.

Here is an example of the desired input-output format:

**Input:**
Major Category: Combat Sports

**Desired Output Keywords:**
- boxing training drills 4k
- MMA sparring session slow motion
- cinematic Taekwondo high kick
- Judo throw tutorial
- female wrestler practice highlights
- Wushu performance competition
...

Now, please generate keywords for the following list of major categories:
- Urban & Outdoor
- Water Sports
- Athletics & Fitness
- Ball Sports
- Combat Sports
- Dance & Artistic
- Gymnastics & Acrobatics
- Ice & Snow Sports
- General & Others
```

**数据过滤。** 原始视频数据常出现频繁场景切换，并含低审美、低分辨率片段，会损害模型性能。为缓解这些问题，我们用 scene detection 模型把原始视频切成片段，并用 aesthetic scoring 模型滤除低审美质量的片段。由于本工作主要关注前景主角色的运动，我们用 bounding box 检测模型识别视频中的个体，并通过比较 bounding box 的数量与面积滤除含多个主角色的样本。此外，用 DWPose 检测视频中可见关键点数量。可见关键点相对较少的数据被过滤，以避免 3D 人体姿态检测阶段的伪影与抖动。

**Human Mesh Recovery 与后处理。** 以跟踪得到的 bounding boxes 为输入，为获得逐帧高质量 SMPL 参数——包括 $\beta \in \mathbb{R}^{10}$、$\theta \in \mathbb{R}^{24 \times 6}$、$\Gamma \in \mathbb{R}^{6}$ 与 $v \in \mathbb{R}^{3}$——我们使用 CameraHMR 做人体网格估计，因其稳定性与精度高。此外，我们对逐帧 SMPL 参数做时间平滑，并通过 motion retargeting 得到 3D 关键点位置 $J \in \mathbb{R}^{24 \times 3}$。最后，用 Qwen-VL-Narrator 为视频做 caption。该构建机制得到约 80k 高质量成对数据。

### A.5 与一线商业模型的对比

我们与领先商业模型做定量对比，在 40 条专门按运动复杂度选出的提示上评估（表 3）。相对 Veo 3.1、Kling 2.5 Turbo 等一线闭源模型仍有差距，这在模型规模、训练数据与计算资源存在巨大差异的情况下是预期结果。尽管如此，结果突出两点：第一，EchoMotion 取得有竞争力的 Motion Smoothness 分数，确认联合建模对运动学质量的有效性。第二，它显著优于其 video-only 基线（Wan-5B），尤其在 Posture Plausibility 与 Human Anatomy 上。这从经验上验证本方法为人视频生成带来实质增益。

**表 3.** 与 state-of-the-art 模型的定量对比。本模型同时对照领先闭源模型与开源基线。**加粗**为最优，<u>下划线</u>为次优。

| Method | Human Anatomy | Motion Smoothness | Dynamic Degree | Aesthetic Quality | Video Quality | Prompt Following | Posture Plausibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Veo 3.1 | **85.5** | **99.3** | <u>67.5</u> | **59.9** | <u>90.2</u> | **89.9** | **92.1** |
| Kling 2.5 Turbo | <u>84.7</u> | 99.0 | **70.0** | <u>59.7</u> | **90.4** | <u>89.3</u> | <u>91.0</u> |
| Wan-5B | 82.3 | 98.7 | 62.2 | 58.3 | 72.7 | 77.5 | 69.1 |
| EchoMotion(Wan-5B) | 83.2 | <u>99.1</u> | 64.0 | 58.2 | 80.4 | 81.4 | 80.8 |

表中左侧四列为 Auto Metrics，右侧三列为 Human Eval。

### A.6 更多定性结果

本节提供扩展的定性结果画廊，进一步展示 EchoMotion 的能力。

图 10 展示多样的 text-to-video 生成结果，突出模型在各种活动与环境中处理复杂提示的能力。图 11 聚焦 motion-to-video 任务，说明 EchoMotion 如何把 3D 运动序列精确转译为写实视频，同时遵循对外观与语境的文本描述。

![图 10. EchoMotion 的额外 text-to-video 生成结果。这些示例展示模型生成多样高质量视频的能力，覆盖皮划艇、滑板、滑雪等活动与多种环境。结果忠实遵循复杂、细致的文本描述。](../../../arxiv/geometry_control/echomotion/extracted/figures/generations_appendix_.png)

![图 11. motion-to-video 任务的额外结果。这些示例说明 EchoMotion 能把给定 3D 运动序列准确渲染为视觉连贯的视频。关键的是，模型对动作遵循来自运动输入的运动学引导，同时对主体外观、服饰与场景语境遵循文本提示。](../../../arxiv/geometry_control/echomotion/extracted/figures/mt2v_appendix_.png)

### A.7 更多定量结果

![图 12. 按类别的定量对比。我们给出本模型（Wan2.2 5B + EchoMotion）与其他基线在九个不同运动类别上的详细表现分解，外加总体平均。模型在四个指标上评估：Aesthetic Quality、Dynamic、Human Anatomy 与 Motion Smoothness。本模型在所有测试场景中持续取得更优表现，尤其在 Human Anatomy 与 Motion Smoothness 上，表明其稳健性与高保真运动生成能力。](../../../arxiv/geometry_control/echomotion/extracted/figures/class_wise_performance_.png)

#### A.7.1 Text-to-Video

为更细粒度地理解模型表现与稳健性，我们给出跨多种运动类别的详细定量对比。图 12 可视化所提出模型 EchoMotion(Wan-5B) 相对若干基线与消融的表现。评估覆盖九个类别——Athletic、Arts、Ball、Daily、Fight、Gymnastics、Ice Show Sports、Outdoor 与 Water Sports——以及总体平均表现。模型在四个关键指标上评估：Aesthetic Quality、Dynamic、Human Anatomy 与 Motion Smoothness。

结果呈现出一致趋势：尽管所有模型在 Aesthetic Quality 上都取得有竞争力的分数，我们的最终模型 EchoMotion(Wan-5B) 在对运动保真最关键的指标——Human Anatomy 与 Motion Smoothness——上建立显著领先。例如，在 “Average Performance” 对比中，本模型在这两项上以明显差距优于次优基线。这强烈表明本方法在生成解剖正确的人体、并保证时间连贯、平滑的运动方面特别有效，而这正是人体视频生成中的常见挑战。

模型的优势还由其在广泛运动类型上的持续高表现所验证。从 “Gymnastics” 与 “Arts” 中精细、精确的动作，到 “Water Sports” 与 “Fight” 中大尺度动态动作，本模型持续排名第一。这证明方法的可泛化性与稳健性：它并未过拟合到某一类运动，而能有效处理多样人体活动。

#### A.7.2 Motion-to-Video

![图 13. Motion-to-Video 任务与基线方法的定性对比。](../../../arxiv/geometry_control/echomotion/extracted/figures/qualitative_compare_mt2v.png)

我们在来自 VACE-Benchmark 的案例上，把 motion-to-video 任务与若干领先方法做定量对比，包括 Text2Video-Zero、Follow-Your-Pose、VACE-14B，以及专用动画模型 Wan2.2-Animate-14B。如表 4 所示，EchoMotion 取得极具竞争力的结果，展现出色的视频质量与强运动学合理性。图 13 的定性结果在视觉上印证这些指标：角色运动高度忠实于输入运动，同时角色外观与背景场景严格遵循文本提示。

需要澄清的是，我们的任务是跨模态补全挑战（同时由文本与运动生成），这与标准图像动画（用运动驱动源图像）不同。与需要源图像的 Wan2.2-Animate-14B 对比时，我们提供真值视频的第一帧，并经 Flux-Kontext 变换后作为其输入。由于该设置不为 Wan2.2-Animate-14B 使用文本提示做控制，Prompt Following 指标对该方法不适用。

结果突出 EchoMotion 的关键优势：它能联合理解并合成文本与运动输入，灵活性更高。尤其值得注意的是，尽管 EchoMotion 更紧凑（5B）且是通用的 motion-video 多任务模型，其表现仍与专攻单一任务的更大模型相当。关键的是，这种通用性以显著的参数效率实现。与需要训练辅助控制模块（例如 ControlNet 分支）的做法不同，EchoMotion 通过统一的 motion-video 多任务训练原生支持 motion-to-video 任务，不引入任何额外参数。

**表 4.** 人体运动生成上的定量对比。最优结果 **加粗**，次优结果 <u>下划线</u>。左侧四列为 Video Quality，右侧三列为 User Study。

| Method | Aesthetic Quality | Motion Smoothness | Overall Consistency | Temporal Flickering | Prompt Following | Pose Consistency | Overall Quality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Text2Video-Zero | 57.6 | 79.7 | 23.9 | 76.6 | 44.2 | 70.1 | 40.9 |
| Follow-Your-Pose | 48.8 | 90.1 | 26.1 | 88.0 | 50.9 | 79.2 | 42.1 |
| VACE-14B | <u>60.2</u> | 98.6 | 26.4 | 97.3 | **79.2** | <u>82.4</u> | <u>66.9</u> |
| Wan2.2-Animate-14B | **61.4** | <u>98.9</u> | **28.1** | **99.2** | - | **87.2** | **68.9** |
| EchoMotion(Wan-5B) | 59.2 | **99.1** | <u>26.9</u> | <u>98.4</u> | <u>78.2</u> | 82.2 | 65.2 |

#### A.7.3 Video-to-Motion

遵循 ChatHuman 等近期工作的评估协议，我们在 3DPW 测试集的 200 个样本上评估 Video-to-Motion 能力。我们报告 Mean Per-Joint Position Error（MPJPE）与 Procrustes-Aligned MPJPE（PA-MPJPE），并把对比方法分为 reconstruction-based 与 generative-based 两组，如表 5 所示。

结果表明，本方法并未超越 HMR2.0 等专用、基于重建的模型，这是预期结果。基于重建的方法通常针对 human mesh recovery 这一单一任务优化，训练时往往依赖来自 2D 关键点标注的强监督。相比之下，EchoMotion 被设计为通用的 motion-video 多任务生成框架，天然涉及专精与灵活之间的权衡。在更可比较的 generative-based 类别内，本方法取得与 state-of-the-art ChatHuman 相当的表现，表明统一模型不仅支持运动恢复，也支持更广泛的生成任务。

**表 5.** 3DPW 数据集上 Human Mesh Recovery 表现的定量对比。方法分为 reconstruction-based 与 generative-based。最优结果 **加粗**。

|  | Method | PA-MPJPE  | MPJPE  |
| --- | --- | ---: | ---: |
| Rec-Based | SPIN | 62.9 | 102.9 |
| Rec-Based | HMR2.0 | **58.4** | **91.0** |
| Gen-Based | ChatPose | 81.9 | 163.6 |
| Gen-Based | ChatHuman | 58.7 | 91.3 |
| Gen-Based | EchoMotion(Video-to-Motion) | 59.8 | 94.1 |

#### A.7.4 Motion Quality

尽管 EchoMotion 主要为视频生成设计，我们也评估其运动合成质量。表 6 将其表现与专用运动生成模型对比。需要注意，用 Frechet Inception Distance（FID）与基线方法直接对比并不适用，因为该指标高度依赖训练数据分布。本模型在所提出的 *HuMoVe* 数据集上训练，而基线模型（例如 MLD、MotionGPT）在 HumanML3D 等不同的仅运动数据集上训练。因此，我们仅对在 *HuMoVe* 上训练的模型报告 FID。

为做公平且实用的对比，我们用 LLM 生成 50 条多样提示，把生成的运动参数渲染为 mesh 视频，再请人类标注者在三个关键方面打分：Pose Plausibility（PP）、Prompt Following（PF）与 Motion Smoothness（MS）。结果表明 EchoMotion 在 Pose Plausibility 与 Motion Smoothness 上取得有竞争力的表现，并在 Prompt Following 上领先。我们将这一更强的指令遵循能力归因于联合训练与采样范式：运动模块受益于大规模预训练视频模型丰富的语义理解。

**表 6.** 运动生成质量的定量对比。指标为 Frechet Inception Distance（**FID**）、Pose Plausibility（**PP**）、Prompt Following（**PF**）与 Motion Smoothness（**MS**）。

| Method | FID  | PP $\uparrow$ | PF $\uparrow$ | MS $\uparrow$ |
| --- | ---: | ---: | ---: | ---: |
| MLD | - | 71.4 | 63.7 | 91.5 |
| MotionGPT | - | <u>80.1</u> | 72.3 | **93.9** |
| EchoMotion (Wan 1.3B) | **10.9** | 79.5 | <u>78.8</u> | 92.2 |
| EchoMotion (Wan 5B) | 11.3 | **80.8** | **82.3** | <u>92.4</u> |

### A.8 更多消融实验

本节给出额外消融，进一步验证 EchoMotion 的架构设计选择。

#### A.8.1 MVS-RoPE 中位置碰撞的影响

为证明所提出 MVS-RoPE 中空间扩展设计的必要性，我们考察模态之间空间索引重叠的影响。我们对比两种配置：

**(a) EchoMotion（本文）：** 运动 token 被当作视觉潜空间的对角扩展，以保证唯一的空间标识。编码表述为：

$$
\hat{\bm{f}}_{t,i}^{m} = \mathrm{MVS\text{-}RoPE}(\bm{f}_{t,i}^{m}, t, i) = \mathcal{R}(t, H+i, W+i) \cdot \bm{f}_{t,i}^{m},
\tag{14}
$$

其中 $H$ 与 $W$ 表示视频 latents 的空间尺寸。

**(b) Positional Collision（基线）：** 运动 token 被赋予从原点开始的空间索引，与初始视觉 token 相同。这导致坐标 “碰撞”：

$$
\hat{\bm{f}}_{t,i}^{m} = \mathrm{MVS\text{-}RoPE}(\bm{f}_{t,i}^{m}, t, i) = \mathcal{R}(t, i, i) \cdot \bm{f}_{t,i}^{m}.
\tag{15}
$$

两模型在相同设置下训练 1,600 次迭代。定性对比见图 14。结果反差鲜明：使用所提出 MVS-RoPE 的模型（左）产生视觉上吸引人、时间连贯的视频。相反，Positional Collision 基线（右）出现严重视觉退化，生成结构崩塌与噪声的输出。

这一退化的原因在于：在 attention 机制中识别 token 的模态高度依赖其位置嵌入。当运动 token 与视频 token 共享同一空间坐标（碰撞）时，会产生位置歧义。这迫使运动 token 在 attention 中干扰视觉 token，破坏视频骨干的预训练生成先验，并阻止模型有效区分两模态的潜表示。

![图 14. 位置嵌入发生 positional collision 的负面效果。](../../../arxiv/geometry_control/echomotion/extracted/figures/positional_collisions.png)

#### A.8.2 两阶段训练方案

为验证 motion-only 预训练的有效性，我们做消融，对比两个模型在 Phase 2（motion-video 联合训练）上的收敛。完整模型从 motion-only 预训练得到的 checkpoint 开始该阶段；基线模型跳过 Phase 1，通过复制视频分支权重来初始化运动分支后直接开始联合训练。如图 15(a) 所示，收益清晰且显著。带预训练的本模型（橙色曲线）不仅 (1) 以显著更低的 Flow Matching Loss 起步，还 (2) 收敛快得多，并且 (3) 相对基线（蓝色曲线）达到更好的最终损失值。这表明在 Phase 1 建立强运动先验，为后续联合训练提供更优初始化。该预训练防止运动分支的学习信号被计算上占主导的视频分支淹没，从而使联合分布学习更稳定、更高效。

![图 15. 关键设计的消融。(a) 两阶段训练方案（w/ Motion-Only Pretraining）与单阶段基线（w/o Motion-Only Pretraining）的训练损失对比。(b) Dual-stream DiT 与 Single-stream DiT 基线的训练表现对比。本架构持续取得更低损失，并收敛到更好的最终值。](../../../arxiv/geometry_control/echomotion/extracted/figures/training_loss_ablation.png)

#### A.8.3 Dual-Branch 架构

我们做消融以验证双分支 DiT 架构相对单流替代方案的优越性。单流基线把拼接后的视频 token 与运动 token 通过共享的一套 Q、K、V 投影与 FFN 层处理。相比之下，双分支设计为每个模态的投影与 FFN 层采用分开的权重（即 experts）。

如图 15(b) 所示，两种架构的训练损失曲线揭示清晰的性能差距。双分支模型（橙色曲线）在整个训练过程中相对单流基线（蓝色曲线）持续取得更低的 flow matching loss。值得注意的是，本模型收敛到显著更低的最终损失，而基线在更高值处进入平台，放大插图中已标出。该结果表明：为视频与运动模态提供专用处理路径，使模型能学习更有效、更专门化的表示。这一做法避免共享权重、单流架构中可能出现的特征干扰，最终带来更优模型表现与更稳健的联合分布模型。

### A.9 实验细节

我们把两个预训练视频基础模型 Wan2.1-1.3B 与 Wan2.2-5B 接入 dual-modality block 进行适配。

- 对 1.3B 模型，我们把全部原始 DiT block 替换为 video-motion block。最终模型参数量为 2.6B。
- 对更大的 5B 模型，我们采用混合策略，把一半视频 block 替换为 dual-modality block，最终模型参数量为 7.5B。

所有模型均在 NVIDIA A100 80GB GPU 上训练。2.6B 模型预训练约需 2,300 GPU hours，7.5B 模型约需 4,000 GPU hours。详细训练超参数见表 7 与表 8。

**表 7.** EchoMotion 1.3B 模型的实验设置。

| Parameter | Value |
| --- | --- |
| Transformer dim | 1536 |
| Numbers of heads | 24 |
| Numbers of layers | 30 |
| Number of video-only blocks | 0 |
| Number of video-motion blocks | 30 |
| Video height | 480 |
| Video width | 832 |
| Video frame | 81 |
| FPS | 16 |
| Batchsize | 2 |
| Train timesteps | 1000 |
| Train shift | 8.0 |
| Optimizer | AdamW |
| Learning rate | 8e-6 |
| Weight decay | 0.001 |
| Sample timesteps | 50 |
| Sample shift | 8.0 |
| Sample guidance scale | 6.0 |

**表 8.** EchoMotion 5B 模型的实验设置。

| Parameter | Value |
| --- | --- |
| Transformer dim | 3072 |
| Numbers of heads | 24 |
| Numbers of layers | 30 |
| Number of video-only blocks | 15 |
| Number of video-motion blocks | 15 |
| Video height | 708 |
| Video width | 1280 |
| Video frame | 121 |
| FPS | 24 |
| Batchsize | 1 |
| Train timesteps | 1000 |
| Train shift | 8.0 |
| Optimizer | AdamW |
| Learning rate | 8e-6 |
| Weight decay | 0.001 |
| Sample timesteps | 50 |
| Sample shift | 8.0 |
| Sample guidance scale | 6.0 |

![图 16. 模型规模与计算代价的可视化。](../../../arxiv/geometry_control/echomotion/extracted/figures/model_comparison_.png)

### A.10 计算量与参数分析

如图 16 所示，有必要区分模型参数量与计算代价。引入双分支架构虽显著增加总参数量，但计算开销（即 FLOPs）的增长不成比例地小。例如，我们的 1.3B 双分支模型参数量是其 video-only 对应模型的两倍，但单次前向仅多需 12.9% 的 FLOPs。这一趋势在更大的 5B 模型上同样成立：混合架构引入 51% 更多参数，但计算代价仅增加 10.3%（从 237.0 到 261.4 PFLOPs）。

FLOPs 的这种不成比例小幅增长是架构设计的直接后果。额外参数专门用于处理运动 token；尽管其信息丰富，它们只占序列总长度的一小部分（例如 1.3B 模型中 4,131 个运动 token vs. 32,760 个视频 token）。因此，对实际推理延迟的影响是边际的。这表明双分支路径是纳入多模态能力的高效方法：以计算足迹的最小增加，显著扩展模型功能。

### A.11 社会影响与安全措施

以运动学模型为根基的可控人体视频生成——以 EchoMotion 为代表——具有深远社会影响。通过使高质量、运动学上合理、并能直接控制人体运动的视频成为可能，本框架降低了获取复杂动画与视觉特效工具的门槛。这一创新可大幅精简电影（预可视化）、电子游戏（写实角色动画）、虚拟现实（逼真 avatar）等行业的工作流，甚至可用于运动科学与物理治疗等专门领域，以可视化复杂生物力学。从结构化运动数据生成视频，提供前所未有的创作灵活性，以及数字人创建的新范式。

然而，此类生成技术的能力也带来显著挑战。高保真动画的自动化可能导致传统 3D 动画师与动作捕捉艺术家的岗位替代，从而需要行业范围的适应，并把重心从手工执行转向创意指导。更关键的是，误用潜力构成严重伦理关切。生成某人执行其从未做过的复杂动作的写实视频，可能被用于制作极具说服力的 deepfakes、身份冒用或虚假信息，从而侵蚀公众信任。此外，确保 *HuMoVe* 数据集及在其上训练的模型免于人口统计或身体层面的偏见，对于防止生成强化有害刻板印象的内容至关重要。

为应对这些问题，我们实施稳健的安全措施。模型继承其基础模型（Wan2.1 与 Wan2.2）的安全机制，包括检测并阻止生成不当或有害内容的过滤器。我们承诺遵守关于本技术使用的严格伦理准则。通过开源模型与 *HuMoVe* 数据集，我们旨在促进透明、使独立审计成为可能，并鼓励社区驱动地发展负责任、合乎伦理的以人为中心的生成式 AI。
