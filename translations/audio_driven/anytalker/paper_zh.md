# AnyTalker：以交互性精炼缩放多人说话视频生成（AnyTalker: Scaling Multi-Person Talking Video Generation with Interactivity Refinement）

**作者：** Zhizhou Zhong$^{1,2}$、Yicheng Ji$^{2,3}$、Zhe Kong$^{1}$、Yiying Liu$^{2*}$、Jiarui Wang$^{2}$、Jiasun Feng$^{2}$、Lupeng Liu$^{2,4}$、Xiangyi Wang$^{2,4}$、Yanjia Li$^{2}$、Yuqing She$^{2,4}$、Ying Qin$^{4}$、Huan Li$^{3}$、Shuiyang Mao$^{2}$、Wei Liu$^{2}$、Wenhan Luo$^{1\dagger}$

**机构：** $^{1}$Hong Kong University of Science and Technology；$^{2}$Video Rebirth；$^{3}$Zhejiang University；$^{4}$Beijing Jiaotong University

**贡献说明：** $^{*}$项目负责人；$\dagger$通讯作者

**出处：** arXiv 2025（CVPR 2026 投稿模板）

**arXiv：** [2511.23475](https://arxiv.org/abs/2511.23475)

**项目页：** https://hkust-c4g.github.io/AnyTalker-homepage

**代码：** https://github.com/HKUST-C4G/AnyTalker（本仓库：[`code/audio_driven/anytalker/`](../../../code/audio_driven/anytalker/)）

**原文 TeX：** [`arxiv/audio_driven/anytalker/extracted/main.tex`](../../../arxiv/audio_driven/anytalker/extracted/main.tex)

---

![图 1](../../../arxiv/audio_driven/anytalker/extracted/fig/teaser.png)

**图 1.** 我们提出 AnyTalker，一个面向交互式多人视频生成的强音频驱动框架。它能生成手势丰富、情绪生动、交互自然的视频，并可自由泛化到任意身份，乃至非人类主体。

---

## 摘要

近年来，多人视频生成开始受到关注。尽管已有若干初步工作探索音频驱动的多人说话视频生成，它们往往受困于多样多人数据的高采集成本，以及在驱动多个身份时难以保持连贯交互。为应对这些挑战，我们提出 AnyTalker，一种具有可扩展多流处理架构的多人生成框架。具体而言，我们在 Diffusion Transformer 的注意力块中引入一种新的身份感知注意力机制，迭代处理身份–音频对，从而允许可驱动身份任意缩放。此外，训练多人生成模型通常需要海量多人数据。我们提出的训练管线仅依赖单人视频学习多人说话模式，并用少量真实多人片段精炼交互性。我们还贡献了一套面向生成多人视频自然性与交互性的评测指标与数据集。大量实验表明，AnyTalker 在口型同步、视觉质量与自然交互上表现突出，在数据成本与身份可扩展性之间取得了有利平衡。

---

## 1. 引言

在数字媒体时代，视频创作已成为媒体平台的关键组成部分。播客、直播带货与娱乐节目往往包含丰富的多人交互，从而催生对多人视频生成的需求。尽管大规模视频生成模型（Stable Video Diffusion、CogVideoX、HunyuanVideo、Wan 等）已为音频驱动说话视频方法（OmniHuman、OmniAvatar、InfiniteTalk、Hallo3、HunyuanVideo-Avatar、Sonic、Wan-S2V、MultiTalk 等）提供了坚实骨干，使单人主体能够获得逼真口型，这些模型仍难以容纳多人交互的复杂性。

这一局限推动了可扩展到任意人数、处理多流信号、并对每个主体实施差异化控制的多人驱动方法。尽管近期工作（InterActHuman、Bind-Your-Avatar、MultiTalk）提出了处理多流音频的方案，这些方法通常需要数百到数千小时精心整理的多人数据，采集成本高昂、可复现性受限。具体而言，多人场景的训练数据采集因轮流说话、角色切换、眼神等非语言线索而更难标注。现有音视频数据集（HDTF、VFHQ、Hallo3、EchoMimicV2、OpenHumanVid、VoxCeleb2 等）大多聚焦单人独白或孤立面部动画，因而难以直接用于训练音频驱动多人生成模型。此外，当前多人驱动方法常常难以建模真正的交互性，导致结果不自然。

为应对上述挑战，我们提出创新性驱动框架 AnyTalker。它建立在预训练视频扩散模型 Wan 之上，以极低的数据成本实现令人印象深刻的多人视频驱动，并明确强调交互性，如图 1 所示。核心想法是：用低成本单人数据支撑可扩展的多人驱动，仅用少量多人数据（少至 $12$ 小时）精炼交互。具体而言，AnyTalker 支持将可驱动身份（ID）扩展到任意数量，并保证各 ID 之间的交互。为容纳多流控制信号，我们设计了一种可扩展的 audio-to-face in-context 注意力机制，支持任意数量的 ID 与音频输入。训练按所用数据类型分为两阶段，单个样本中的说话人 ID 数目从一演变到多。第一阶段沿水平方向随机拼接单人说话视频，以模拟多人说话场景，使模型获得多人说话模式的基线能力。第二阶段用少量多人数据微调，以增强交互能力。

常用的单人 talking head 基准（HDTF、VFHQ、CelebV-HQ）缺少多人交互，因而不适于评估多人生成方法。InterActHuman 虽提供了相关基准，但其测试集聚焦单说话人，对交互分析帮助有限。为此，我们引入一套精细标注的基准：视频中两人同时参与说话与眼神接触，并细粒度标注说话与聆听区间，从而便于评估聆听状态下的交互。我们还首次提出一种新指标，通过测量聆听时段眼部关键点的活动来评估交互性。所提出的基准与指标将填补多人生成方法在交互性评估上的空白，惠及后续研究。

主要贡献总结如下：（1）提出一种可扩展的多流处理架构，使可驱动身份能够任意缩放。（2）提出一种新颖的两阶段训练管线：从单人数据学习多人说话模式，再用多人数据精炼跨身份交互。（3）首次提出定量评估多人交互性的指标，并配以专门基准数据集，以便全面评估。（4）全面实验表明 AnyTalker 达到先进性能，并在身份可扩展性、交互性、口型同步与数据成本之间取得有利平衡。

---

## 2. 相关工作

![图 2](../../../arxiv/audio_driven/anytalker/extracted/fig/method.png)

**图 2.** (a) AnyTalker 架构：新型多流音频处理层 Audio-Face Cross Attention，可处理多路人脸与音频输入。(b) 训练分两阶段：第一阶段用由单人数据拼接得到的多人数据与单人数据混合，学习准确口型；第二阶段用真实多人数据增强生成视频中的交互性。(c) Audio-Face Cross Attention 的实现细节：可递归调用的结构，用脸部掩码对输出做 masking。

### 2.1 音频驱动说话视频生成

文本到图像领域的近期关键进展（DDPM、LDM、CFG、ControlNet、DiT 等）显著催化了下游应用。早期工作如 EMO 将预训练扩散文本到图像模型扩展为端到端音频驱动虚拟人视频生成（Loopy、EchoMimic、Hallo、CyberHost 等）。它们通常集成时间注意力模块（AnimateDiff）、经 ReferenceNet 的身份控制，以及用预训练音频模型（Whisper、Wav2Vec2）做音频条件化，条件信号经级联注意力层注入。视频基座模型（Stable Video Diffusion、CogVideoX、HunyuanVideo、Wan）使端到端音频驱动虚拟人模型成为可能（EchoMimicV3、Hallo3、Sonic、StableAvatar、OmniHuman、InfiniteTalk、Wan-S2V 等）。这些工作大体沿用早期方法的音频注入实践，并在长视频生成、全身合成、非人类主体驱动、同步口型与连贯手部运动等方面报告突破。因而它们在质量与可控性上超越了早期基于 GAN 的生成方法（SadTalker、VASA、LivePortrait、MuseTalk）以及基于图像扩散的技术。然而，大多数方法仍针对单人场景；在多人场景中，它们往往让所有说话人同步做出相同动作或口型，多人交互受限。

### 2.2 多人视频生成

多人视频生成通过专门架构迅速发展，包括肖像视频生成、舞蹈视频生成与说话视频生成。在音频驱动 talking-head 这条轴上，Bind-Your-Avatar 引入细粒度 Embedding Router，把**谁**与**说什么**绑在一起。InterActHuman 训练掩码预测器以识别应激活的身体区域，从而实现定向控制。MultiTalk 提出 Label Rotary Position Embedding（L-RoPE）以解决音频–人物绑定。上述三种方法都依赖昂贵数据，从数百到数千小时的多人说话数据不等。另一些在单人数据上训练的模型通过专门控制器泛化到多人驱动：HunyuanVideo-Avatar 利用 Face-Aware Audio Adapter 有选择地在不同角色上激活注意力；Playmate2 在无分类器引导框架内使用 token 级掩码实现类似绑定。尽管这些方法能够产出多人结果，它们仍可能在不同角色之间产生破碎的交互，个体之间的交互性有限。

在本文中，AnyTalker 探索从单人数据学习多人说话模式的潜力，并设计可扩展的多流音频处理注意力架构，以较低训练数据成本在生成视频的交互性、身份可扩展性与口型同步之间取得有利平衡。

---

## 3. 方法

所提出 AnyTalker 的整体框架如图 2 所示。AnyTalker 继承 Wan I2V 模型的若干架构组件。为处理多流音频与身份输入，我们引入专门的多流处理结构，称为 *Audio-Face Cross Attention*（AFCA），将在第 3.2 节进一步说明。训练管线分为两阶段，在第 3.3 节概述。

![图 3](../../../arxiv/audio_driven/anytalker/extracted/fig/attn_mask_face_mask.png)

**图 3.** (a) 视频 token 到音频 token 的映射，由自定义注意力掩码实现。除第一个视频 token 外，每 $4$ 个音频 token 绑定到 $1$ 个视频 token。(b) Audio-Face Cross Attention 中用于输出 masking 的 mask token。

### 3.1 预备知识

作为基于 DiT 的模型，AnyTalker 通过对 3D VAE 特征 $f_{\mathrm{video}}$ 做 patchify 与展平来得到视频 token，文本特征 $f_{\mathrm{text}}$ 由 T5 编码器生成。此外，AnyTalker 包含 Reference Attention Layer，这是一种交叉注意力机制：用 CLIP 图像编码器 $E_{\mathrm{CLIP}}$ 从视频首帧提取特征 $f_{\mathrm{ref}}$。Wav2Vec2 也用于提取音频特征 $f_{\mathrm{audio}}$。整体输入特征 $f_{\mathrm{input}}$ 可写为

$$
f_{\mathrm{input}} = [f_{\mathrm{video}},\, f_{\mathrm{text}},\, f_{\mathrm{ref}},\, f_{\mathrm{audio}}]. \tag{1}
$$

与 Wan 模型一致，所有注意力层都接到最终输出的 FFN 层（图 2 中省略）。

### 3.2 Audio-Face Cross Attention

要做多人说话，模型必须能处理多流音频输入。一种可能方案是 MultiTalk 使用的 L-RoPE 技术：为不同音频特征分配独特标签与偏置。然而这些标签的取值范围需要预先明确指定，从而限制可扩展性。有鉴于此，我们设计一种更可扩展的结构，以可缩放的方式驱动多个 ID 并实现准确控制。如图 2(a) 与图 2(c) 所示，我们引入专门结构 Audio-Face Cross Attention（AFCA）。该结构可按输入的脸–音频对数量循环多次。如式 $(4)$ 与图 2(c) 所示，它能灵活处理多样的音频与身份输入，各次迭代的输出求和得到最终注意力输出。

**音频 Token 建模。** 我们用 Wav2Vec2 编码音频特征。第一个潜空间帧关注全部音频 token，而之后的每个潜空间帧只关注对应四个音频 token 的局部时间窗。视频流与音频流之间的这种结构化对齐，通过施加时间注意力掩码 $M_{\mathrm{temporal}}$ 实现，如图 3(a) 所示。此外，为使信息充分融合，AFCA 计算中使用的每个音频 token $f_{\mathrm{audio}}$ 会与由 $E_{\mathrm{CLIP}}$ 编码的脸部 token $f_{\mathrm{face}}$ 拼接。该拼接使所有视频查询 token $Q_{\mathrm{video}}$ 能够有效地关注不同的音频–脸信息对，计算如下：

$$
\begin{aligned}
K_{\mathrm{af}} &= \mathrm{Concat}(f_{\mathrm{audio}},\, f_{\mathrm{face}})\cdot W_{K}, \\
V_{\mathrm{af}} &= \mathrm{Concat}(f_{\mathrm{audio}},\, f_{\mathrm{face}})\cdot W_{V}, \\
\mathrm{Attn}_{\mathrm{out}} &= \mathrm{MHCA}(Q_{\mathrm{video}},\, K_{\mathrm{af}},\, V_{\mathrm{af}},\, M_{\mathrm{temporal}}).
\end{aligned} \tag{2}
$$

其中 MHCA 表示 *Multi-Head Cross Attention*，$W_{K}$ 与 $W_{V}$ 分别表示 key 矩阵与 value 矩阵。注意力输出 $\mathrm{Attn}_{\mathrm{out}}$ 随后将由脸部掩码 token 精炼，见式 $(3)$。

**脸部 Token 建模。** 训练时用 InsightFace 在所选视频片段的首帧上在线裁剪得到面部图像；面部掩码 $M_{\mathrm{face}}$ 则离线预计算，覆盖整段视频中脸部掩码的最大范围，即 *全局* 脸部边界框。该掩码保证面部运动不会超出此区域，从而避免在图 3(b) 所示的 reshape 与 flatten 之后错误地激活视频 token，尤其是在面部位移较大的视频中。该掩码与 $\mathrm{Attn}_{\mathrm{out}}$ 维度相同，可直接用于逐元素相乘，以计算 Audio-Face Cross Attention 的输出：

$$
\begin{aligned}
M_{\mathrm{token}} &= \mathrm{Patchify}(\mathrm{Flatten}(M_{\mathrm{face}})), \\
\mathrm{AFCA}_{\mathrm{out}} &= M_{\mathrm{token}} \odot \mathrm{Attn}_{\mathrm{out}}.
\end{aligned} \tag{3}
$$

因此，每个 I2V DiT 块的隐状态 $H_{i}$ 可写为

$$
H_{i}' = H_{i} + \mathrm{AFCA}_{\mathrm{out}}^{(1)} + \cdots + \mathrm{AFCA}_{\mathrm{out}}^{(n)}, \tag{4}
$$

其中 $i$ 表示注意力块的层索引，$n$ 表示 ID 数目。注意所有 $\mathrm{AFCA}_{\mathrm{out}}$ 项由 *同一* AFCA 层、共享参数产生。AFCA 计算对每个个体迭代执行 $n$ 次。该架构使可驱动 ID 的数量能够任意缩放。

![图 4](../../../arxiv/audio_driven/anytalker/extracted/fig/interactivity.png)

**图 4.** InteractiveEyes 中的两段视频，标注 $Motion$ 分数（像素）：左为原始视频，右为裁剪人脸与眼部关键点。听者转头看向说话人或抬眉会提高 $Motion$ 与 Interactivity；持续静止则两者都低。

### 3.3 训练策略

AnyTalker 探索单人数据学习多人说话模式的潜力，低成本单人数据构成训练数据的主体。

**单人数据预训练。** 我们同时使用标准单人数据，以及由水平拼接生成的合成双人数据。每个 batch 以均等的 $50\%$ 概率随机配置为双人或单人模式，如图 2(b) 所示。在双人模式下，batch 内每个样本与下一索引处的数据及其对应音频沿水平方向拼接。该做法使两种模式下每个数据 batch 的 batch size 保持相同。此外，当发生数据拼接时，我们预定义若干描述两人说话的通用文本提示。

尽管上述数据构造策略增强了模型在局部区域内定位音视频特征、学习双说话人说话模式的能力，完全省略单人数据并不可行。那样会显著损害模型生成准确口型的性能，导致驱动结果不稳定，我们将在表 4 中讨论。

**多人数据精炼。** 下一阶段，我们用少量真实多人数据精炼模型，以增强不同 ID 之间的交互性。尽管训练数据仅包含两个身份之间的交互，我们惊讶地发现：配备 AFCA 模块的模型能够自然泛化到超过两个 ID 的场景，如图 1 所示。我们推测这是因为 AFCA 机制使模型学到了人类交互的一般模式，不仅包括对音频的准确口型同步，也包括对其它 ID 说话动作的聆听与回应行为。

为构造高质量多人训练数据，我们构建了严格的质量控制管线：用 InsightFace 保证多数帧中有两张脸；用音频 diarization 分离音频并确保只有一或两个说话人；用光流过滤过度运动；用 Sync 分数将音频与人脸配对。该管线细节见补充材料。管线最终得到共计 $12$ 小时的高质量双人数据，相对先前方法（MultiTalk、InterActHuman、Playmate2）而言数量很少。由于 AnyTalker 的 AFCA 层在设计上天然支持多 ID 输入，双人数据以与第一阶段拼接数据相同的格式送入模型，无需额外处理。

总结而言，单人数据训练过程增强模型的口型同步能力与生成质量，同时学习一种泛化的多人说话模式；随后轻量的多人数据精炼弥补了单人数据无法完全覆盖的真实交互。

---

## 4. 交互性评估

尽管已有进展，主流单人 talking head 生成评测基准（HDTF、VFHQ、CelebV-HQ）仍不足以评估角色之间的自然交互。InterActHuman 虽引入了可比基准，但其测试集限于仅有一名说话人的场景，不利于评估多角色之间的交互。为填补这一空白，我们从网络收集了一组包含两个不同身份的视频，用于评测。

### 4.1 数据集构造

我们选择具有交互性的双人视频，构造名为 *InteractiveEyes* 的视频数据集。其中两段如图 4 所示。每段视频时长约 $10$ 秒，整段始终恰好展示两张脸。此外，通过细致的人工流程，我们对每段视频的音频进行切分，确保大多数视频捕捉到两人同时参与 *说话* 与 *聆听*，以及多样丰富的眼神交互场景，如图 5 所示。我们还确保每段视频包含互相注视与头部运动，以提供真实参考。

![图 5](../../../arxiv/audio_driven/anytalker/extracted/fig/time.png)

**图 5.** 每位说话人的聆听与说话时段。

### 4.2 所提出的交互性指标

除该数据集外，我们引入一种新指标——以眼部为中心的 *Interactivity*，用于评估说话人与听者之间的自然交互。由于眼神交互是对话情境中基本且自发的行为，我们将其作为交互性的关键指示。受 CyberHost 中 Hand Keypoint Variance（HKV）指标启发，我们通过跟踪眼部关键点的运动幅度来定量评估交互。

为此，我们在从生成帧中提取的、已对齐人脸的眼部关键点序列上定义 $Motion$，其中 $S$ 表示帧序列，$E$ 表示眼部关键点。$Motion$ 计算如下：

$$
\mathrm{Motion} = \frac{1}{|S|-1}\sum_{j=1}^{|S|-1}\left(\frac{1}{|E|}\sum_{i=1}^{|E|}\lvert E_{i,j+1}-E_{i,j}\rvert\right). \tag{5}
$$

此处 $i$ 与 $j$ 分别表示眼部关键点索引与帧索引，$E_{i,j}$ 表示各帧中的眼部关键点。该公式直观地计算眼部区域的位移与旋转。随后我们计算聆听时段内的运动。原因是：大多数生成方法在激活说话主体时表现良好，而聆听主体往往显得僵硬。因此在聆听时段评估更有针对性、也更有价值。每人聆听与说话时段的长度如图 5 所示，分别记为 $L_{1}$、$L_{2}$、$L_{3}$、$L_{4}$。为量化生成虚拟人的回应性，我们计算聆听阶段 $L_{2}$ 与 $L_{3}$ 的平均运动强度：

$$
\mathrm{Interactivity} = \frac{L_{2}\cdot\mathrm{Motion}_{L2}+L_{3}\cdot\mathrm{Motion}_{L3}}{L_{2}+L_{3}}. \tag{6}
$$

该指标有效地度量生成多角色视频中的交互性。如图 4 所示，所提出指标与人类感知一致：静止或迟缓的眼动得到低 $Motion$ 分数，转头与抬眉会提高分数，从而指示更高交互性。此外，为避免将异常眼动误判为高交互，我们实现了排除算法，细节见补充材料。

**表 1.** 与其它竞争方法在 HDTF 与 VFHQ 基准上的定量比较。此处 OmniHuman-1.5$^{*}$ 指经即梦（JiMeng）平台访问的 “Master Mode” 版本，目前不支持多人生成。**加粗**为最优，*斜体*为次优。

| 方法 | 多人 | 身体 | HDTF Sync-C $\uparrow$ | HDTF FID  | HDTF FVD  | HDTF ID $\uparrow$ | VFHQ Sync-C $\uparrow$ | VFHQ FID  | VFHQ FVD  | VFHQ ID $\uparrow$ |
| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AniPortrait | $\times$ | $\times$ | 3.44 | 18.74 | 241.84 | *0.94* | 2.63 | 28.54 | 269.24 | **0.95** |
| FantasyTalking | $\times$ | $\checkmark$ | 3.97 | 14.93 | 166.79 | 0.93 | 3.57 | 24.83 | 272.13 | *0.94* |
| StableAvatar | $\times$ | $\checkmark$ | 4.11 | 14.67 | 166.44 | 0.91 | 3.53 | 22.91 | 275.73 | 0.88 |
| EchoMimic | $\times$ | $\times$ | 5.23 | 61.53 | 381.55 | **0.95** | 4.87 | 58.72 | 486.75 | 0.86 |
| Hallo3 | $\times$ | $\times$ | 7.53 | 17.12 | 195.61 | 0.91 | 6.32 | 41.26 | 371.24 | 0.91 |
| Sonic | $\times$ | $\times$ | 7.81 | 52.96 | 286.12 | **0.95** | 7.71 | 36.68 | 385.37 | 0.89 |
| OmniHuman-1.5$^{*}$ | $\times$ | $\checkmark$ | 7.23 | 35.26 | 173.23 | 0.90 | 7.67 | 35.36 | 283.39 | 0.90 |
| MultiTalk | $\checkmark$ | $\checkmark$ | *8.91* | **13.54** | *162.58* | 0.93 | *7.77* | 24.25 | **243.66** | *0.94* |
| AnyTalker-1.3B | $\checkmark$ | $\checkmark$ | 6.85 | 14.47 | 218.01 | 0.91 | 5.81 | *21.88* | *267.08* | 0.91 |
| AnyTalker-14B | $\checkmark$ | $\checkmark$ | **9.05** | *13.84* | **160.87** | *0.94* | **7.79** | **20.99** | 290.73 | *0.94* |

**表 2.** 在多人基准 InteractiveEyes 上与其它竞争方法的定量比较。**加粗**为最优，*斜体*为次优。

| 方法 | Interactivity $\uparrow$ | Sync-C$^{*}$ $\uparrow$ | FVD  |
| --- | ---: | ---: | ---: |
| Ground Truth | 0.77 | 6.01 | 0 |
| Bind-Your-Avatar | 0.45 | 3.03 | 695.58 |
| MultiTalk | 0.49 | *6.88* | 500.03 |
| AnyTalker-1.3B | *0.97* | 4.56 | *467.84* |
| AnyTalker-14B | **1.01** | **6.99** | **424.15** |

---

## 5. 实验

**数据集。** 我们在单人数据集（HDTF、VFHQ、Hallo3、CelebV-HQ、VoxCeleb2）基础上补充互联网收集的数据，第一阶段训练约 $1000$ 小时；第二阶段收集双人对话片段，过滤后仅保留约 $12$ 小时。评测在两类基准上进行：（i）标准 talking-head 基准 HDTF 与 VFHQ；（ii）我们自收集的多人对话数据集（头肩与身体，两个身份都说话）。随后从每个基准各选 $20$ 段视频，严格确保其身份未出现在训练集中。

**实现细节。** 为全面评估方法，我们训练两种规模的模型：Wan2.1-1.3B-Inp 与 Wan2.1-I2V-14B，作为实验的基础视频扩散模型。所有阶段中，文本（T5）、音频（Wav2Vec2）、图像（CLIP）编码器以及 3D VAE 保持冻结。DiT 主网络（含新加入的 AFCA 层）全部参数开放训练。第一阶段预训练学习率为 $2 \times 10^{-5}$；第二阶段微调学习率为 $5 \times 10^{-6}$。所有模型用 AdamW 在 $32$ 块 NVIDIA H200 上优化。

**评测指标。** 对单人基准，我们采用若干常用指标：Fréchet Inception Distance（FID）与 Fréchet Video Distance（FVD）评估生成质量，Sync-C 度量音频与口型同步，ID 相似度在首帧与其余帧之间由 ArcFace 计算。

对多人基准，我们从不同维度评估。新引入的指标 *Interactivity* 作为主要评估指标。FVD 的计算与单人基准类似。对 Sync-C，我们将其精炼为 Sync-C$^{*}$，只关注每个角色说话时段的口型同步，从而避免长聆听片段对最终口型分数的影响，具体为

$$
\mathrm{Sync\text{-}C}^{*} = \frac{L_{1}\cdot\mathrm{Sync\text{-}C}_{L1}+L_{4}\cdot\mathrm{Sync\text{-}C}_{L4}}{L_{1}+L_{4}}. \tag{7}
$$

其中 $L_{1}$ 与 $L_{4}$ 表示图 5 中的说话阶段。

**比较方法。** 我们将 AnyTalker 与若干先进说话视频生成方法比较。单人生成方面，比较 AniPortrait、EchoMimic、Hallo3、Sonic、FantasyTalking、StableAvatar、OmniHuman-1.5 与 MultiTalk。多人生成方面，选择 Bind-Your-Avatar 与 MultiTalk 做定量与定性比较。

### 5.1 与 SOTA 方法的比较

**定量比较。** 首先，我们将 AnyTalker 与若干单人生成方法比较，以验证其单人驱动能力。定量结果见表 1。尽管并非专为驱动说话人脸而设计，AnyTalker 在所有指标上取得最佳或有竞争力的结果。此外，AnyTalker 的 $1.3$B 模型在口型同步上显著优于参数量相近的 AniPortrait、EchoMimic 与 StableAvatar。这些结果说明 AnyTalker 框架具有优秀而全面的驱动能力。

随后，我们用第 4 节所述多人数据集 InteractiveEyes 及相关指标，评估 AnyTalker 在驱动多个 ID 时同时保持准确口型同步与自然交互的能力。该比较中，我们将 AnyTalker 与已开源的多人驱动方法 MultiTalk 与 Bind-Your-Avatar 对照。表 2 结果表明，AnyTalker 的 $1.3$B 与 $14$B 模型均在 **Interactivity** 指标上取得最佳。此外，$14$B 模型在所有指标上均为最佳，从而验证所提出训练管线的有效性。我们进一步通过定量评估说明 AnyTalker 生成富含交互性视频的能力。

![图 6](../../../arxiv/audio_driven/anytalker/extracted/fig/qualitative.png)

**图 6.** 多种多人驱动方法的定性比较。在相同文本提示、参考图像与多路音频输入下，比较 Bind-Your-Avatar、MultiTalk 与 AnyTalker 的生成结果。左例使用 InteractiveEyes 数据集中的输入图像，右例使用文本到图像生成模型（Kolors）产出的图像。

**定性比较。** 我们随后从 InteractiveEyes 选取真实人物输入，并使用 AIGC 模型生成的输入，二者均配以相应文本提示与双路音频，用 Bind-Your-Avatar、MultiTalk 与 AnyTalker 做比较。如图 6 所示，AnyTalker 相对其它方法生成带有眼神与头部交互的更自然视频。MultiTalk 的眼神交互较弱，Bind-Your-Avatar 往往产生更静态的表情。这一趋势进一步验证了第 4.2 节所提出 Interactivity 指标的有效性。AnyTalker 不仅能生成自然的双人交互说话场景，还能良好缩放到多个 ID，如图 1 所示，它有效处理了四个 ID 之间的交互。单人基准 HDTF 与 VFHQ 上的定性结果见补充材料。

**表 3.** 在 HDTF 上用 $1.3$B 模型对 AnyTalker 组件的消融。“Baseline” 表示仅配备基本音频注意力层的模型。“AFCA” 表示加入 Audio-Face Cross Attention。“Single” 表示该阶段未使用真实多人数据。**加粗**为最优，*斜体*为次优。

| 设置 | Sync-C $\uparrow$ | FID  | FVD  | ID $\uparrow$ |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 5.42 | **13.01** | *170.96* | *0.90* |
| w/o AFCA | *6.71* | 14.97 | 207.47 | 0.88 |
| w/o Mask Token | 5.84 | 14.81 | 193.78 | 0.89 |
| w/o Concatenated Data | 6.21 | *14.73* | 202.01 | **0.91** |
| AnyTalker-1.3B (Single) | **6.97** | 15.58 | **166.27** | **0.91** |

### 5.2 消融实验

**组件。** 我们对图 2 中提到的三个重要组件做消融：Audio-Face Cross Attention、注意力输出的 Mask Token，以及拼接得到的多人数据。从仅使用单人数据的完整第一阶段 $1.3$B 模型出发，我们逐步移除相关组件，以评估它们对生成视频口型同步、生成质量与身份相似度的影响。表 3 表明每个组件都起显著作用，其中拼接多人数据对口型准确性最为有益。尽管初始阶段完成模型的 FID 略高于 Baseline，我们将其归因于 Baseline 倾向于生成面部表情或头部运动很少的说话视频。相反，完成模型生成更动态的动作，这在一定程度上会影响 FID 计算。我们将这一权衡视为模型设计的自然后果。此外，完成模型在其它评测指标上优于 Baseline。关键的是，第一阶段模型已具备驱动多个个体的基本能力，而这是 Baseline 所缺乏的。

**表 4.** 在 InteractiveEyes 上用 $1.3$B 模型做的消融。“RS” 表示第一阶段使用真实单人数据。“CM” 表示第一阶段使用拼接多人数据。“RM” 表示第二阶段使用真实多人数据。**加粗**为最优，*斜体*为次优。

| RS | CM | RM | Interactivity $\uparrow$ | Sync-C$^{*}$ $\uparrow$ | FVD  |
| :---: | :---: | :---: | ---: | ---: | ---: |
| $\times$ | $\checkmark$ | $\times$ | 0.55 | 3.21 | 672.18 |
| $\checkmark$ | $\times$ | $\times$ | 0.47 | 4.13 | 475.31 |
| $\checkmark$ | $\checkmark$ | $\times$ | 0.58 | **4.89** | **393.86** |
| $\checkmark$ | $\times$ | $\checkmark$ | *0.71* | 3.63 | 511.51 |
| $\checkmark$ | $\checkmark$ | $\checkmark$ | **0.97** | *4.56* | *467.84* |

**多人数据。** 随后我们聚焦多人数据，在 InteractiveEyes 基准上做更有针对性的消融。第一阶段用单人数据构造拼接多人数据。如表 4 所示，使用该拼接数据的模型在所有评测维度上优于不使用的模型，尤其在口型同步与交互性上，凸显拼接数据在学习多人说话模式中的作用。值得注意的是，表中第一行结果也说明第一阶段混合单人数据的必要性；否则生成结果将不稳定。

用真实多人数据微调之后，第一阶段使用拼接数据的模型表现更加优越，进一步证明它们已提早适应多人说话模式。尽管用真实多人数据微调会导致口型同步略有下降，它显著提升交互性，我们认为这是合理的权衡。

---

## 6. 结论

本文介绍 AnyTalker，一种用于生成多人说话视频的音频驱动框架。它提出可扩展的多流处理结构 Audio-Face Cross Attention，在实现身份缩放的同时保证无缝的跨身份交互。我们进一步提出一种可泛化的训练策略：通过基于拼接的增强，最大限度地利用单人数据学习多人说话模式。此外，我们提出首个交互性评测指标以及专门基准，以进行全面评估。大量实验表明，AnyTalker 在多人场景中平衡了口型同步、身份可扩展性与交互性。

---

## 补充材料

本补充材料提供关于 AnyTalker 的两组额外信息。

**A. 实验细节**

- 训练时使用的数据处理管线
- 推理数据的构造方案
- 训练超参数及其它技术细节
- 推理设置

**B. 扩展实验**

- 对 Interactivity 指标的更多分析
- 额外实验结果
- 交互性精炼的有效性

---

### A. 实验细节

#### A.1 训练数据处理

![图 A1](../../../arxiv/audio_driven/anytalker/extracted/fig/sync.png)

**图 A1.** 双人数据中的 SyncNet 分数矩阵。

**辅助模型。** 我们首先介绍数据处理阶段使用的辅助模型。为对从每段视频提取的音轨做句子切分与分块，我们采用预训练 speaker-diarization-3.1 模型，并结合 OpenAI 的 Whisper。人声分离与人声特征提取使用 Kim_Vocal_2 与 Wav2Vec2 的预训练模型。人脸检测与面部特征处理采用 RetinaFace 与 ArcFace，二者均可通过 InsightFace 访问。为度量音视频同步，我们采用 LatentSync 中的预训练 SyncNet 模型。文本提示由 Gemini 2.5 Pro 获得，其特征用 T5 编码器提取。最后，我们用 CoTracker 过滤相机运动过大的视频。

![图 A2](../../../arxiv/audio_driven/anytalker/extracted/fig/data.png)

**图 A2.** 训练数据属性。

**单人数据。** 有效的单人片段是 $24$ fps 视频：持续呈现一张可识别的脸，并伴有与口型完全同步的语音。我们首先用 CoTracker 剔除手持拍摄中相机抖动明显或场景突变的录像。随后用 RetinaFace 保证多数帧恰好包含一张脸。满足上述准则的视频由 speaker-diarization-3.1 得到粗粒度句子级切分。长于 $5$ 秒的话语再用 Whisper 进一步切成语义连贯单元，之后丢弃任何包含重叠说话人的片段。剩余干净的单说话人片段平均时长约 $2$ 秒。由于 $2$ 秒片段无法饱和 GPU 显存，我们随机拼接连续片段以延长每条序列；所得片段长度服从均值为 $4$ 秒、标准差为 $0.5$ 秒的高斯分布。随后用人声分离模型提取人声轨并用 Wav2Vec2 编码，音视频同步由 SyncNet 打分。同步分数低于预定义阈值的片段从训练集中移除。最后，每条保留片段被转写并标注帧级人脸边界框。我们再用 Gemini 2.5 Pro 为这些片段生成文本描述，并用 T5 编码器提取文本特征。所有输入文本统一截断或填充到恰好 $512$ 个 token。经上述管线处理后，得到约 $1000$ 小时高质量 $480$P 单人数据。

**双人数据。** 处理双人片段沿用单人管线，并有三处关键修改。

1. **人脸数量：** 视频必须在多数帧中包含 *恰好两张* 脸。
2. **说话人活动：** speaker-diarization-3.1 被约束为只输出两种有效状态：（i）两名说话人均在说话，或（ii）仅一名说话人在说话。
3. **空间一致性：** 两张脸的左右空间顺序在整段片段中必须保持不变；身份互换通过 InsightFace 检测并拒绝。

为建立正确的声–脸对应，我们计算 $2 \times 2$ 的 SyncNet 置信度矩阵（图 A1），并要求两个最大分数落在对角线上；否则丢弃该片段。再施加与单人设置相同的最低同步阈值。过滤后约保留 $12$ 小时干净双人数据。

**数据拼接。** 第一阶段训练时，我们水平拼接随机选取的单人片段。由于原始视频分辨率极高（许多为 2K/4K），若沿高度轴朴素缩放再堆叠，脸在画面中只会占极小一块。为避免这一点，我们采用特殊裁剪策略。首先在两个候选片段中定位脸中心。对 $480$P 素材，帧尺寸为 $(H,W)=(480,832)$，因此我们在每个脸中心周围扩展最小裁剪 $(480,416)$。随后做进一步增强：在裁剪保持于原图内部的前提下，保持 $480/416$ 宽高比随机扩大窗口。所得裁剪包含更大的面部区域，使模型能学习更准确的唇–音映射。

**数据属性。** 每个处理后的样本由字典式条目汇总，存储解码视频、清洗音频与预提取特征的绝对路径。训练时数据加载器通过该条目索引全部信息。单人与双人片段共享相同的键结构；列表中脸–语音条目的数量明确指示样本包含一名还是两名说话人。进一步细节见图 A2。

![图 A3](../../../arxiv/audio_driven/anytalker/extracted/fig/multi-data.png)

**图 A3.** InteractiveEyes 中的两个例子。两名说话人均有说话时段。

![图 A4](../../../arxiv/audio_driven/anytalker/extracted/fig/single-data.png)

**图 A4.** 基准数据集中的输入例子，从左到右：HDTF、VFHQ、EMTD。

#### A.2 基准数据处理

**单人数据。** 我们采用 HDTF 与 VFHQ 作为单说话人评测基准。二者是 talking-head 生成的事实标准；其图像紧紧裁在面部区域周围。包含手的半身序列上的额外实验见第 B.2 节。

对每段测试视频，我们提供（i）该主体的单张参考图像，以及（ii）对应语音片段。为尊重 GPU 显存约束，我们将音频长度固定为 $6$ 秒——所有竞争方法都能在不溢出的情况下处理这一时长。从每个数据集随机选取二十段片段，确保其中没有任何身份出现在 AnyTalker 训练集中。三个评测划分的参考图像见图 A4。

**双人数据。** 我们为正文第 4 节引入的双说话人测试集 *InteractiveEyes* 提供额外细节。每段视频用稳定相机拍摄，*每一帧恰好包含两张脸*；时长约 $10$ 秒。$80\%$ 的情形中两名说话人都产生语音，其余 $20\%$ 仅一名说话人说话，另一名保持聆听姿态。为避免切分错误，我们 *不* 应用 speaker-diarization-3.1；说话区间改为人工标注，以便准确计算正文提出的 Interactivity 指标。两个代表性例子见图 A3。

**表 A1.** EMTD 基准上的定量结果。**加粗**为最优，*斜体*为次优。

| 方法 | Sync-C $\uparrow$ | FID  | FVD  | ID $\uparrow$ |
| --- | ---: | ---: | ---: | ---: |
| FantasyTalking | 3.76 | 67.66 | 818.71 | 0.76 |
| EchoMimic v2 | 6.27 | 63.43 | *671.18* | 0.76 |
| MultiTalk | *8.38* | 64.71 | 787.99 | **0.79** |
| AnyTalker-1.3B | 5.83 | *56.01* | 789.59 | 0.74 |
| AnyTalker-14B | **8.45** | **50.61** | **664.58** | *0.77* |

#### A.3 实现细节

**训练细节。** 为全面评估方法，我们训练两种规模的模型：Wan2.1-1.3B-Inp 与 Wan2.1-I2V-14B，作为实验的基础视频扩散模型。所有阶段中，文本、音频、图像编码器以及 3D VAE 保持冻结、参数不变。DiT 主网络（含新加入的 AFCA 层）全部参数开放训练。第一阶段采用较高学习率 $2 \times 10^{-5}$ 做预训练，第二阶段采用较低学习率 $5 \times 10^{-6}$ 做微调，含 warm-up 策略，并用 AdamW 优化。$14$B 模型用 $32$ 块 NVIDIA H200 训练。第一阶段全局 batch size 设为 $32$，训练 $2.4$M 步。第二阶段 batch size 调整为 $16$，再训练 $50$K 步。$1.3$B 模型用 $8$ 块 NVIDIA H200 训练。全程全局 batch size 保持为 $48$，总训练步数与 $14$B 模型一致。

**推理细节。** 对所有竞争方法，我们严格遵循公开实现，并采用其默认推荐的推理超参数。需要文本输入的方法在单人基准上接收固定提示：`this person is talking`。多人基准的提示由 Gemini 2.5 Pro 按第 A.1 节自动生成。AnyTalker 用无分类器引导（CFG）推理，引导尺度为 $4.0$；在无条件分支中，文本与音频特征均置零。脸部掩码由 InsightFace 提取，并均匀膨胀以提供稍大的生成区域。

---

### B. 扩展实验

#### B.1 对 Interactivity 指标的更多分析

![图 B1](../../../arxiv/audio_driven/anytalker/extracted/fig/case.png)

**图 B1.** 会得到高 Interactivity 分数的四种典型情形。

**好例子。** 图 B1 可视化四种强烈指示对话参与的代表性听者行为：抬眉、点头、转头、视线转移。这些线索的出现对最终 Interactivity 分数贡献很大。由于 Interactivity 在整段视频上计算，我们不在图 B1 中报告帧级数值。如图 B3 所示，AnyTalker 生成富含这些交互动作的序列，因而获得相对较高的 Interactivity 评分。

**指标的稳健性。** 一些生成基线偶尔产生高度不合理的运动。例如，Bind-Your-Avatar 会生成夸张的**躺倒**动作，如图 B2 所示。若无对策，这类伪影会剧烈抬高 $Motion$ 分数，尽管它们与交互性无关。因此我们引入轻量异常抑制规则：若两连续帧之间的平均面部关键点位移超过 $10$ 像素（所有脸预先对齐到 $256 \times 256$ 画布），则冻结关键点位置，直到后续位移低于 $10$ 像素。如图 B2 右侧两帧所示，这一简单钳制阻止了绝大多数异常运动进入 Interactivity 计算。

![图 B2](../../../arxiv/audio_driven/anytalker/extracted/fig/bind.png)

**图 B2.** Bind-Your-Avatar 生成的一个失败例子。

![图 B3](../../../arxiv/audio_driven/anytalker/extracted/fig/more.png)

**图 B3.** AnyTalker 生成的更多结果。

#### B.2 额外实验结果

**EMTD 基准上的定量结果。** EMTD 是包含手的半身数据集，如图 A2 所示。我们将 AnyTalker 与三种能够生成半身序列的方法比较：EchoMimic v2、FantasyTalking 与 MultiTalk。评测协议严格遵循正文第 5.1 节对单人基准描述的指标。AnyTalker-14B 在三项指标上取得最佳，仅在身份保持（ID）上略落后于 MultiTalk。这些结果表明 AnyTalker 是一个全面的框架，不仅处理紧裁人脸输入，也能处理半身场景。

**单人基准上的定性结果。** 如图 B4 所示，我们给出 EchoMimic、StableAvatar、Sonic、MultiTalk、OmniHuman-1.5 与 AnyTalker-14B 在 HDTF 与 VFHQ 基准上的定性比较。AnyTalker 持续产生清晰齿列与准确口型运动。发音位置已用红色并加下划线标出。

![图 B4](../../../arxiv/audio_driven/anytalker/extracted/fig/qualitative_single.png)

**图 B4.** HDTF（左）与 VFHQ（右）基准上的定性结果。发音位置已用红色并加下划线标出。

**更多多人结果。** 图 B3 展示 AnyTalker 生成的额外多人动画。模型从容处理广泛输入：真实照片、AIGC 图像与卡通。无论涉及多少身份，它都能产生自然、符合情境的交互。更多有说服力的例子见项目主页上的视频。

#### B.3 交互性精炼的有效性

![图 B5](../../../arxiv/audio_driven/anytalker/extracted/fig/ab-inter.png)

**图 B5.** 在真实多人数据上微调之后（右），各身份之间的交互得到改善。

如图 B5 所示，用真实多人数据精炼交互性，使各身份交换显著更自然的眼神接触。仅在单人数据上训练的模型能够正确激活每张脸，但每个身份在不说话时都保持空白，这在对话场景中看起来极不自然。

#### B.4 未来工作

目前 AnyTalker 仅支持由文本提示驱动的初级相机运动。受近期可控视频生成方法启发，我们可以纳入额外条件信号，例如相机轨迹。通过将轻量、训练高效的模块（LoRA 等）嫁接到近期相机轨迹控制技术中，我们期望丰富生成视频的视觉叙事，在无需人工干预的情况下自动取景并跟踪当前说话人。
