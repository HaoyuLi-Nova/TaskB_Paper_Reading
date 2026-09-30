# Bind-Your-Avatar: Multi-Talking-Character Video Generation with Dynamic 3D-mask-based Embedding Router

- **Authors / Year / Venue**: Yubo Huang, Weiqiang Wang, Sirui Zhao, Tong Xu, Lin Liu, Enhong Chen / 2025 / ICCV 2025（arXiv:2506.19833）
- **Link**: [https://arxiv.org/abs/2506.19833](https://arxiv.org/abs/2506.19833)
- **Project**: [https://yubo-shankui.github.io/bind-your-avatar/](https://yubo-shankui.github.io/bind-your-avatar/)
- **One-line summary（一句话概括）**: 在 CogVideoX MM-DiT 上，用去噪过程中预测的 3D 人物掩码把谁和哪路音频乘成音频–视觉路由，做同场景多人说话视频；不改音色。

## 1. Motivation（动机）

- 动机：
  - 音频驱动说话人生成已经能把单人肖像做成口型同步的 talking head，但主流设定仍是：**一张参考图 + 一路语音 + 往往还要一张作为第一帧的 inpainting 图**。
  - 若要两人对话，常见偷懒是各生成一条单人视频再左右拼接。画面不共享空间，没有共同背景、站位、遮挡和近距离互动，不是同场景对话。
  - 作者把真正想做的任务定成 **Multi-Talking-Character Video Generation**：多张参考脸、多路语音、文本描述环境/站位/动作，生成**同一空间环境里同时在场**的多人说话视频；并希望支持 **重叠说话**，以及 **可以没有 inpainting 首帧**（用文本构图）。
  - 叙事上拆成两句：要把谁（身份）和说什么（哪路音频）绑在一起；社区缺少同场景多人说话的数据和流水线。

- 是否成立 / 是否有异味（理由）：
  - **痛点成立。** 拼接左右不是同场景；静态首帧框在走动、换位、两人靠近时会把音频灌进错的脸。这和 MultiTalk / AnyTalker / HarmonizedVA 面对的是同一类绑定问题，不是人造题目。
  - **有两处夸大。** （1）largely unaddressed：同期已有 MultiTalk（L-RoPE）、HunyuanAvatar，后还有 AnyTalker、InterActHuman；本文实验表里却拿 Sonic / Hallo3 的拼接版和 **不吃音频** 的 Ingredients 打，绑定方法之间的正面对比弱。（2）first dedicated dataset是 ICCV 时间线上的抢位，MTCC 有价值，但不宜当成问题本身不存在的证据。
  - **范围要看清：** 动机停在口型/表情/头姿跟对的那路声，**不改音色**。对 HarmonizedVA声要像这个人是另一条问题，本文不回答。

- **结论：** 动机站得住，针对的是同场景绑定而非拼接；无人做 / 首个数据的措辞偏满，实验对照也没有把同期绑定方法摆上台面。

## 2. Problem（问题）

- 核心问题：
  给定
  - 文本 $\mathbf{x}$（环境、站位、动作）；
  - $n$ 路语音 $\mathbf{a}=\{\mathbf{a}_1,\ldots,\mathbf{a}_n\}$；
  - $n$ 张参考脸 $\mathbf{I}_r$；
  - 可选 inpainting 帧 $\mathbf{I}_i$；
  - 可选音频–人物矩阵 $\mathbf{A}^{\mathrm{ac}}\in\mathbb{R}^{n\times n}$，
  生成 $\mathbf{V}\in\mathbb{R}^{T\times 3\times H\times W}$，使得第 $i$ 个人的口型/表情/头动只由 $\mathbf{a}_i$ 驱动，且身份跟 $\mathbf{I}_{r,i}$ 走。
  - 操作上要同时解决三件事：
    1. **对应：** 同画面里哪路音频打到哪个视觉 token（含重叠说话）；
    2. **动态空间：** 人会动、会靠近，首帧 2D 框复制全视频不够；
    3. **构图：** 多人不能把参考图简单拼成第一帧，需要能无 inpainting、靠文本出场景。
  - 本文实验固定 $n=2$。

- 普适性 & 重要性（理由）：
  - 问题在多路音频 → 一条视频这条线上是真问题， intervies / 对话生成都要。重要性主要在**空间路由**：对时他们用 Wav2Vec 卷积压到视频 $T'$，和 MultiTalk 的按帧打包同类，不是新的时钟问题。
  - 子问题分化清楚：Pre-Denoise 失败 = 静态框；Post-Denoise 能绑但贵；Intra-Denoise 要在去噪中途、画面还很噪时预测出能用的 3D mask。这和 InterActHuman没有成片就没有区域、没有区域就不能注音频是同一循环依赖。
  - 本文**不**包含：联合生成音频、音色与外貌和谐、可变人数集合、人手/身体交互（作者自己在 limitation 里承认）。

- **结论：** 问题真实、对视频侧多人绑定重要；定义成同场景 + 可重叠 + 可无首帧是清晰的；不要把它读成音视频联合生成或音色问题。

## 3. Method & Core Novelty（方法与创新）

### 3.1 骨干与条件怎么进

- 方法概述：
  - 底座 **CogVideoX MM-DiT**（不是 Wan）。视频 $V$ 经 3D VAE 得 $\mathbf{z}_0\in\mathbb{R}^{T'\times C'\times H'\times W'}$。可选 inpainting、多张参考图在 RGB 上拼成 $\mathbf{I}_R$，同样走 VAE，时间维 pad 到 $T'$，再与 $\mathbf{z}_0$ **沿通道拼成 $3C'$**（I2V 式额外通道）。Patch 成视觉 token $\mathbf{v}\in\mathbb{R}^{S\times d}$，$S=T'\cdot (H'/\tau)\cdot (W'/\tau)$，与 T5 文本嵌入一起进 $L$ 层 DiT，预测噪声，扩散损失 $\mathcal{L}_d$。
  - **文本：** T5。
  - **音频：** Wav2Vec → 投影 → **2D 卷积下采样**，时间分辨率对齐到 $T'$，得到 $\mathbf{e}_a$。对时在进入 DiT 之前就做完，后面 router 只管空间（以及每帧更新）。
  - **脸：** CLIP 全局 + InsightFace 局部 → Q-Former → $\mathbf{e}_c$（跟 ConsisID）。
  - 编码前：inpainting 的脸加噪、参考图去背景，避免捷径抄背景。
  - 真正打到谁身上靠 mask-guided cross-attn，见下。

### 3.2 Embedding Router：把 who 和 what 拆成两个矩阵再乘

- 预测时空软掩码 $\mathbf{M}\in[0,1]^{n\times T'\times H'/\tau\times W'/\tau}$，展平为 **人物–视觉** $\mathbf{A}^{\mathrm{cv}}\in\mathbb{R}^{n\times S}$。背景当第 $n+1$ 类，减轻背景乱分类。
- **音频–人物** $\mathbf{A}^{\mathrm{ac}}\in\mathbb{R}^{n\times n}$：可当输入，或用现成 speech–face 对齐模型打分再投票。单位阵 = 音频 $i$ 对人 $i$；对调两路音频只改这个矩阵。
- **音频–视觉路由：**
  $$
  \mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\,\mathbf{A}^{\mathrm{cv}}.
  $$
  这是全文最干净的一句话：身份绑定和音频绑定解耦，重叠说话 = 两路 $\mathbf{e}_a$ 同时在、乘到不同空间支撑集上。
- 注入分两步（先脸后声）：
  $$
  \mathbf{v}'=\mathbf{v}+\mathbf{A}^{\mathrm{cv}}\cdot\mathrm{FaceCrossAttn}(\mathbf{e}_c,\mathbf{v}),
  $$
  $$
  \mathbf{v}''=\mathbf{v}+\mathbf{A}^{\mathrm{av}}\cdot\mathrm{AudioCrossAttn}(\mathbf{e}_a,\mathbf{v}').
  $$
  人脸 embedding 只加到是这个人的 token；音频只加到这路音频对应的人的 token。
- 推理时音频 mask 再 **inflate**：翻转数值后沿人物维对调。两人时人1的音频支撑集 $\approx 1-M_2$（人1+背景），方便头肩跟着动，同时尽量不灌进另一张脸。低置信丢掉，再半监督聚类做时空光滑。

### 3.3 三种 Router（方法进化，不是三个并列产品）

| 名称 | 掩码从哪来 | 失败 / 代价 |
| --- | --- | --- |
| **Pre-Denoise** | 首帧检脸，2D 框复制到所有 $t$ | 人一动就偏；框太粗，靠近会重叠。即静态空间绑定 |
| **Post-Denoise** | 先不注入脸/声去噪出无声粗视频 → 每帧检脸得 3D mask → 再去噪一遍 | 绑定准，推理两轮，贵 |
| **Intra-Denoise（采用）** | 去噪中途用当前层视觉 token + 脸 embedding 预测 3D mask，一轮前向 | 要训 router；早期噪声大，靠损失和 teacher-forcing |

Ablation 只定性：近距离时稠密 mask 优于 bbox；开车等动态场景 3D 优于首帧 2D。这正好打在 Pre-Denoise 的两个失败模式上。

### 3.4 Intra-Denoise 网络与损失

- 结构（附录）：复用 **已预训练的 face cross-attn** 的 Q/K（视觉 token ↔ 脸 embedding 的对应），线性变换后算权重，加 **3D RoPE**，再经若干时空注意力，最后线性 + softmax 出 mask。不是从零学分割，是在脸已经能对上视觉的注意力上长出一张时空图。
- 监督：SAM2 每人稠密 mask，降采样到 token 网格当 GT。
  - $\mathcal{L}_r$：逐层、逐人、逐时空格的交叉熵（背景一类）；
  - $\mathcal{L}_{\mathrm{st}}$：掩码的时空梯度 L1，要光滑；
  - $\mathcal{L}_{\mathrm{layer}}$：同一格各层 mask 的方差，层间一致。
  $$
  \mathcal{L}_{\mathrm{router}}=\mathcal{L}_r+\lambda_{\mathrm{st}}\mathcal{L}_{\mathrm{st}}+\lambda_l\mathcal{L}_{\mathrm{layer}},
  $$
  权重 $1 / 0.001 / 8$。作者认为 $\mathcal{L}_r$ 已逼同一格只能一个人（Identity Exclusivity）。相对 Ingredients：Ingredients 也做 token–人物相关性，但无几何先验，mask 碎、时间上不稳。

### 3.5 训练：三阶段 + teacher-forcing

1. **身份 / inpainting：** 无音频；解冻 transformer 与脸编码器；50% 丢 inpainting，否则无首帧不会生成；Dynamic Mask Loss 50%。
2. **加上音频：** DiT 上 LoRA；只训音频/脸编码器、两个 cross-attn、LoRA；条件随机 drop 做 CFG。
3. **加上 router：** 只训 router、两个 cross-attn、LoRA。单人数据把条件复制 $n$ 份冒充多人；多人来自 VICO（拼接）、Friends-MMC、自建 MTCC。步数 1万 / 4万 / 1万，batch 16，lr $10^{-5}$。

- **Teacher-forcing（关键实现细节，也是隐患）：** 联合训 router 和去噪容易塌成token 很好分类但没有视觉内容，且早期 mask 错会让条件注不进去、扩散越训越废。因此 **mask-guided attn 训练时用 SAM2 真值**，router 图从去噪路上 **detach**；对强制 mask 做 dropout + 高斯噪声当增强。推理才用预测 mask。
  - 含义：DiT 学的是GT 路由下如何用脸和声；router 是另训的分割头。测试时两者接上，存在 **train/test 路由分布差**。

### 3.6 数据 MTCC

- 约 200h：脱口秀、访谈、新闻、网课、剧集。清洗：分辨率/时长/fps → 正好两人脸 → 排除站太远 → 分轨后再算 Sync-C（重叠说话不能直接对混合音做 Sync）。
- 处理：SAM2 出每人 mask（router GT）；MossFormer2 TSE 分轨并得到 $\mathbf{A}^{\mathrm{ac}}$；Wav2Vec；Qwen2-VL 出 caption。
- Benchmark：40 对照片（含奇幻/机器人等），CosyVoice2 合成双路语音；20 对有 inpainting、20 对没有。

### 3.7 核心创新点（vs 已有工作的 delta）

- **真增量：**
  1. 把绑定写成 $\mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\mathbf{A}^{\mathrm{cv}}$，身份与音频条件解耦，重叠说话有结构上的位置；
  2. **去噪中**预测逐帧稠密 3D mask（Intra-Denoise），相对首帧框 / 静态最大脸框是明确的动态绑定；
  3. 几何先验损失 + 复用 face cross-attn Q/K，而不是孤立地给每个 token 打分类分（对照 Ingredients）。
- **不是新故事的部分：** 通道拼接参考/inpainting、CLIP+InsightFace+Q-Former、Wav2Vec 再对齐到 $T'$、LoRA 插音频，都是 talking-head / ConsisID 常规组合。
- **和 MultiTalk 的差：** MultiTalk 对时靠 reshape，绑定靠 amap→L-RoPE（软身份轴）；本文对时靠卷积对齐 $T'$，绑定靠可监督掩码挡 cross-attn。一个在旋转里做人，一个在注意力支撑集上做人。本文 related work 把 MultiTalk 写成仍依赖 inpainting + 静态 mask，与 MultiTalk 实际的逐层 amap 不完全相符——定位上有意画远。
- 方法能否支撑动机：同场景、可无首帧、重叠说话，结构上对得上。人手/身体互动、实时性，作者承认撑不住。音色问题不在目标里。

- **结论：** 创新在动态 3D mask 路由和矩阵分解，不在底座也不在对时；teacher-forcing 让训练能收敛，也把预测 mask 是否真能驱动去噪留到了推理。

## 4. Related Work（相关工作）

- 关键相关工作：
  - 单人扩散 talking head：Hallo3、Sonic、FantasyTalking、MoCha 等——依赖 inpainting、单人。
  - 同期多人：MultiTalk、HunyuanAvatar——作者称仍要 inpainting、静态 mask。
  - 多概念视频：Ingredients（token–人物相关性、无音频）、MAGREF、DanceTogether、Concat-ID。
- 本文定位与区别：在 MM-DiT 上加音频，并用几何约束的 3D mask 做动态路由；强调无 inpainting 和重叠说话。
- 缺失 / 错位：
  - 实验未报 MultiTalk / HunyuanAvatar 数字（表里是 Sonic、Hallo3 concat、Ingredients）。
  - InterActHuman（掩码预测器 + 逐步局部音频）几乎同一循环依赖，arxiv 同月量级，正文未对话。
  - AnyTalker 更晚，不必苛责未引；对照时静态最大脸框 vs 3D mask 才是该比的。
  - 未讨论联合生成（Ovi 等），对本文任务不是必须，但对 HarmonizedVA 读者要自己补。

- **结论：** 方法定位清楚，related work 对静态 mask的划界偏硬，实验对照偏软。

## 5. Citation / Seed Potential（高引潜力）

- 判断：中等。作为同场景多人 talking + 动态 mask 路由会被引用，很难单独定义整条线。
- 依据：
  - 问题会被 MultiTalk / AnyTalker / InterActHuman 一起引用，份额被分掉；
  - $\mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\mathbf{A}^{\mathrm{cv}}$ 和 Intra-Denoise 可迁移到别的 DiT；
  - MTCC / 处理流水线若真开源，引用会比方法本身稳；
  - $n=2$、CogVideoX 全注意力贵、不改音色、实验没打强绑定 baseline，限制天花板。

- **结论：** 路由写法值得当对照，不是必须跟的底座。

## 6. Future Work & Improvements（改进空间）

- 局限性（作者 + 额外）：
  - 缺显式人体运动，手势和互动弱；多人交互数据少；
  - MM-DiT 全注意力推理贵；
  - teacher-forcing：训练见 GT mask，推理见预测 mask；
  - $n=2$；音频–人物矩阵依赖外部 speech–face 模型；
  - ablation 几乎只有定性图；
  - 不生成、不改音频。
- 可做的未来工作 / 研究机会：
  - **跟进性小改：** 把 Intra-Denoise 接到 Wan/Ovi；窗内 inflate 策略；router 与 DiT 联合微调、去掉 detach。
  - **值得自己做：** 动态 mask / 实体槽接到**联合音视频生成**（本文只有视频轨）；重叠说话时音频 token 级而不是整路级路由；换位 + 音色绑定（本文 mask 跟人走，但不碰声音身份）。InterActHuman 的前若干步不用 mask可直接对比 teacher-forcing。

- **结论：** 对 HarmonizedVA，该抄的是去噪中更新的 3D 归属和$\mathbf{A}^{\mathrm{ac}}$ 与 $\mathbf{A}^{\mathrm{cv}}$ 分离，不是 CogVideoX 流水线本身。

## Overall Verdict（总评）

- 是否值得深读：**值得当多人绑定对照精读**，尤其 Router 三档和矩阵乘法；不必当联合生成或音色论文读。
- 最大亮点 / 最大隐患：亮点是 Intra-Denoise 3D mask + $\mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\mathbf{A}^{\mathrm{cv}}$，把动态谁在哪和哪路声拆开。隐患是训练用 GT 路由、实验避开同期最强绑定方法、问题停在视频侧口型对齐。
