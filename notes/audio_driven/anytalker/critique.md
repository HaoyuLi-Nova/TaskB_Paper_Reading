# AnyTalker: Scaling Multi-Person Talking Video Generation with Interactivity Refinement

- **Authors / Year / Venue**: Zhizhou Zhong, Yicheng Ji, Zhe Kong, Yiying Liu, Jiarui Wang, Jiasun Feng, Lupeng Liu, Xiangyi Wang, Yanjia Li, Yuqing She, Ying Qin, Huan Li, Shuiyang Mao, Wei Liu, Wenhan Luo / 2025 / arXiv:2511.23475
- **Link**: [https://arxiv.org/abs/2511.23475](https://arxiv.org/abs/2511.23475) · 项目页 [https://hkust-c4g.github.io/AnyTalker-homepage](https://hkust-c4g.github.io/AnyTalker-homepage)
- **原文 TeX**: `arxiv/audio_driven/anytalker/extracted/main.tex`
- **方法总览 / 问答 / AFCA 形状**: [`note.md`](note.md) · [`QA.md`](QA.md) · [`afca.md`](afca.md)
- **One-line summary（一句话概括）**: 在 Wan I2V 上用共享参数的音–脸交叉注意力循环处理身份–音频对，空间上乘全视频最大脸框；单人水平拼接预训练后再用约 $12$ 小时真双人精炼听者反应。不改音色。

## 1. Motivation（动机）

- 动机：
  - 单人音频驱动已经能出同步口型，但播客 / 直播 / 对话要的是**同画面多人**，还要听者有反应（转头、眼神），不是所有人一起张嘴。
  - 直接训多人贵：轮流、角色切换、眼神难标。作者把贵的部分压缩成少量真双人精炼，把规模部分交给单人数据。
  - 第二条线是可扩展性：MultiTalk 的 L-RoPE 要预先规定身份标签范围，人一多标签轴就要重定义。作者要推理时 $n$ 任意增加。

- 是否成立 / 是否有异味（理由）：
  - **痛点成立。** 多人数据标注贵、听者僵硬、身份轴不好扩，都是社区在讨论的真问题，不是人造题目。这和 MultiTalk / Bind-Your-Avatar / InterActHuman / HarmonizedVA 面对的是同一类**空间绑定 + 数据成本**问题。
  - **异味 1（把交互说成新任务，评测却很窄）。** 引言写首次提出交互性指标。CyberHost 已经用关键点方差评手；本文只是把对象换成眼、窗口换成听者时段。新在切片，不在统计。InteractiveEyes 还要求稳定相机、每帧恰好两张脸、左右顺序不变——把换位、遮挡、走动这些**真正难的交互**在建集时就滤掉了。
  - **异味 2（数据成本对比不公平）。** 只需 $12$ 小时多人成立的前提是前面还有 $\approx 1000$ 小时单人 + $32\times$ H200、$2.4$M step。相对 MultiTalk数百到数千小时多人是真省了多人标注，但不是低成本训练。
  - **范围要看清：** 动机停在口型 / 听者眼动，**不改音色**。对 HarmonizedVA声要像这个人本文不回答。

- **结论：** 动机站得住，针对的是数据成本与身份可扩展；首次交互评测 / 任意 ID的措辞偏满，评测分布比叙事窄。

## 2. Problem（问题）

- 核心问题：
  给定首帧（可含多人）、$n$ 路语音、$n$ 张脸（或从首帧裁出）和文本，生成一条视频，使第 $k$ 个人的口型跟第 $k$ 路音频，听者在对方说话时有可见反应，且 $n$ 不必在训练时钉死。操作上要同时解决：
  1. **对应：** 哪路音频打到哪些视觉 token；
  2. **数据：** 如何从单人语料迁到多人说话模式；
  3. **交互：** 听者不能像贴纸。
- 普适性 & 重要性（理由）：
  - 问题在多路音频 $\to$ 一条视频这条线上是真问题，访谈、对话生成都要。重要性主要在**空间路由和数据配方**，不是新的对时时钟（$4$ 音频 token / 视频格是 Wan 常规）。
  - 本文**不**包含：联合生成音频、音色与外貌和谐、动态换位、重叠说话的结构隔离、可变人数的**定量**评测（四人只在 teaser）。
  - 训练集过滤左右顺序不变、多数帧恰好两张脸，等于把问题定义成了**位置几乎不动的双人对话**。换位失败是问题被定义掉了，不是方法意外修好。

- **结论：** 问题真实、对视频侧多人绑定重要；定义成可扩展多流 + 低多人数据是清晰的；不要把它读成动态路由或音色论文。

## 3. Method & Core Novelty（方法与创新）

- 方法概述：
  - 底座 Wan I2V；冻 T5 / Wav2Vec2 / CLIP / 3D VAE；训 DiT + 新加的 AFCA。
  - AFCA：对每对 $(\mathrm{face},\,\mathrm{audio})$ 做一次 cross-attn，输出乘该人静态 $M_{\mathrm{token}}$，再对 $k$ 求和。时间上 $M_{\mathrm{temporal}}$ 做 $4$-to-$1$ 对齐。
  - Stage 1：$50\%$ 单人 + $50\%$ 水平拼接伪双人。Stage 2：$12$ 小时真双人、更低学习率。
  - 评测：HDTF / VFHQ / EMTD + 自建 InteractiveEyes；主指标是听者时段眼动 $Motion$。

- 核心创新点（vs 已有工作的 delta）：
  - **真增量 1：共享参数循环 $n$ 对。** 相对 L-RoPE预定义 label 集合，这是更干净的可变人数接口。代码 `WanAF2VCrossAttention` 就是一个 `for i in range(n)`。
  - **真增量 2：单人拼接配方有消融。** 表 4 把 RS / CM / RM 拆开：拼接管口型，真双人管交互，缺一不可。这比我们用了合成数据有内容。
  - **换皮部分：静态最大框。** 和原 HarmonizedVA、Bind 的 Pre-Denoise 路由器是同一失败模式。论文把它写成防止面部位移激活错 token——只解释了框内晃头，没解释人走出框怎么办。InterActHuman 的 Fixed mask 消融已经定量过这条路：口型还行、FVD 崩。
  - **代码与公式不完全同构。** 论文 $\mathrm{Concat}(f_{\mathrm{audio}},f_{\mathrm{face}})\cdot W_{K}$；实现是两套 Linear 后再拼 KV。精读以代码为准。
  - 方法能支撑少用多人数据学说话；撑不住任意交互 / 换位 / 重叠说话。teaser 的四人和卡通是展示，不是证明。

- **结论：** 架构增量在循环 AFCA 和拼接配方；空间绑定是静态框，和任意 ID、自然交互的标题不匹配。

## 4. Related Work（相关工作）

- 关键相关工作：
  - 单人驱动：EMO、FantasyTalking、OmniHuman、Hallo3、Sonic、StableAvatar、Wan-S2V。
  - 多人绑定：MultiTalk（L-RoPE）、Bind-Your-Avatar（3D mask router）、InterActHuman（跨步 mask cache）。
  - 单人数据泛化到多人：HunyuanVideo-Avatar（Face-Aware Audio Adapter）、Playmate2（token 级 CFG 掩码）。
- 本文定位与区别：
  - 相对三条绑定主线，AnyTalker 站在最省多人数据、最静态的空间门。定位本身清楚。
  - 写 InterActHuman测试集聚焦单说话人是事实（其协议一人 meaningful、一人 mute），但把对方说成不利于评估交互有点过：InterActHuman 的问题是局部音频，不是听者眼动。两边评的不是同一个交互。
- 缺失的重要引用（如有）：
  - 同期 FantasyTalking 已在单人表里，但讨论身体 / 半身时对 OmniAvatar 着墨少。
  - 没有把 Bind 的 Pre / Intra / Post-Denoise 三种路由器当作自己静态框的对照光谱来写——读者需要自己把 AnyTalker 对上 Pre-Denoise。
  - HarmonizedVA 关心的音色线完全不在 related work 里（可以理解，问题不同，但引用时不要误读成音画和谐）。

- **结论：** 直接前驱引用齐；和 Bind / InterActHuman 的对照停在他们数据贵，没有在路由动态性上正面打。

## 5. Citation / Seed Potential（高引潜力）

- 判断：
  - **中等偏上的工具论文潜力，难成 seed。** InteractiveEyes + 听者眼动会被后人当评测引用；拼接伪多人会被数据配方引用；AFCA 循环会被可变人数实现引用。但路由思想（静态框）会被 Bind / InterActHuman 盖过。
- 依据：
  - **Timing 好：** 2025 末，Wan 生态多人驱动窗口正开，开源 1.3B / 14B 权重 + Gradio，复现门槛低于 Bind 的 CogVideoX + SAM2。
  - **问题普适，方法可迁移性中等：** 拼接配方和循环 CA 可搬到别的 DiT；静态框不能当通用绑定解。
  - **评测隐患会限制引用质量：** Interactivity 高于 GT（$1.01$ vs $0.77$）若不被社区校准，后来者会刷一个动得更厉害的指标。
  - 复现成本：推理开源；训练是 $32\times$ H200、$2.4$M step，配方可学、权重可下，从头复现贵。

- **结论：** 会作为Wan 上省数据的多人驱动被引，不太可能定义绑定问题的标准解。

## 6. Future Work & Improvements（改进空间）

- 局限性：
  - **静态框：** 走动、换位、近距离重叠时灌错脸。作者自己的数据过滤把这些难例拿掉了，limitation 里却去谈相机控制，避重就轻。
  - **指标未校准：** 生成视频 Interactivity 高于真值，没有用户研究证明分高 = 更自然，只有看起来一致的定性。钳制规则（$10$ px）是事后补丁，阈值没有灵敏度分析。
  - **测试集太小：** 每基准 $20$ 条；$10$ 秒、$6$ 秒音频。方差和显著性都没报。
  - **1.3B vs 14B 口型鸿沟大：** 多人表上 1.3B Sync-C$^{*}$ $4.56$ 对 MultiTalk $6.88$，交互高、嘴不准。把两个规模捆在SOTA里会误导。
  - **重叠说话：** 训练 diarization 丢弃 overlap；推理要预分轨。框重叠时两路残差直接加。
  - **不改音色。**
- 可做的未来工作 / 研究机会：
  - **跟进性小改：** 把最大框换成 Bind 式 Intra-Denoise 3D mask，或 InterActHuman 式跨步 cache；给 Interactivity 做人评校准和对调两路音频的绑定测试。
  - **值得自己做：** 静态框失败已经被本文和原 HarmonizedVA 共同证实——下一步不是再做一个更巧的框，而是**动态实体槽 + 独立音频潜变量**。拼接配方可以当 Stage 0 省数据预训练，不要当最终绑定。相机轨迹（作者 future work）对 HarmonizedVA 是旁路。

- **结论：** 最大技术债是空间路由停在 Pre-Denoise；最大评测债是 Interactivity $>$ GT。这两处都是可发的后续，不是边角。

## Overall Verdict（总评）

- 是否值得深读：
  - **值得，当省数据的多人预训练 + 循环 CA 接口读，不要当绑定问题的终点。** 公式短、开源全、和 MultiTalk / Bind / InterActHuman 对照清楚。
- 最大亮点 / 最大隐患：
  - 亮点：拼接伪双人的消融和共享参数循环 $n$ 对，是 presently 最便宜的可变人数实现。
  - 隐患：静态最大框在换位上结构失败；Interactivity 高于 GT 却被当成 SOTA 主指标。
