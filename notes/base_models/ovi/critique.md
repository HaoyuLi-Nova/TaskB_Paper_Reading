# Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation

- **Authors / Year / Venue**: Chetwin Low, Weimin Wang (Character AI); Calder Katyal (Yale) / 2025 / arXiv 2510.01284（ICLR 2026 投稿稿；文内日期 2025-09-29）
- **Link**: https://arxiv.org/abs/2510.01284 · 项目页 https://aaxwaz.github.io/Ovi · 原文 TeX `arxiv/base_models/ovi/extracted/main.tex` · 译本 `translations/base_models/ovi/paper_zh.md`
- **One-line summary（一句话概括）**: 把 Wan2.2-5B 的 DiT 镜像成音频塔、从头训语音+音效，再在每一层做双向 cross-attention + 缩放 RoPE，做成开源侧可复用的**一阶段联合音视频生成**工程模板；概念增量有限，数据与训练配方才是真正的壁垒。

## 1. Motivation（动机）

- **动机**：电影式内容要求语音口型、音效与画面**同时被生成、互相决定**，而开源主流是**固定一路、生成另一路**再事后对齐（T2V→V2A，或 A2V talking-head）。闭源 Veo 3 能联合生成，但方法不透明，所以要做一个真正 one-pass 的开源统一范式。
- **是否成立 / 是否有异味（理由）**：
  - **成立的部分**：事后对齐无法让口型与语音互相约束，这是真痛点；Veo 3 之后社区确实缺一个可复现的联合生成配方。把**说话头 + 人脸 mask**当成 A2V 的能力上限，用来反衬联合生成的必要性，逻辑也通。
  - **异味 1（夸大空白）**：引言写 “truly unified one-pass … remains largely unexplored”，但 Related Work 自己就引了 UniVerse-1（Wan + ACE-Step 拼接）和 JavisDiT（同骨干 + prior estimator）。空白不是**没人做联合生成**，而是**开源还做不好、而且音频塔不够像视频塔**。把 Veo 3 当唯一对照，有一点 strawman。
  - **异味 2（术语通胀）**：文中反复说 one-stage / single generative process，但**训练是两阶段**（先单模态音频、再融合微调），只是**推理**一次 ODE 出两路。这与**从第一天就把 AV 当同一个生成对象**不是一回事。
  - **异味 3（回避更简单的替代）**：没有认真回答**为什么不先把 Wan 训好、再接一个强 V2A（如 MMAudio）做级联**。联合生成的理论好处（互相决定）被当作自明，实验里也没有 vs 强级联的对照，只赢了两个弱开源联合模型。
- **结论**：动机方向对，但**开源联合生成尚未被探索**写过头了；真正站得住的句子是**开源还没有一个对称、可同步、音频塔自己会说话+会出音效的配方**。

## 2. Problem（问题）

- **核心问题**：在潜空间 DiT + flow matching 框架下，如何**同时**生成时间分辨率不同的视频与音频，并学到口型/动作–声音对齐，同时保住两路各自的生成质量。可拆成三个子问题：
  1. 音频塔与视频塔的架构不匹配（深度、隐维、注意力结构不同 → 要插块、投影、额外对齐损失）；
  2. 音频既要 TTS 又要 SFX/环境声，专用模型做不到**一条音轨里两者共存**；
  3. 31 个视频潜帧 vs 157 个音频 token 的时间错位。
- **普适性 & 重要性（理由）**：
  - 问题本身**广泛且重要**：Veo 3 之后这是视频生成的下一块必争之地，不是人造任务。对电影预告、UGC、数字人、本课题的**音画和谐**都直接相关。
  - 但本文把问题收窄成**5 秒、720×720、24 fps、单条共享音频潜变量**。多人对话、参考音色、镜头切换、分钟级叙事都被排除。这不是问题不重要，而是**问题定义偏系统演示**，评测无法回答**联合是否真比级联好**。
- **结论**：问题真实、重要；本文实际求解的是**短片联合生成的可运行配方**，不是联合生成的一般理论。

## 3. Method & Core Novelty（方法与创新）

- **方法概述**：
  - **对称双塔**：视频塔从 Wan2.2 5B 初始化；音频塔同结构（dim 3072 / 30 block / 24 head）从头训。每层：self-attn + 文本 cross-attn + **AV 双向 cross-attn**。隐维相同，故无中间投影。
  - **时间对齐**：音频 RoPE 频率乘 \(31/157 \approx 0.197\)（明确借鉴 MMAudio），让 cross-attn 的亲和矩阵走对角。
  - **条件**：冻结 T5，一次编码**视觉叙述 + `<S>…<E>` 台词 + `<AUDCAP>` 声学描述**的组合提示，两路共享。
  - **音频表示**：MMAudio 的 16 kHz 1D VAE（STFT→mel→latent）+ BigVGAN 声码器；flow matching。
  - **训练配方**：音频预训练（最长 12 s、偏语音、batch 2880、50k step）→ 5.04 s 微调并混入 VGGSound/AudioSet/WavCaps → 融合阶段冻结全部 FFN，只训 self-attn 与两类 cross-attn（11B 中 5.7B 可训），\(\lambda_v=0.85,\lambda_a=0.15\)，共享 \(t\)、独立噪声、**没有显式 sync loss**。推理用 UniPC + CFG。
  - **数据**：百万级内部成对视频；SyncNet \(|\mathrm{offset}|\le 3\) 且 confidence \(>1.5\)；人脸检测保证单人/多人/无人混合；MLLM 七帧+全音频打组合 caption。
- **核心创新点（vs 已有工作的 delta）**：
  - **真增量（工程/系统，不是新算子）**：
    1. **把音频塔做成视频塔的镜像并从头训成**会说话也会出音效**的基础模型**。这是相对 UniVerse-1（复用 ACE-Step 音乐模型 → 架构不对齐、要插块/投影/语义对齐损失）最实在的一步。JavisDiT 已经**同骨干**，但靠额外 prior estimator；Ovi 用对称 + 共享 T5 + 缩放 RoPE 把同步塞进双向注意力里，配方更干净。
    2. **严格的 AV 数据管线**（SyncNet 硬过滤 + 带台词/声学标签的组合 caption）几乎肯定是同步质量的主因，但实验完全没有 ablate 数据。
  - **换皮/组合（不要当新方法）**：
    - DiT + flow matching + RoPE + CFG：标准作业。
    - 逐层双向 cross-attn：UniVerse-1 已有 blockwise 融合。
    - 缩放 RoPE：论文自己写 “taking inspiration from MMAudio”。
    - **双流双向混合**在图像侧的祖先是 SD3 的 MM-DiT，文中完全不提。
    - 冻结 FFN 只训注意力：参数高效微调的常规手段，不是 AV 理论贡献。
  - **方法能否支撑 motivation**：推理确实是单次联合积分，能支持**互相看见**；但（1）共享 \(t\) + 独立噪声 + 无 sync loss，同步完全靠数据与注意力归纳，多人/错位说话时没有绑定机制；（2）\(\lambda_v\gg\lambda_a\) 等于承认视频塔不能被音频带跑，联合并不对称；（3）视频质量相对 Wan2.2 下降，说明**不牺牲单模态保真**的宣称被他们自己的结果打脸（他们轻描淡写成 expected / marginal）。
- **结论**：核心不是新融合算子，而是**对称音频塔 + 干净融合配方 + 同步数据**。作为开源 Veo 3 近似模板成立；作为方法论文，novelty 偏薄。

## 4. Related Work（相关工作）

- **关键相关工作**：
  - 视频基座：Wan2.2（直接祖先）、Sora、DiT、Flow Matching。
  - 联合生成直接对手：Veo 3（闭源标杆）、UniVerse-1（专家拼接）、JavisDiT（同骨干 + hierarchical prior）。
  - V2A / 音频潜空间：MMAudio（VAE、缩放 RoPE、评测协议都被直接拿来用）、Diff-Foley、Frieren。
  - A2V：HunyuanVideo-Avatar、HuMo、MagicInfinite、作者自己的 TalkingMachines——用来证明**不要人脸 mask**。
- **本文定位与区别**：相对级联 A2V/V2A，强调两路一起采样、无 mask/无辅助 sync 模块；相对 UniVerse-1，强调音频塔不是音乐模型、架构不必打补丁；相对 JavisDiT，强调不需要 learned prior。这个定位**清楚且公平**。
- **缺失的重要引用（该引而未引）**：
  1. **SD3 / MM-DiT（Esser et al., 2024）**：双流、层内双向混合、共享语义条件——Ovi 的双塔在结构谱系上更接近 MM-DiT 而不是**一种全新 paradigm**。不引会让读者高估架构新颖度。
  2. **2025 最强 A2V 人体动画**（OmniHuman-1、FantasyTalking、MultiTalk 等）：Related Work 把 A2V 写成 talking-head + mask 的狭隘子任务。对**联合生成是否必要**这个动机，不跟最强音频驱动视频比，对照不完整。（时间上 OmniHuman 2025-02 早于本稿。）
  3. **Meta Movie Gen（2024）** 等工业联合/级联音视频系统：至少应作为**非开源但方法更可查**的对照，而不是只点 Veo 3。
  4. 评测侧：SyncNet / LSE-C·LSE-D 是口型同步的社区标准，数据过滤用了 SyncNet，**评测却不用**，也未解释为何改用主观 PWR。
- **结论**：对手选得对（UniVerse-1 / JavisDiT），但谱系叙述故意从 MM-DiT 和强 A2V 旁绕开，让**unified paradigm**显得比实际更新。

## 5. Citation / Seed Potential（高引潜力）

- **判断**：**作为**开源联合 AV 配方 / 基线**高引潜力大；作为概念 seed 论文中等偏下。** 2026 年联合生成论文很大概率会引用 Ovi，类似 2023–24 年引用 AnimateDiff：不是因为定理新，而是因为可跟、可改、有权重。
- **依据**：
  - **时机对**：Veo 3 热度窗口内放出开源 5 s / 720p 模板，项目页给 demo / code / model。
  - **可迁移**：镜像塔 + 逐层双向注意力 + 缩放 RoPE + 组合 T5 prompt，别人换视频基座就能跟。本课题 V1**拆成每人一路音频、共享 Audio DiT**就是直接长在这套骨架上。
  - **不可复现的部分也构成引用壁垒**：内部百万视频、数十万小时语音、SyncNet 阈值与 MLLM caption 协议不公开。后来者会 cite**Ovi recipe**，但很难 verify 数据贡献有多大。
  - **限制天花板**：5 s、16 kHz、视频掉点、音频指标明显弱于专用 T2A/TTS。它更像**第一代开源联合生成**而不是**定义问题的 seed**。真正的 seed 更可能是：如何把联合生成做到长视频、多说话人、参考音色、或证明联合严格优于级联。
- **结论**：值得当基座引用，不值得当**新范式**神化。

## 6. Future Work & Improvements（改进空间）

- **局限性（论文已承认 vs 未承认）**：
  - 已承认：时长 5 s；稠密融合 + CFG 采样慢；16 kHz 1D-VAE 压扁音乐/空间声/细音色；视频质量相对 Wan2.2 下降。
  - **未承认但更关键**：
    1. **评测偏软**：联合生成只报 50 人 pairwise PWR，基准是 Verse-Bench（UniVerse-1 自家），对手只有 JavisDiT 与 UniVerse-1。没有 LSE-C/D、没有 vs 强级联（Wan2.2 + MMAudio / 强 TTS）、没有 vs Wan-S2V 这类 A2V。T2A/TTS 数字直接从 MMAudio / F5-TTS 论文抄 baseline，Ovi-Aud 全面落后（CLAP 0.224 vs MMAudio 0.348；WER 0.035 vs Fish Speech 0.008），却被叙述成 “comparable”。
    2. **几乎没有架构 ablation**：唯一消融是**CLAP+T5 分路 vs 单一 T5**。没有：双向 vs 单向、每层融合 vs 每隔 \(k\) 层、RoPE 缩放有无、\(\lambda\) 权重、冻 FFN vs 全微调、共享 \(t\) vs 独立时间表。
    3. **单条共享音频潜变量**：注意力可视化显示**语音看嘴、鼓点看鼓**，这是单声源场景的归纳，**不是说话人绑定**。多人同时说话时，谁的口型跟哪段波形，模型没有实体槽。
    4. **音色/内容都挤在文本里**：年龄、性别、口音、情绪靠 `<AUDCAP>` 形容词，没有参考音频条件。这与**音画和谐**（画面人物决定 *who*，驱动音频决定 *what/how*）正好相反。
    5. **同步监督被藏进数据**：训练无 sync loss，全靠 SyncNet 滤数据。换域、弱口型、侧脸、多人时这个归纳会碎。
- **跟进性小改**（不值得当课题主线）：更长 chunk 拼接、DMD2 蒸馏加速、换更高带宽 VAE / bandwidth extension、多分辨率 RoPE、把融合层变稀疏。论文 Limitations 已经把这些写掉了。
- **值得自己做的研究机会**（对本课题）：
  1. **实体槽 + 独立 speech stem**：把 Ovi 的一条音频潜变量拆成 \(K\) 路，共享 Audio DiT 参数，用动态空间路由把 stem \(k\) 送到人物 \(k\) 的空间区域。这直接打在 Ovi 最大的建模空洞上（单轨、无绑定）。
  2. **who / what 解耦**：内容条件用参考语音（如 Whisper）进音频流；身份用参考图进视频流。不要走 Ovi 的**文本形容音色**。HarmonizedVA 已在试 Whisper + mask；需要的是可泛化的路由而不是死 mask。
  3. **把**联合是否优于级联**做成可证伪实验**：Ovi 回避了这个问题。若联合只在口型上赢、在 SFX/音乐上输，则课题应只对语音 stem 做联合，环境声仍可级联。
  4. **显式同步与说话人损失**：数据滤 SyncNet 不够；需要对人的 lip-sync 与 stem-to-face matching。否则 V1 会重演**注意力自己找嘴**在多人上的失败。
- **结论**：论文自己指出的未来工作都是工程加长/加速；真正的研究缺口是**多实体音频绑定与身份–语音解耦**。

## Overall Verdict（总评）

- **是否值得深读**：值得当**系统/配方论文**精读（数据、两阶段、冻 FFN、\(\lambda\)、缩放 RoPE、组合 prompt），不值得当新融合理论精读。架构增量薄，开源权重与可抄骨架的价值高。
- **最大亮点**：对称音频塔从头训成 TTS+SFX 基础模型，避免了 UniVerse-1 式的专家拼接补丁；再加一套认真的同步数据管线，使**无 mask、无辅助 sync 模块**的联合生成在 5 s 上能跑。
- **最大隐患**：把工程模板说成 unified paradigm；评测几乎不能支持**联合优于级联**或**同步已解决**；单共享音频轨 + 文本隐含音色，正好是多人、音画和谐设定下会先碎的两处。对本课题：把 Ovi 当视频–音频联合先验，但不要继承它的**一条音轨、文本定音色、注意力自己找人**。
