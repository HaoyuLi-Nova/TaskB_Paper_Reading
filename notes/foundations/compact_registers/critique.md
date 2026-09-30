# Efficient Vision-Language Models by Summarizing Visual Tokens into Compact Registers

- **Authors / Year / Venue**: Yuxin Wen, Qingqing Cao, Qichen Fu, Sachin Mehta, Mahyar Najibi / 2024 / arXiv:2410.14072（ICLR 2025 **withdrawn**）
- **Link**: [https://arxiv.org/abs/2410.14072](https://arxiv.org/abs/2410.14072) · OpenReview [tNxr38vfYR](https://openreview.net/forum?id=tNxr38vfYR)
- **原文 TeX**: `arxiv/foundations/compact_registers/extracted/`
- **LTX-2 引用语境**: thinking tokens**可携带聚合的语境信息**（Wen et al., 2024）
- **One-line summary（一句话概括）**: 在 LLaVA 式 VLM 的 LLM 前 $k$ 层把视觉 token 汇总进少量 learnable register，然后**丢掉全部原始视觉 token**，用寄存器 + 文本继续跑完；8 个寄存器（约原视觉序列 1%）换 3.3× 吞吐、训练时长短 43%，平均分掉不到 4%。

## 1. Motivation（动机）

- 动机：
  - LLaVA-NeXT 一张图可到 2880 visual token，十二个评测集平均指令不到 70 token。二次注意力被视觉前缀主导，训练和推理都贵。
  - FastV 按 attention 分数丢 token：腰斩还行，再丢就崩，而且必须物化 attention 矩阵，**和 FlashAttention / SDPA 不兼容**。
  - Perceiver / Q-Former 能压到很少 query，但多一截 Transformer、参数以亿计。
  - 作者要一条**几乎不加参数、兼容高效 attention、高压缩比还不崩**的中间路。
- 是否成立 / 是否有异味（理由）：
  - **痛点成立。** 视觉 token 冗余、前几层之后文本对视觉的 attention 掉近零，是 FastV 已经测过的真现象，不是人造题。FlashAttention 不兼容也是部署真约束。
  - **异味 1（**不到 4% 掉点**是 12 项归一化均值）。** 附录分项里 TextVQA / MM-Vet / OCR 类对分辨率敏感的任务掉得更狠；ScienceQA 几乎不动。把**平均 4%**写成方法卖点，等于让不读图的任务给读图任务垫背。
  - **异味 2（吞吐测的是 batch=16、prompt=64、生成 2 或 128 token）。** 这是为了让视觉前缀占比极大的实验室设置。真实多轮、长指令、batch=1 时，3.3× 会缩水——作者自己的 128-token 曲线已经比 2-token 平很多。
  - **异味 3（ICLR 2025 撤稿）。** 方法能跑，但作为**SOTA 高效 VLM**投稿，审稿场上 FastV / PruMerge / TokenPacker 同期卷得很凶；撤稿本身不否证实验，说明 **增量被认为不够成一篇顶会方法文**。

- **结论：** 效率动机真实；数字包装偏实验室分布，且方法增量偏工程，和顶会预期有落差。

## 2. Problem（问题）

- 核心问题：
  在冻结 CLIP 视觉塔、可训 LLM 的 LLaVA 管线里，把 $N$ 个投影后视觉 token 在第 $k$ 层压成 $M\ll N$ 个寄存器，使
  $$
  x = [x_V;\, x_R;\, x_T] \xrightarrow{\text{layers }0..k-1} x[N:] = [x_R;\, x_T]
  $$
  后续层只在短序列上跑，同时尽量保住 VQA / 综合基准均值。
- 普适性 & 重要性（理由）：
  - **视觉 token 太多**是 2024 年 LLaVA 系的主赛道问题（FastV、PruMerge、LLaVA-PruMerge、MQT、TokenPacker）。重要性在**部署吞吐**，不是新的多模态理论。
  - 问题被定义成 **LLaVA-style 前缀拼接**。Flamingo 式 cross-attn、早期融合（Fuyu / Chameleon）不在范围内；那些架构视觉长度本来就不进 LLM 全文。
  - 本文 **不** 保证：OCR / 数数 / 高分辨率细节在 $M=8$ 时还在；也不保证训练后还能在推理时改 $M$（附录 A.5：head/tail 截断远差于重训）。

- **结论：** 问题真实但架构绑定 LLaVA 前缀；对 LTX-2 那种**条件 token 要保留原文**的设定，本文的问题定义是反的。

## 3. Method & Core Novelty（方法与创新）

- 方法概述：
  - 可学习 $x_R$（默认跟在 $x_V$ 后），$k=3$（约 LLM 10% 层）处切掉 $x_V$。
  - **没有** 额外 summary loss；靠语言模型自己把视觉写进寄存器。注意力可视化：前两层 register→vision 很弱，**第 3 层才真正汇总**——和 $k<3$ 就崩的消融一致。
  - 预训练只训 projector + registers；指令微调解冻 LLM。底座 LLaVA-1.5（CLIP-L + Vicuna-7B），再换 Vicuna-13B / Llama-3-8B / Mistral / LLaVA-NeXT+Qwen2。
  - 对照：FastV、Perceiver Resampler（+252.86M vs Victor +1.78M）。
- 核心创新点（vs 已有工作的 delta）：
  - **真增量：把 Darcet register 从**额外 scratch space**改成**可丢弃的视觉摘要**，并卡在 FastV 发现的早期层。** 单独看，early drop 是 FastV，register 是 Darcet，用 LLM 自己做 resample 是 Perceiver 的穷人版。组合点是 **drop 的不是按分数选的 patch，而是整段 $x_V$，摘要必须进专用 token**。附录 A.6 对照**保留最后 $M$ 个视觉 token**显著更差，说明 **专用寄存器不是摆设**。
  - **换皮部分：** 相对 Q-Former / Perceiver，少了独立 Transformer 和对比/匹配损失，换来的是必须 **从头训**（作者自己列为 limitation：不能 post-hoc 插到已有 VLM 上）。
  - **对 LTX-2 的错位：** Victor 的核心动作是 **扔掉原始视觉 token**；LTX-2 文本连接器是 **原始 token 与 thinking tokens 一并送进 DiT**。Wen 支撑的是**寄存器能聚合**，不是**聚合完还把原文留下**。LTX-2 若真按 Victor 做，应该 drop 或压缩 caption token，而不是只填 padding。
  - 不 drop、只加 8 个 register（§5.6，对齐 Darcet）有 ~1% 的**免费午餐**——这条反而更接近 LTX-2，但作者把它当旁路实验，主卖点仍是极端压缩。

- **结论：** 工程组合有效，专用寄存器有消融支撑；和 LTX-2**双保留**接口不是同一件事。

## 4. Related Work（相关工作）

- 关键相关工作：
  - VLM：LLaVA / InstructBLIP / Flamingo / Qwen-VL / 早期融合。
  - token 减少：DynamicViT、EViT、PuMer、FastV、Perceiver、Q-Former。
  - register：Burtsev Memory Transformer、Darcet 2023。
- 本文定位与区别：
  - 相对 FastV：训练期就可 drop、兼容 FlashAttention、高压缩更稳。
  - 相对 Perceiver：参数少两个数量级，用 LLM 前几层当 resampler。
  - 相对 Darcet：目标从**别污染 patch**变成**让 LLM 写摘要然后扔掉视觉**。
- 缺失的重要引用（如有）：
  - **LLaVA-PruMerge、TokenPacker、MQT-LLaVA、DeCo** 等同窗口视觉压缩，主文几乎只打 FastV + Perceiver，对照光谱偏窄。
  - 没有把**在视觉塔里就压**（C-Abstractor / Adaptive Average Pool）和**在 LLM 里压**做成同 FLOPs 对照——前者更便宜，可能已经够 4% 掉点那个区。
  - 因果 mask 下 register 只能看左边的 $x_V$；双向视觉编码器里做同样的事会不会更好，没比。

- **结论：** 直接打的两个 baseline 合理；2024 视觉 token 压缩密集区引用不够，部分解释了撤稿。

## 5. Citation / Seed Potential（高引潜力）

- 判断：
  **中低。工具论文，难成 seed。** 会被**早期层摘要 + 丢视觉 token**的实现引用，但会被 FastV（问题发现）和后续训练免费的 token pruning 盖过。ICLR 撤稿进一步限制引用惯性。
- 依据：
  - 问题普适，方法可迁移性中等：依赖**必须重新微调 LLM**，不能即插即用。
  - Timing 一般：2024 秋，LLaVA-1.5 已经不是最强底座，高分辨率 / 视频 VLM 的 token 数问题更凶，本文只在图上做到 2880→16。
  - LTX-2 引用它当**聚合语境**证据，是 **概念借用** 不是方法继承；这篇不太会因为 LTX-2 而变成高引。

- **结论：** 读作**Darcet register 在 VLM 里可以当压缩瓶颈**的存在性证明即可，不必当效率 SOTA。

## 6. Future Work & Improvements（改进空间）

- 局限性：
  - 必须进训练；推理期改 $M$ 要重训。
  - $k$ 与 $M$ 绑定架构深度，换 LLM 要重扫（他们扫了几个 7–13B，没有到 70B）。
  - 高压缩下 OCR / 空间关系几乎必然丢，平均分掩盖了这一点。
  - 没有理论：为何第 3 层才汇总，是优化伪影还是信息流必要？
- 可做的未来工作 / 研究机会：
  - **跟进性小改：** 辅助 loss 让 $M$ 可变；post-training 只训 register；视频多帧共享寄存器。
  - **值得做的：** 在 **扩散条件编码器**（Gemma / T5 连接器）里做**前几层混合、后几层短序列**——这才是 LTX-2 能直接抄的效率杠杆，原文没碰生成模型。
  - 把 Victor 的 drop 和 Darcet 的**保留 patch、寄存器只当 scratch**在同一 VLM 上拆开，量化**摘要**vs**隔离**各自贡献。

- **结论：** 缺的是训练免费与任务自适应压缩；不是再发一条 LLaVA-1.5 Pareto 曲线。

## Overall Verdict（总评）

- 是否值得深读：
  **扫一遍方法和 §5.6 / A.6 即可，不必精读全部基准图。** 对理解 LTX-2 第 161 行：**聚合**来自这篇，但 LTX-2 **没有做 drop**，引用只借了半句。
- 最大亮点 / 最大隐患：
  亮点是专用 register 相对**截断原视觉 token**的消融，以及和 FlashAttention 兼容的实际吞吐。隐患是平均掉点掩盖 OCR，以及把 Darcet 的隔离寄存器讲成压缩瓶颈后，被生成论文继续误引成**多几个 token 就能带全局语境**。
