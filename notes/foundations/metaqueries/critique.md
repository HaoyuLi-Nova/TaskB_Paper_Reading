# Transfer between Modalities with MetaQueries

- **Authors / Year / Venue**: Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, Saining Xie / 2025 / arXiv:2504.06256（ICLR 2026 投稿；HTML 修订 2026-08-11）
- **Link**: [https://arxiv.org/abs/2504.06256](https://arxiv.org/abs/2504.06256) · 项目页 [https://xichenpan.com/metaquery/](https://xichenpan.com/metaquery/)
- **原文 TeX**: `arxiv/foundations/metaqueries/extracted/`
- **LTX-2 引用语境**: thinking tokens 可携带**在语义上比在视觉/音频空间中更易生成的缺失细节**（Pan et al., 2025）
- **One-line summary（一句话概括）**: 向冻结 MLLM 塞入一组 learnable query，把查出的隐状态经 connector 接到扩散解码器；理解能力原封不动，生成只训 query + connector +（可选）DiT，用图文对 + 标准 denoising。

## 1. Motivation（动机）

- 动机：
  - 统一多模态（既出文字又出像素）的主流做法是 **改 MLLM 骨干**：回归视觉特征、自回归视觉 token、或在同一 Transformer 里混 diffusion。配方复杂、数据/损失要平衡，理解往往被生成带崩。
  - 作者的哲学：**理解留给冻结 MLLM，像素留给扩散**。缺的只是一座桥，把 MLLM 的世界知识 / 推理 / in-context 转到生成条件上。
  - 只用 LLM last-layer embedding 当 T2I 文本编码器（Sana、Lumina-Next）做得到对齐，但做不到**先在 LLM 里回答再画**——例如 *Yellowstone 所在国的国旗*。
- 是否成立 / 是否有异味（理由）：
  - **痛点成立。** Janus / Show-o / Transfusion / Emu 确实在多任务平衡上吃过亏；**冻住理解、外挂生成**是对 2024–2025 统一模型疲劳的合理反击。GILL（Koh 2023）已经探过 learnable token + 冻结 MLLM，但目标是检索+回归而不是 denoising。
  - **异味 1（**SOTA 理解**几乎是同义反复）。** 骨干冻住，MME / MMMU 分数就是 Qwen2.5-VL / LLaVA-OV 原分数。这不是方法赢了理解，是 **没动理解**。表 4 把 MetaQuery-XL 的 58.6 MMMU 和 Janus-Pro-7B 的 41.0 放一起，读者会以为是统一模型的理解胜利。
  - **异味 2（生成 SOTA 高度依赖底座 Sana，且 GenEval 带 rewriting 标记）。** 表 4 里 MetaQuery 的 GenEval 标了 $\dagger$（rewritten），Janus-Pro 部分数字不带；MJHQ FID 又对 Janus-Pro 自己重测。对照不完全同协议。**SOTA-level generation**更像 **Sana 的美学 + MLLM 的 prompt 理解**，桥本身的增量要看表 1–2 的 controlled study，不要看表 4 的全家桶。
  - **异味 3（和 GPT-4o 图像的类比过满）。** 引言说 token→transformer→diffusion→pixels**或许**和 GPT-4o 同哲学。这是营销句，没有证据。

- **结论：** **别把理解和生成焊进同一套损失**站得住；把冻结骨干的理解分写成方法贡献，有包装异味。

## 2. Problem（问题）

- 核心问题：
  给定冻结 MLLM $f$ 和条件扩散 $g$，学习 query $\mathcal{Q}\in\mathbb{R}^{N\times D}$ 与 connector $h$，使得
  $$
  \mathcal{C}=h\bigl(f([\text{multimodal prefix};\,\mathcal{Q}])\bigr),\quad g(\cdot\mid\mathcal{C})\approx p(\text{pixels}\mid\text{prefix}).
  $$
  约束：不改 $f$ 的权重与因果 mask；训练只需图文对 + 原扩散目标；进一步可 instruction-tune 做编辑 / 主体驱动。
- 普适性 & 重要性（理由）：
  - **MLLM 条件化扩散**是 2025 统一模型的真问题（Kosmos-G、MetaMorph、LMFusion、ELLAs）。重要性在 **可插拔**：换 Qwen-VL 或换 Sana 不必重训理解。
  - 对 LTX-2 尤其相关：LTX-2 冻 Gemma、只训连接器 + thinking tokens，再条件化音视频 DiT——和 MetaQuery 是 **同一接口形状**（extra token 查询冻结 LLM，投影后进扩散）。Pan 才是三篇引用里架构上最近的一篇。
  - 本文 **不** 覆盖：联合音视频、原生多模态交错生成（query 在因果序列末端，不是交错解码）、以及**缺失细节在语义空间更好生成**的直接实验——这句是 LTX-2 的解读，原文用的是 reconstruction / editing / WISE 推理。

- **结论：** 问题重要、接口可迁移；LTX-2 第 161 行的**语义空间补细节**是合理外推，不是本文定理。

## 3. Method & Core Novelty（方法与创新）

- 方法概述：
  - $\mathcal{Q}$ 随机初始化，整段仍因果 mask（query 能看 prefix，prefix 看不见 query——标准 decoder 前缀）。
  - connector：先按 MLLM 隐维做双向 Transformer（Enc-Proj，24 层），再投影到 DiT 条件维。比先投影再编码更省参数、更好。
  - $N$ 可缩放：T2I 美学约 64 饱和，对齐和重建随 $N$ 继续涨；主模型 $N=256$。
  - 两阶段：25M 公开图文对 pretrain；用 mmC4 自然图对 + SigLIP 聚类 + MLLM 写指令，挖 2.4M pair 做 instruction tune（主体驱动、视觉联想、logo）。
  - 骨干：LLaVA-OV-0.5B / Qwen2.5-VL-3B / 7B；头：SD-v1.5 或 Sana-1.6B。
- 核心创新点（vs 已有工作的 delta）：
  - **真增量 1：controlled ablation 说明 learnable query ≠ last-layer embed。** 表 1：随机 query 美学还行（FID 8.59）但对齐崩（GenEval 0.35）；可学习 64 token 对齐回到 0.56。表 9：WISE / CommonsenseT2I 上 query 明显赢 last-layer——这是**先推理再画**真正需要 in-context 槽位的证据。
  - **真增量 2：冻 MLLM ≈ 全量调 MLLM 的生成（表 2）。** Frozen+训 DiT 的 FID 甚至略好。这为**理解生成解耦**提供了比口号硬的数字。
  - **真增量 3：自然图对 instruction 数据。** 不靠 InstructPix2Pix 专家模型合成，挖网页共现。DreamBench 上 DINO 0.737 超过 Kosmos-G。这条是数据贡献，可独立于 query 接口。
  - **换皮部分：** learnable query + connector + 冻结 LLM 在 Flamingo / BLIP-2 / GILL / Kosmos-G 都出现过。delta 是 **训练目标改成纯 denoising、且系统论证冻骨干足够**。不是新算子。
  - **对 LTX-2 的对齐与错位：**
    - 对齐：冻结文本塔、可学习 extra token、connector、扩散目标——几乎是 LTX-2 文本连接器的图像版。
    - 错位：MetaQuery 的 query **替代**了**把 LLM hidden 当条件**；LTX-2 是 **原文 token + thinking tokens 并存**。Pan 的重建实验（$N$ 越大越能从 query 里找回像素）支持**extra token 能扛细粒度条件**，这才是**缺失细节在语义侧生成**最接近的证据。LTX-2 没有做对应的**去掉 thinking tokens 后细节是否只能靠像素 DiT 硬编**消融。

- **结论：** 接口是组合，贡献在**冻骨干 + denoising 就够**的实证和数据配方；对 thinking tokens 的支持是**query 能从 LLM 查出比 caption embedding 更富的条件**，不是**DiT 里需要思考槽**。

## 4. Related Work（相关工作）

- 关键相关工作：
  - 统一模型：SEED-X、Emu、MetaMorph、LaVIT、Chameleon、Show-o、EMU3、Janus 系、DreamLLM、Transfusion。
  - 冻结 LLM 生成：LMFusion（并行 FFN/QKV）、GILL（learnable token + 对比/回归）。
  - LLM 当 T2I 编码器：Lumina-Next、Sana、Kosmos-G。
  - query 条件：Flamingo resampler、BLIP-2 Q-Former、NEXT-GPT、Kosmos-G（$N=77$ 对齐 CLIP 文本长度）。
- 本文定位与区别：
  - 相对 Janus 系：不训视觉 tokenizer、不改理解。
  - 相对 GILL：目标换成扩散、任务从**上下文插图**扩到 T2I / 编辑 / 主体。
  - 相对 Kosmos-G：作者之一重叠；Kosmos-G 仍调 MLLM 且绑 CLIP 77 token。MetaQuery 把 $N$ 做成可扫超参并冻骨干。
- 缺失的重要引用（如有）：
  - **Darcet registers / Victor** 几乎不出现。LTX-2 把三篇绑成**thinking tokens**，但 MetaQuery 自己的 related work 走的是统一模型线，不承认自己是 register 论文。概念上 query ≈ register，谱系被写断了。
  - IP-Adapter、ELLAs、unCLIP 的**可学习条件投影**讨论不足——很多时候一个 MLP adapter 已经能把 LLM embed 接到 SD，24 层 connector（316M）是否必要，只在 Proj-Enc vs Enc-Proj 内部比，没和外接 adapter 比。
  - 视频 / 音频扩散条件化（LTX-2、Ovi）完全不在范围，引用时不要写成跨模态生成通解。

- **结论：** 统一模型对照全；和 register 文献、轻量 adapter 的对话故意或无意地缺了，导致 LTX-2 的捆绑引用在 MetaQuery 原文里找不到对应句子。

## 5. Citation / Seed Potential（高引潜力）

- 判断：
  **中高，有 seed 相。** **冻 MLLM + query 桥接扩散**会成为统一模型的一条标准配方，类似 IP-Adapter 之于图像条件。是否高引取决于开源权重和是否被 Qwen-VL / 内部 Meta 产品线采用。
- 依据：
  - 问题普适、方法可迁移（换骨干 / 换 DiT）、训练相对 Transfusion 类简单。
  - Timing 好：2025 春正值 GPT-4o 图像、Janus-Pro 之后，社区要**理解别掉点**的解。
  - 复现成本：25M 图文 + 24 层 connector 不是玩具，但远小于从头训统一 Transformer。
  - 隐患：若社区把**冻骨干**视为显然，引用会缩成一句 recipe；WISE SOTA 可能随更强 MLLM 自动刷新，论文变成**换了 Qwen2.5-VL**的注脚。

- **结论：** 值得当统一生成的基线接口引用；seed 潜力高于 Victor，低于 Darcet。

## 6. Future Work & Improvements（改进空间）

- 局限性：
  - 因果 mask 下 query 是单向读取，作者为兼容故意不开双向；双向 query 会不会更好只靠**简单**挡掉。
  - 24 层 connector 很重，和**minimal recipe**的自我定位有张力。
  - 表 8：指令微调 / 视觉指令微调对 T2I 几乎正交——等于承认 **生成吃到的主要是预训练语言模型的语言建模，不是多模态理解**。那**知识增强生成**有多少来自视觉编码器，多少来自纯文本世界知识，拆得不够。
  - 重建实验用冻结 MLLM 仍要额外 fine-tune query；**轻松迁移到编辑**是 1000 step 的定性，没有标准编辑基准（如 MagicBrush）主表。
  - 音频 / 视频只在一句 future work。
- 可做的未来工作 / 研究机会：
  - **跟进性小改：** 减薄 connector、开双向 attention、把 $N$ 做成动态。
  - **值得做的（对本仓库）：** 把 MetaQuery 接口接到 **音视频双流 DiT**（Ovi / LTX-2）：冻结 Gemma，query 分别产视频条件与音频条件，直接检验 LTX-2**语义侧补缺失细节**——例如用 query 承载**音色 / 空间音**这类比波形更容易在 token 空间说清的属性。这比再刷 MJHQ 有增量。
  - 把 Darcet 的**输出丢掉 register**和 MetaQuery 的**输出就是条件**做成同一骨干上的对照，分清 scratch space vs interface。

- **结论：** 图像统一生成的接口已经够用；真正开放的是跨模态（音、视频）和**query 里到底编码了什么**。

## Overall Verdict（总评）

- 是否值得深读：
  **值得读 §3 的 controlled study 和 §5.5 的 WISE / CommonsenseT2I。** 这是三篇里和 LTX-2 文本连接器最同构的一篇：冻结 LLM、learnable extra token、connector、扩散损失。
- 最大亮点 / 最大隐患：
  亮点是冻骨干仍能把推理槽位交给扩散（query vs last-layer 的 WISE 差）。隐患是全家桶表把底座分数算进方法，以及 LTX-2 把**语义空间生成缺失细节**写成本文结论——原文只证明 query 能当更富的条件，没有证明像素空间生成不了这些细节。
