# LoRA: Low-Rank Adaptation of Large Language Models

- **Authors / Year / Venue**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen / 2021–2022 / ICLR 2022（arXiv:2106.09685 v2）
- **Link**: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- **Code**: [microsoft/LoRA](https://github.com/microsoft/LoRA)（本仓库：[`code/foundations/lora/`](../../../code/foundations/lora/)）
- **TeX**: [`arxiv/foundations/lora/extracted/`](../../../arxiv/foundations/lora/extracted/)
- **One-line summary（一句话概括）**: 冻结 $W_0$，用 $BA$（$r\ll d$）参数化 $\Delta W$，训练时可合并进原权重、推理零额外延迟；假设是适配时的权重更新本身低秩。

## 1. Motivation（动机）

- 动机：
  - NLP 范式已是大规模预训练 + 下游适配。全量 fine-tuning 在 GPT-3 175B 上变成部署问题：每个任务一份 175B 副本，存储与切换成本不可接受。
  - 已有 PEFT 两条路都有硬伤：adapter 在残差旁插入瓶颈层，在线、短序列、小 batch 时引入可测 latency；prefix / prompt tuning 占用可用序列长度，且优化不稳定、参数变多不一定更好。
  - 从 Li et al. (2018)、Aghajanyan et al. (2020) 的 **intrinsic dimension** 跳到新假设：不仅解落在低维流形上，**适配时的 $\Delta W$ 本身也有低 intrinsic rank**。据此只优化 $B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$。

- 是否成立 / 是否有异味（理由）：
  - **痛点成立。** 175B 级多任务托管在 2021 年已经是真问题，不是为发论文造的。后续 SD / DiT / 视频基座上的每人一份 LoRA生态，说明同一存储–切换逻辑跨模态成立。
  - **一万倍参数、三倍显存有口径。** 10,000× 是 GPT-3 上只改 $W_q,W_v$、$r=4$、相对 **全量 350GB checkpoint**；不是相对 adapter / BitFit 的公平参数预算。3× VRAM 来自 Adam 不必存冻结参数的动量——对自适应优化器成立，对 SGD 没那么夸张。
  - **从 intrinsic dim 到 intrinsic rank 是启发式跳跃。** Aghajanyan 测的是随机投影后仍能学的解空间维度，不等于 $\Delta W$ 的矩阵秩。论文后半用 SVD / Grassmann 重叠把这条补成经验事实，但引言里写得像定理。
  - **对 adapter latency 的攻击针对特定 serving 形态。** GPT-2 medium、batch=1、seq=128 时 Adapter$^H$ +30%；batch=32、seq=512 只 +2–3%。吞吐型离线推理上，adapter 的深度税没那么致命。LoRA 真正打中的是 **可合并、可热切换**，不是adapter 一律不可用。

- **结论：** 动机站得住；夸大处在于把相对全量 FT的压缩数字当成相对所有 PEFT 的胜利，以及把经验假设写成近乎必然。

## 2. Problem（问题）

- 核心问题：
  给定预训练 $\Phi_0$，把每个下游任务的增量 $\Delta\Phi$ 编成远更小的 $\Theta$（$|\Theta|\ll|\Phi_0|$），使得
  $$
  \max_\Theta\sum_{(x,y)\in\mathcal{Z}}\sum_t\log p_{\Phi_0+\Delta\Phi(\Theta)}(y_t\mid x,y_{<t}),
  $$
  同时满足三条部署约束：
  1. **不增加推理深度**（相对已合并的 dense 层）；
  2. **不占用 prompt 长度**；
  3. **任务切换只换小模块**，共享同一份冻结骨干。
  - 实验范围：RoBERTa / DeBERTa 的 GLUE，GPT-2 的 E2E/WebNLG/DART，GPT-3 175B 的 WikiSQL / MNLI / SAMSum。原则声称适用于任意 dense 层，实验几乎只动 **attention 的 $W_q,W_v$**，MLP 冻结。

- 普适性 & 重要性（理由）：
  - 问题是 2019–2021 PEFT 主赛道（Houlsby adapter、Prefix-Tuning、BitFit），不是小众题。LoRA 的贡献是把参数少和推理图与全量 FT 同构绑在一起。
  - 重要性被事后验证得过猛：Hugging Face PEFT 默认后端、扩散社区的风格 LoRA、EasyControl 的 Condition Injection LoRA、LTX-2 trainer 的 IC-LoRA，都是同一条 $\Delta W=BA$ 的后代。对本仓库的生成线，它是 **ControlNet 之后、OminiControl / EasyControl 之前** 必须钉死的算子。
  - 本文 **不** 回答：量化后的 LoRA（QLoRA）、权重范数与方向分离（DoRA）、多 LoRA 同 batch 不合并时的路由、以及扩散 UNet/DiT 上该打哪些矩阵。这些是 follow-up，不是原文缺口里的没做所以不重要。

- **结论：** 问题广泛、重要、定义清楚；实验故意缩到 attention，读的时候不要把只打 Q/V 就够推广成定律。

## 3. Method & Core Novelty（方法与创新）

- 方法概述：
  - 预训练 $W_0\in\mathbb{R}^{d\times k}$ 冻结。更新写成
    $$
    h = W_0 x + BAx,\qquad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k},\; r\ll\min(d,k).
    $$
  - 初始化：$A\sim\mathcal{N}$，$B=0$，故起步 $\Delta W=0$，不破坏预训练前向。再乘 $\alpha/r$；Adam 下他们把 $\alpha$ 当常数、几乎不随 $r$ 重调（后文 rsLoRA 会指出 $\alpha/r$ 在大 $r$ 时把更新压没）。
  - Transformer 里可打 $W_q,W_k,W_v,W_o$ 与 MLP 两套；**主实验只打 attention，且常用 $\{W_q,W_v\}$**。部署时显式 $W\leftarrow W_0+BA$，前向与全量 FT 的 dense GEMM 相同。
  - 科学三问（GPT-3 175B、18M 预算）：打哪些矩阵、最优 $r$、$\Delta W$ 与 $W$ 的关系。结论：$\{W_q,W_v\}$ 优于只打 $W_q$/$W_k$；$r=1$–$4$ 在 WikiSQL/MNLI 上已饱和；$\Delta W$ 放大的是 $W$ **未强调** 的方向（$r=4$ 时放大因子约 20），而不是重复 $W$ 的主奇异方向。

- 核心创新点（vs 已有工作的 delta）：
  - **真创新不在低秩三个字。** Adapter 的瓶颈已经是低秩；Compacter 用 Kronecker 更狠。LoRA 的 delta 是：把低秩约束从 **串行插入的额外层** 改成 **与 $W_0$ 并行的、可代数合并的 $\Delta W$**。这一步同时解决 latency 和表达力在 $r\to d$ 时回收全量 FT的极限论证。
  - **可合并** 是产品级贡献，不是数学新定理。Prefix 做不到合并进 $W$；Houlsby adapter 深度变了，也不能无痛折进原 GEMM。
  - **子空间实验** 比方法本身更像论文：用 Grassmann 重叠说明增大 $r$ 并不覆盖新的有用子空间，给为什么 $r$ 极小仍够一个可检验叙事。这是相对 BitFit / Adapter 多出来的解释层。
  - **不是换皮。** 公式一眼像 adapter 去掉非线性、挪到旁路；但旁路 + 零初始化 $B$ + 可合并改变了部署合同，后续整个 LoRA 生态吃的就是这份合同。

- 方法能否支撑 motivation / problem：
  - 存储与切换：能。GPT-3 $r=4$、只打 Q/V → 约 35MB / 任务 vs 350GB。
  - 零推理延迟：能，**前提是合并**。文中自己写：合并后很难在同一 forward 里按样本换不同 $(A,B)$。多 LoRA 同 batch（ComfyUI、EasyControl 多条件）必须 **不合并、额外算 $BAx$**，latency 承诺在这里失效。
  - 匹配全量 FT：GLUE / E2E / GPT-3 三套表上 overall 持平或略好。RTE 等小集方差大；部分 GLUE 格是引用前人数字（`*`），对齐训练配方的程度要打折。GPT-2 的 $r$ 扫在 E2E 上峰值在 4–16，**不是** GPT-3 上的 $r=1$，文内已承认。

- 官方代码对照（[`code/foundations/lora/`](../../../code/foundations/lora/README_LOCAL.md)）：
  - `loralib/layers.py` 的 `Linear.forward`：未合并时 `F.linear(x, W_0) + (x A^\top B^\top)\,(\alpha/r)`，与式 $W_0x+BAx$ 等价（$A$ 存成 $[r,k]$）。`eval()` 且 `merge_weights=True` 时把 $BA$ 加进 `weight`，前向退化成一次 GEMM。
  - GPT-2 用 `MergedLinear` 切 `c_attn` 的 Q/V（`examples/NLG/src/model.py`）；RoBERTa / DeBERTa 只把 `query`/`value` 换成 `lora.Linear`——与论文主实验只打 $W_q,W_v$一致。
  - **实现 ≠ 论文原文：** `Linear` 用 Kaiming $A$、$B=0$（注释写明与论文高斯 $A$ 不同）；`Embedding` 甚至把 $A$、$B$ 的零/随机对调。起步 $\Delta W=0$ 仍成立，但论文怎么写不能当仓库怎么训。

- **结论：** 核心 delta 是可合并的并行低秩更新+ GPT-3 级实证与子空间解释；公式本身极度简单，简单是优点不是罪名。读生成论文时记住：他们几乎没打 MLP，扩散/DiT 上默认打哪些矩阵是后人经验，不是本文定理。

## 4. Related Work（相关工作）

- 关键相关工作：
  - **Adapter**：Houlsby et al. 2019；Lin et al. 2020（本文 Adapter$^L$）；Pfeiffer AdapterFusion；AdapterDrop。直接 baseline。
  - **Prompt / Prefix**：Prefix-Tuning（Li & Liang 2021）、Prompt-Tuning（Lester et al. 2021）、WARP、P-Tuning。占用长度、优化难，是 LoRA 攻击的另一极。
  - **只训子集**：BitFit（Zaken et al.，同期）、FT$^{\text{Top2}}$。
  - **低秩先验**：intrinsic dimension（Li 2018；Aghajanyan 2020）；训练期显式低秩分解（Sainath、Jaderberg 等）——那些是压 $W$ 本身，不是压 **frozen $W$ 的 $\Delta W$**。
  - **Compacter**（Mahabadi 2021）：同期用 Kronecker 参数化 adapter，文中点名可与 LoRA 结合。

- 本文定位与区别：
  - 相对 adapter：同为瓶颈，但 LoRA 在权重空间并行、可吸收进 $W_0$。
  - 相对 prefix：不减有效上下文，参数效率随 $r$ 单调更好控。
  - 相对训练时低秩约束 $W$：对象是适配增量，预训练 $W_0$ 保持满秩特征。

- 缺失的重要引用（如有）：
  - 2021 年底 Diff pruning / FISH mask / LoRA 同期的若干 PEFT 调查不必强求。真正后来证明该引方向的是：**把 LoRA 接到量化（QLoRA）、接到扩散 UNet、接到 DoRA 的幅度–方向分解**——这些是 2022–2024 的下游，不是原文漏引。
  - 文内对 Compacter 一笔带过、几乎无实验对照；若强调低秩参数化 uniqueness，这组对照偏弱。

- **结论：** 定位清楚，最直接的对手是 adapter 与 prefix；科学叙事接 intrinsic dim。不是文献真空里的发明，是把已有低秩直觉改到可合并的 $\Delta W$上。

## 5. Citation / Seed Potential（高引潜力）

- 判断：**已经是 seed 论文**（不是有潜力）。PEFT、开源 LLM 微调、Stable Diffusion 风格化、DiT 条件 LoRA，三条产业线都把它当默认算子。

- 依据：
  - **问题普适**：任意带 dense 层的预训练模型。
  - **方法可迁移**：实现是替换 `nn.Linear`；`loralib` / PEFT 把复现成本压到几行。
  - **开新方向**：不是新架构，是新 **适配接口**。后续 QLoRA、DoRA、LoRA+、rsLoRA、PiSSA、EasyControl CI-LoRA、IC-LoRA 都是在这接口上加约束或换注入点。
  - **Timing**：GPT-3 刚把全量 FT 变成政治不正确的 2021 年；开源权重（RoBERTa/GPT-2）+ 闭源 175B 数字同时给学术和工业两边故事。
  - **复现**：官方仓库给 NLU/NLG 例子；GPT-3 数字不可复现（闭源），但不妨碍方法扩散。

- **结论：** 高引已成事实。对本项目的价值是 **算子课**：读 EasyControl / LTX-2 LoRA trainer 之前，必须能默写 $h=W_0x+BAx$、合并条件、以及只打 Q/V不是普适。

## 6. Future Work & Improvements（改进空间）

- 局限性：
  - 主结果几乎只适配 attention；MLP / LayerNorm / bias 明确留给 future work。扩散和现代 LLM 实践里 FFN LoRA 往往同样关键。
  - $r=1$ 就够高度依赖任务与预训练模型；跨语种、大分布偏移时他们自己用思想实验承认要抬 $r$。
  - 合并与 **多样本多 LoRA** 互斥；今天 Comfy 式多风格融合走的是不合并路径，原文的 latency 承诺不适用。
  - $\alpha/r$ 缩放在大 $r$ 时压制更新（Kalajdzievski rsLoRA）；$B=0$ 起步使早期梯度只走 $A$ 一侧（LoRA+、PiSSA 后续打补丁）。
  - GPT-3 实验无法外部复核；GLUE 部分格子是抄前人。
  - 子空间分析集中在第 48 层、$\Delta W_q/\Delta W_v$，解释力强但不是全层、全矩阵类型的定理。

- 可做的未来工作 / 研究机会：
  - **跟进性小改**（多数已被做完）：QLoRA、DoRA、自适应 $r$、换初始化、FFN 也打、多 LoRA 路由。
  - **仍值得自己做的**（对齐本仓库）：
    1. DiT / 双流（Wan、Ovi、LTX-2）上 **哪些投影该 LoRA、哪些该全量或冻结**——原文的 $W_q,W_v$ 启发式不能直接搬到 MM-DiT / 音视频 cross-attn。
    2. **不合并** 时的多条件 LoRA（EasyControl CI-LoRA + KV cache）如何在延迟与可加性之间取舍。
    3. IC-LoRA / 视频 LoRA 里 $\Delta W$ 是否仍放大 $W$ 未强调方向，还是在运动/身份这种下游上变成另一套几何。这是原文第 6 节真正可迁移的科学问题，不是再扫一遍 GLUE。

- **结论：** 原文把接口做对了，把层选择和缩放细节留给生态；我们跟进应打在 **视频/音频 DiT 的注入点与多 LoRA 服务语义**，而不是再发明一个低秩分解。

## Overall Verdict（总评）

- 是否值得深读：**值得，而且要读公式和科学两节，不要停在 abstract 的压缩数字。** 方法半页就能写完，真正要带走的是：可合并合同、$\Delta W$ 低秩的经验范围、以及只打 Q/V的实验边界。
- 最大亮点：把 PEFT 从加模块改成加一个可吸收的 $\Delta W$，同时在 175B 上把故事讲圆。
- 最大隐患：把 GPT-3 + 小 $r$ + 只打 attention 的成功，误读成所有模态、所有层、所有 $r$ 的定律；在生成模型里照抄会漏 FFN、漏多 LoRA 不合并、漏 $\alpha$ 缩放。
