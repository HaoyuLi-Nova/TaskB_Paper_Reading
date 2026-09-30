# Vision Transformers Need Registers

- **Authors / Year / Venue**: Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski / 2023–2024 / ICLR 2024 Oral（Outstanding Paper）
- **Link**: [https://arxiv.org/abs/2309.16588](https://arxiv.org/abs/2309.16588) · ICLR [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/0b408293619f725fd30162af057e531a-Paper-Conference.pdf)
- **原文 TeX**: `arxiv/foundations/vit_registers/extracted/`
- **LTX-2 引用语境**: 作为 thinking tokens**全局信息载体**的视觉侧前驱（Darcet et al., 2023）
- **One-line summary（一句话概括）**: 大且训得够久的 ViT 会把低信息背景 patch **回收成内部寄存器**（高范数 outlier）；显式加几个可学习 `[reg]` token 把这个角色隔离出去，特征图伪影消失，稠密任务与无监督物体发现变好。

## 1. Motivation（动机）

- 动机：
  - DINO 的 attention map 干净、可被 LOST 一类无监督发现直接用；DINOv2 稠密探针（分割 / 深度）更强，却**对不上 LOST**，特征图上能肉眼看到尖峰伪影。
  - 同一套可视化打到 DeiT-III（标签监督）和 OpenCLIP（图文监督）上，**除了原始 DINO 几乎人人有伪影**。作者要把**DINO 是例外、DINOv2 才是 ViT 基线行为**说清楚，并给一个不改主干的修法。
- 是否成立 / 是否有异味（理由）：
  - **痛点成立。** 伪影不是审美问题：它让 patch token 丢掉局部信息，直接伤稠密预测和依赖平滑特征图的下游。ICLR oral + DINOv2 后续默认带 register，事后验证了这条动机。
  - **异味 1（把**可解释 attention**写成主叙事，真正的定量收益却偏窄）。** 引言从 LOST / 无监督分割切入，实验里分类几乎不动（DeiT-III IN top-1 84.7→84.7），DINOv2 的 ADE20k / NYUd 只有小幅涨。最大数字在 LOST corloc（DINOv2 VOC07 35.3→55.4），而那套算法对特征平滑极度敏感——等于用**对伪影过敏的下游**来证明伪影有害。
  - **异味 2（cutoff=150 是手工阈值）。** **高范数 = artifact**在 DINOv2-g 上双峰明显，换模型就要重画分布。作者承认 cutoff 随模型变，但主文仍把它当操作定义。
  - **附录自己拆了一层：** 官方 DINOv2 的 pos-emb 插值没有 antialiasing，outlier 空间分布带竖条纹，和 bicubic 梯度条纹同构。伪影不全是**模型学会回收背景**，有一部分是**实现细节**。正文把这条埋进 appendix，主故事仍写成纯粹的计算角色回收。

- **结论：** 动机站得住，现象真实且跨监督范式；叙事把**attention 好看**和**稠密任务涨点**绑得比数据更紧，对插值伪影的坦白不够靠前。

## 2. Problem（问题）

- 核心问题：
  在足够大、训得足够久的 ViT 里，约 2% 的 patch token 在中后层长成 ~10× 范数的 outlier：它们出现在与邻居高度相似的冗余区域，局部位置 / 像素可探性差，却能当单 token 图像分类器（接近 `[CLS]`）。问题是：**模型已经在用 patch 当寄存器，但这会污染要用局部特征的下游。**
  操作定义：给序列追加 $N$ 个与图像无关的可学习 token，末端丢掉它们，只留 `[CLS]` 与 patch。
- 普适性 & 重要性（理由）：
  - 问题在 2023 年已经跨 DeiT / CLIP / DINOv2 出现，不是 DINOv2 独有 bug。后续 SigLIP、DINOv3、大量 VLM 视觉塔默认带 register，说明它变成了 **ViT 预训练的默认补丁**。
  - 重要性分两层：（1）稠密特征干净，这是分割 / 深度 / 匹配真正要的；（2）概念上证明 Transformer **会自发开辟 scratch space**，不给它位子它就抢背景 patch。
  - 本文 **不** 回答：因果 LM 的 attention sink、扩散 DiT 条件侧的 extra token、以及**寄存器内容是否语义可解码**。后两者是 LTX-2 thinking tokens 想借的，原文没有做。

- **结论：** 问题真实、对视觉预训练重要；不要把它读成**生成模型里 extra token 能补语义细节**的定理——那是引用链上的外推。

## 3. Method & Core Novelty（方法与创新）

- 方法概述：
  - 诊断：输出范数直方图双峰；outlier 约在 40 层 ViT-g 的第 15 层分化；训练前 1/3 还没有；Tiny/S/B 没有、L 及以上才有。
  - 探针：outlier 的位置分类 / 像素重建更差，图像级线性探针远好于普通 patch、接近 `[CLS]`。
  - 修复：patch embedding 之后拼 $N$ 个 learnable register（默认 4），输出端丢弃。原样接到 DeiT-III、OpenCLIP、DINOv2 三条训练配方上。
  - 消融：1 个 register 就够清掉可见伪影；稠密任务有最优 $N$（约 4）；ImageNet 随 $N$ 继续涨。4 个 register 的 FLOPs 增幅 $<2\%$。
- 核心创新点（vs 已有工作的 delta）：
  - **真增量：不是发明 memory token，而是证明 ViT 已经在偷偷用，并把它隔离。** Memory Transformer（Burtsev 2020）、Perceiver latent、DETR object query 都是**额外 token**。作者自己写得很清楚：这些 token **不注入信息、输出也不用**，只是把已有行为从 patch 里搬出去。这是诊断论文，不是架构论文。
  - **换皮风险低。** 方法本身一行能写完，价值在现象刻画（norm、层、尺度、冗余位置、局部 vs 全局探针）和**三条监督配方都修得掉**。
  - **对 LTX-2 的错位：** Darcet 的 `[reg]` **末端丢弃**，不进入下游表示；LTX-2 的 thinking tokens **保留并经 cross-attn 条件化 DiT**。前者是**别污染 patch**，后者是**多准备几个条件 token**。机制同构（可学习 extra token 当全局缓冲），接口语义相反。
  - OpenCLIP+reg 的 LOST 略降（38.8→37.1），作者在附录说 CLIP 特征上 LOST 对伪影不敏感。等于 ****清伪影 ⇒ 发现变好**不是全称命题**，取决于下游怎么用 gram / key。

- **结论：** 创新在现象与隔离，不在算子；能支撑**ViT 需要寄存器**；撑不住**extra token 应当作为生成条件输出**。

## 4. Related Work（相关工作）

- 关键相关工作：
  - 特征骨干：AlexNet → ResNet/DETR、DeiT-III、CLIP、MAE、DINO / iBOT / DINOv2。
  - extra token：BERT `[SEP]`/`[CLS]`/`[MASK]`、AdaTape、DETR/YOLOS/ViDT query、Perceiver latent、Memory Transformer、Recurrent Memory、Sandler et al. 视觉微调 memory（跨任务迁移差）。
  - attention 可视化：DINO、TOAST、改层结构、可学习 pooling 产 `[CLS]`。
- 本文定位与区别：
  - 相对 Memory Transformer：NLP 翻译涨点 vs 视觉预训练隔离伪影；相对 Sandler：预训练而非微调。定位清楚。
  - 相对 DINO**干净 attention**：把 DINO 标成例外而不是目标，这是有判断的。
- 缺失的重要引用（如有）：
  - 2023 同期 / 稍早的 **attention sink**（StreamingLLM, Xiao et al. 2023）与高范数 sink token 几乎同构，LTX-2 初稿甚至注释过 Xiao 2023，正式引用却换成了 Darcet。视觉论文不引 LLM sink 情有可原，但**寄存器吸收多余 attention**这条物理图像更完整。
  - Slot Attention（Locatello 2020）只在定性图里点了一下，没有把 register 的自然分化做成和 slot 的定量对照。
  - Register 清伪影之后，和 **token merging / EViT** 等**处理冗余 patch**路线几乎没有对话——那些工作是丢掉冗余，本文是改用途。

- **结论：** 视觉侧前驱引得齐；和 LM attention sink / token pruning 的概念桥留给读者。

## 5. Citation / Seed Potential（高引潜力）

- 判断：
  **已经是 seed。** ICLR 2024 outstanding，DINOv2 权重默认 `+reg`，后续几乎所有**现代 ViT 特征图**讨论都会引这篇。
- 依据：
  - **问题普适、修法可迁移、复现成本低：** 加几个 token，官方 DINOv2 仓库可对。
  - **Timing：** 正好卡在 DINOv2 开源、稠密任务开始用冻结 ViT 当骨干的窗口。
  - **可被误引：** 生成 / VLM 论文常把它缩写成**extra token = 全局信息**，丢掉**必须从 patch 里搬出去、输出还要丢掉**这个约束。LTX-2 第 161 行就是这种缩写。

- **结论：** 高引已成事实；引用质量取决于读者有没有读到**隔离而非增补输出**。

## 6. Future Work & Improvements（改进空间）

- 局限性：
  - 未能说清 **哪条训练要素** 触发 outlier（规模、时长、配方三者缠在一起；OpenCLIP/DeiT 在 B 就有，DINOv2 要 L）。
  - 线性探针**outlier 含全局信息**是相关不是因果：高范数本身就更好线性可分。
  - 定性里 register 出现 slot 式分化，但**从未要求**——也就是 **不可控、不可当检测头用**。
  - 只验证图像 ViT；视频 / 多模态 / 扩散骨干是否同样回收 token，原文没做。
- 可做的未来工作 / 研究机会：
  - **跟进性小改：** 自适应 $N$、对 register 加多样性正则、推理期动态开关。
  - **值得做的：** 把 register 当可解码的全局槽（物体级），接到发现 / 跟踪；在 DiT / 双流音视频里复现**高范数 token 是否出现在低信息时空块**——这才是 LTX-2 thinking tokens 该做、却没做的诊断。
  - **不要做的：** 再发一篇**我们在 X 上加了 register，attention 变好看**。

- **结论：** 诊断已经够用；下一跳是**寄存器里到底存了什么、生成模型要不要输出它**，不是再加几个 token。

## Overall Verdict（总评）

- 是否值得深读：
  **值得。** 短、实验干净、把一个被当成 DINOv2 瑕疵的现象升级成 ViT 的一般行为。读 LTX-2 thinking tokens 之前应先钉死：Darcet 的 register **不进入下游表示**。
- 最大亮点 / 最大隐患：
  亮点是**模型已在用寄存器，不给位子就抢 patch**这条机制假说，和跨三种训练配方的修复。隐患是后续引用把它加成**多几个 token 就能带全局 / 缺失语义**，和原文的隔离目标相反。
