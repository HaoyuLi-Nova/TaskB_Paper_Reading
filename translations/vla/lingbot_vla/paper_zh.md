# 一种实用的 VLA 基础模型（A Pragmatic VLA Foundation Model）

**作者：** Wei Wu$^{*}$、Fan Lu$^{*}$、Yunnan Wang$^{*}$、Shuai Yang$^{*}$、Shi Liu$^{*}$、Fangjing Wang$^{*}$、Qian Zhu、He Sun、Yong Wang、Shuailei Ma、Yiyu Ren、Kejia Zhang、Hui Yu、Jingmei Zhao、Shuai Zhou、Zhenqi Qiu、Houlong Xiong、Ziyu Wang、Zechen Wang、Ran Cheng、Yong-Lu Li、Yongtao Huang、Xing Zhu、Yujun Shen、Kecheng Zheng$^{\dagger}$
（$^{*}$同等贡献；$^{\dagger}$项目负责人）
**arXiv：** [2601.18692](https://arxiv.org/abs/2601.18692)
**项目页：** https://technology.robbyant.com/lingbot-vla
**代码：** https://github.com/robbyant/lingbot-vla
**模型：** https://huggingface.co/collections/robbyant/lingbot-vla
**原文 TeX：** [`arxiv/vla/lingbot_vla/extracted/main.tex`](../../../arxiv/vla/lingbot_vla/extracted/main.tex)
**插图来源：** TeX 资源 `extracted/figures/`（PDF 经 `scripts/figures_to_png.py` 原地转为 PNG）

---

## 摘要

视觉-语言-动作（Vision-Language-Action, VLA）基础模型在机器人操作中具有巨大潜力。一个合格的 VLA 基础模型应能在任务与平台间忠实泛化，同时保证成本效率（例如适配所需的数据量与 GPU 小时数）。

为此，我们基于 9 种主流双臂机器人构型、约 **20,000 小时**真实世界数据，开发了 **LingBot-VLA**。在 4 个机器人平台上、每平台 100 项任务、每任务 130 条后训练 episode 的系统评估中，我们的模型相对竞品表现出明显优势，展现出**强性能**与**广泛泛化能力**。

我们还构建了**高效**代码库：在 8-GPU 训练配置下达到 **261 samples/s** 的吞吐，相对现有 VLA 导向代码库（取决于所依赖的 VLM 基座）实现约 **1.5–2.8×** 加速。上述特性使模型适合真实部署。

为推动机器人学习领域发展，我们开放代码、基座模型与评测数据，重点支持更具挑战性的任务，并促进健全的评测标准。

---

## 1. 引言

视觉-语言-动作（VLA）基础模型已成为使机器人按自然语言指令执行多样化操作任务的有前景方法。经大规模预训练，这些模型可获得能快速适配不同任务与机器人平台的可泛化技能。

尽管进展显著，关于真实机器人性能如何随预训练数据规模扩大而扩展，仍缺少全面的实证研究；社区也缺少能在海量数据上高效开展扩展性评测的高度优化训练代码库。因此，真实世界设定下亟待回答的基本问题是：***VLA 模型如何随大规模真实机器人数据真正扩展？***

理解 VLA 的扩展行为对机器人学习至关重要，尤其是在规模庞大、来源多样的真实世界数据集上。本工作对 VLA 预训练阶段**成功率如何随数据量与多样性扩展**做了系统实证。将预训练数据从 3,000 小时扩展到 20,000 小时后，下游成功率持续、显著提升；即使在 20,000 小时处，扩展趋势仍未见饱和，表明 VLA 性能仍可从更大的数据量中受益。这些结果为真实世界机器人学习中的有利扩展性质提供了首批实证证据，对未来 VLA 开发与大规模数据策展具有关键启示。

扩展分析揭示了有利的性能趋势；要将这些认识转化为可靠、可部署的系统，还需要在真实机器人平台上进行大规模严格评测。借助 GM-100（含 100 项精心设计任务）我们在 **4 个机器人平台**上开展系统评估，每个 embodiment 每任务 130 条 episode。通过强调任务多样性与多平台一致性，我们的评测框架为健全的 VLA 评测提供了新的标准选择。

本文提出 **LingBot-VLA**：在 9 个机器人平台上、约 20,000 小时真实操作数据上预训练的实用 VLA 基础模型。在全面评测上的系统评估表明，LingBot-VLA 相对现有方法达到最先进性能与出色泛化。除模型能力外，我们强调大规模机器人学习需要高计算效率；为此开发了优化代码库，在 8-GPU 集群上达到 **261 samples/s** 吞吐，显著缩短训练周期、降低计算开销与总体成本。结合优异性能、广泛泛化与计算效率，LingBot-VLA 适用于真实机器人应用。为促进社区进步，我们开放代码、基座模型与评测数据。

![图1：LingBot-VLA 概览。我们在真实世界采集的双臂机器人数据上扩展预训练；LingBot-VLA 可轻松、高效迁移到下游任务。此外，我们在三种机器人 embodiment 上开展系统评估，展示模型的明显优势。](../../../arxiv/vla/lingbot_vla/extracted/figures/Teaser.png)

---

## 2. 相关工作

### 2.1 视觉-语言-动作模型

**基础 VLA。** 视觉-语言-动作基础模型通常采用强大的预训练视觉-语言模型（VLM）作为语义骨干，并配合基于扩散的动作头。近期 VLA 基础模型（$\pi_0$、$\pi_{0.5}$、GR00T N1.6、Gemini Robotics、WALL-OSS、Galaxea G0、Magma、GR-3 等）在更大规模、更多样数据预训练后，展现出更强的多任务执行与多 embodiment 适配能力。与先前 VLA 基础模型所用数据集不同，我们的模型在约 **20,000 小时**多 embodiment 数据上预训练；该大规模、高行为多样性语料显著增强了模型在各类机器人操作任务上的泛化能力。

**空间 VLA。** 传统 VLA 擅长语义理解，但在复杂空间操作所需的精确几何推理与深度感知上常显不足。为此，SpatialVLA、Spatial Forcing、InternVLA-M1、OmniVLA、Magma、Gemini Robotics 1.5、GeoVLA 等工作将空间表示融入 VLA 框架：部分研究强化 VLM 在具身场景中的空间感知，以增强下游空间操作；另一些在 VLA 训练阶段显式或隐式引入深度信息。Spatial Forcing 采用简洁的对齐策略，迫使 VLA 视觉嵌入与空间表示融合，显著改善空间理解。

### 2.2 机器人策略评测

当前机器人策略评测主要分为两类：**仿真**（LIBERO、CALVIN、SimplerEnv、RoboCasa、RoboTwin 2.0 等）与**真实 embodiment**（RoboChallenge、RoboArena 等）。仿真评测可低成本、大规模并行测试；但其理想化物理模型往往无法完全代表真实物理世界的复杂性。真实世界评测效率常受硬件并行规模瓶颈限制，因此多数先前 VLA 研究仅在少量任务、少数方法间比较。为更全面评估策略的真实性能，本工作在 **三个不同的机器人平台**上、每平台 100 项任务开展评估，并分析主流 VLA 如何适应真实场景多样性。

### 2.3 高效 VLA 训练

VLA 的快速迭代推动了专用训练基础设施的发展。OpenPI 提供支持 JAX/PyTorch 的通用框架；StarVLA 提供模块化、用户友好的代码库，优化 VLA 与 VLM 共训练；Dexbotic 则统一并高效化 VLA 从数据摄入到部署的全生命周期。尽管如此，在多节点集群上训练大规模 VLA 仍因数据 I/O 瓶颈与通信开销而具有挑战。我们提出 LingBot-VLA **高性能开源代码库**，在数据加载、分布式训练策略与算子级加速上实现系统性优化，全面提升吞吐与可扩展性。

---

## 3. 预训练数据集

![图2：LingBot-VLA 所用预训练数据集可视化。](../../../arxiv/vla/lingbot_vla/extracted/figures/data/data_mask.png)

### 3.1 数据采集

预训练数据集基于 **9 种主流双臂机器人 embodiment** 的大规模遥操作数据（见图 2），各平台如下：

- **AgiBot G1：** 两条 7-DoF 臂 + 三路 RGB-D 相机；通过 VR 遥操作采集。
- **AgileX：** 三路相机 + 两条 6-DoF 臂；遥操作时使用同构臂控制。
- **Galaxea R1Lite：** 两条 6-DoF 臂 + 一路立体相机 + 两路腕部相机。
- **Galaxea R1Pro：** 两条 7-DoF 臂 + 一路立体相机 + 两路腕部相机。
- **Realman Rs-02：** 三路相机；16 维配置/动作空间（双 7-DoF 臂 + 双平行夹爪）。
- **Leju KUAVO 4 Pro：** 双足人形；双 7-DoF 臂 + 双平行夹爪 + 头载相机 + 双腕相机。
- **Qinglong：** 人形；双 7-DoF 臂 + 头载与双腕共三路相机。
- **ARX Lift2：** 三路相机 + 双 6-DoF 臂。
- **Bimanual Franka：** 双 7-DoF 臂 + 双平行夹爪（16 维动作空间）+ 三路相机。

合计约 **20,000 小时**跨 embodiment 遥操作演示。

### 3.2 数据标注

为获得精确语言指令，我们执行：

1. **视频分段：** 多视角机器人视频由人工按预定义原子动作切分为片段；并去除起止静态帧以减少冗余。
2. **指令标注：** 获得完整运动轨迹与各原子动作片段后，使用 **Qwen3-VL-235B-A22B** 精确标注任务级与子任务级指令（见图 1）。

---

## 4. 模型训练

### 4.1 架构

为利用预训练视觉-语言表示，LingBot-VLA 将预训练 VLM（**Qwen2.5-VL**）与初始化的动作生成模块 **action expert** 结合，采用类似 BAGEL 的 **Mixture-of-Transformers (MoT)** 架构：视觉-语言与动作模态经不同的 Transformer 通路处理，并通过共享自注意力实现逐层统一序列建模。MoT 使 VLM 的高维语义先验在各层持续指导 action expert，同时通过模态专属处理减轻跨模态干扰。架构见图 1：多视角操作图像与任务指令经 VLM 统一编码为多模态条件；机器人本体感知序列（初始状态与 action chunk）输入 action expert 预测动作。我们采用 **Flow Matching** 建模连续动作，实现流畅、平滑的机器人控制。

VLM 与 action expert 通过共享自注意力交互，实现统一的逐层表示。时刻 $t$ 的联合建模序列为观测条件 $\mathbf{O}_t$ 与 action chunk $\mathbf{A}_t$ 的拼接。观测上下文为：

$$
\mathbf{O}_t = [\mathbf{I}_t^1, \mathbf{I}_t^2, \mathbf{I}_t^3, \mathbf{T}_t, \mathbf{s}_t]
$$

包含双臂机器人三视角操作图像 token $\mathbf{I}_t^{1,2,3}$、任务指令 $\mathbf{T}_t$ 与机器人状态 $\mathbf{s}_t$。动作序列为：

$$
\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \dots, \mathbf{a}_{t+T-1}]
$$

其中 $T$ 为 action chunk 长度（预训练阶段设为 **50**）。训练目标是通过条件流匹配刻画 $p(\mathbf{A}_t | \mathbf{O}_t)$。对 flow 时间步 $s \in [0,1]$，在高斯噪声 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 与真值动作 $\mathbf{A}_t$ 之间线性插值得中间动作 $\mathbf{A}_{t,s}=s\mathbf{A}_{t}+(1-s)\epsilon$，条件分布为：

$$
p(\mathbf{A}_{t,s} | \mathbf{A}_t) = \mathcal{N}(s\mathbf{A}_{t}, (1-s)\mathbf{I})
$$

action expert $v_\theta$ 通过最小化 Flow Matching 目标学习条件向量场：

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{s \sim \mathcal{U}[0, 1], \mathbf{A}_t, \epsilon} \left\| v_\theta(\mathbf{A}_{t,s}, \mathbf{O}_t, s) - (\mathbf{A}_t - \epsilon) \right\|^2
$$

目标速度场为 $\mathbf{A}_t - \epsilon$。

遵循 $\pi_0$，对联合序列 $[\mathbf{O}_t, \mathbf{A}_t]$ 使用 **分块因果注意力（blockwise causal attention）**。序列分为三块：$[\mathbf{I}_t^1, \mathbf{I}_t^2, \mathbf{I}_t^3, \mathbf{T}_t]$、$[\mathbf{s}_t]$、$[\mathbf{a}_t, \dots, \mathbf{a}_{t+T-1}]$。块间施加因果掩码：每块 token 只能 attend 自身及前序块；块内为双向注意力。这样 action expert 可利用全部观测，同时避免未来动作 token 向当前观测泄漏。

为显式捕获操作环境的空间感知并增强执行鲁棒性，我们采用视觉蒸馏：对三视角操作图像使用可学习 query $[\mathbf{Q}^1_t, \mathbf{Q}^2_t, \mathbf{Q}^3_t]$，经 VLM 处理后与 **LingBot-Depth** 的深度 token $[\mathbf{D}^1_t, \mathbf{D}^2_t, \mathbf{D}^3_t]$ 对齐，最小化：

$$
\mathcal{L}_{distill} = \mathbb{E}_{\mathbf{Q}_t} \left| \text{Proj}(\mathbf{Q}_t) - \mathbf{D}_t \right|
$$

其中 $\text{Proj}(\cdot)$ 为交叉注意力投影层。该集成将几何信息注入 LingBot-VLA，支持复杂操作任务的精确感知。

### 4.2 训练效率优化

动作数据本质上高频，需建立涵盖分布式训练与算子优化的高效流水线：

**分布式策略：** VLA 参数量适中，但需在显存占用与吞吐之间权衡。我们使用 **FSDP**（ZeRO 的高效 PyTorch 实现）分片优化器状态、参数与梯度。借鉴 VeOmni 的 **HSDP**，为 action expert 模块构建专属分片组，减轻过度分片带来的通信开销。混合精度：归约使用 `float32` 以保证数值稳定，存储与通信使用 `bfloat16`。

**算子级优化：** 多模态融合本质上是稀疏注意力，使用 **FlexAttention** 优化；并通过 `torch.compile` 做算子融合，降低 kernel 启动开销、提升带宽利用率。

---

## 5. 实验

### 5.1 大规模真实世界评测

我们设计严格评测以评估多 embodiment 泛化与真实鲁棒性，包含：(1) **25 台物理机器人**、**4 个商业平台**；(2) **GM-100** 评测：100 项操作任务、39,000 条专家演示；(3) 受控协议下与三种最先进基线对比，共 **22,500 次试验**。

#### 5.1.1 硬件平台

实验覆盖 **AgileX、Agibot G1、Galaxea R1Pro、Leju KUAVO 4 Pro** 四个平台；均为双臂 + 平行夹爪；各配双腕相机 + 头载相机（第一人称视角）。任务均为桌面操作，底盘与腰部固定。

#### 5.1.2 数据采集与处理

![图3：(a) 预训练数据集中的原子动作词云。](../../../arxiv/vla/lingbot_vla/extracted/figures/data/trainset.png)

![图3：(b) 评测集中的原子动作词云。](../../../arxiv/vla/lingbot_vla/extracted/figures/data/testset.png)

每项 GM-100 任务按标准化协议遥操作采集专家演示：

- **轨迹规模：** 每任务每平台 150 条原始轨迹，按完成度、平滑度、协议遵循度保留 top **130** 条用于训练。
- **标准化物体：** 按 GM-100 规格采购，保证跨站点可复现。
- **环境多样性：** 每条轨迹随机化物体位姿，防止空间过拟合。
- **遥操作规范：** (1) 末端与桌面保持间隙；(2) 接触阶段降速；(3) episode 起止视觉状态可区分。
- **自动过滤：** 排除关节静止、约束违反、抖动/不连续、多模态时序不同步等异常。
- **人工复核：** 多视角视频检查，剔除含额外物体或偏离协议的 episode。

词云分析（图 3）显示：测试集中约 **50%** 原子动作不在训练集 top-100 高频动作中，从而保证对泛化能力的严格评估。

#### 5.1.3 评测协议

与 **$\pi_{0.5}$、GR00T N1.6、WALL-OSS** 在严格控制下对比：

- **标准化训练：** 均从公开预训练检查点出发，使用相同后训练流水线；验证过的数据集（每任务 130 条）与一致超参（batch size=256，epochs=20）。
- **严格机-任务配对：** 评测使用与采集相同的物理机器人单元；在同一硬件-任务对上以随机顺序依次测试各模型。
- **受控评测环境：** 与采集一致，随机化物体位姿，任务规格不变。
- **推理与记录：** 每模型每任务-机器人 **15 次试验**；环境恒定；以 rosbag 记录第三人称视角、机器人状态与模型预测，并将开源。

#### 5.1.4 评测指标

- **成功率 SR：** 3 分钟内完成全部任务步骤的试验比例；反映部署可行性。
- **进度分 PS：** 按顺序子任务检查点的部分完成度。例如：6 步**叠碗**完成 1–4 步而失败于第 5 步，得 $\frac{4}{6}\approx 0.67$；用于诊断失败模式。
- **终止条件：** 同一子任务连续失败 3 次，或发生安全事件（碰撞等）；按终止前完成的子任务计分。

报告 100 任务总体 SR/PS，并按机器人类型分层评估跨 embodiment 泛化。

### 5.2 真实世界评测对比

表 1 对比 LingBot-VLA 两变体（含/不含深度）与三种强基线。所有平台上，**LingBot-VLA（不含深度）** 在 SR 与 PS 上显著优于 WALL-OSS 与 GR00T N1.6。引入深度空间信息后，**LingBot-VLA（含深度）** 在三 embodiment 平均 SR 上相对 $\pi_{0.5}$ 提升 **4.28%**，PS 提升 **7.76%**。

GR00T N1.6 在 Agibot G1、AgileX、Leju KUAVO 4 Pro 上表现一般，但在 **Galaxea R1Pro** 上与 $\pi_{0.5}$ 相当——因其预训练大量包含 R1Pro 数据，说明预训练可显著增强结构相似下游任务的性能。完整逐任务结果见附录。

**表 1：GM-100 真实世界评测结果。** SR = 成功率，PS = 进度分。

| 平台 | WALL-OSS SR | WALL-OSS PS | GR00T N1.6 SR | GR00T N1.6 PS | $\pi_{0.5}$ SR | $\pi_{0.5}$ PS | Ours w/o depth SR | Ours w/o depth PS | Ours w/ depth SR | Ours w/ depth PS |
|------|-------------|-------------|---------------|---------------|----------------|----------------|-------------------|-------------------|------------------|------------------|
| Agibot G1（轮式） | 2.99% | 8.75% | 5.23% | 12.63% | 7.77% | 21.98% | **12.82%** | 30.04% | 11.98% | **30.47%** |
| AgileX（轮式） | 2.26% | 8.16% | 3.26% | 10.52% | 17.20% | 34.82% | 15.50% | 36.31% | **18.93%** | **40.36%** |
| Galaxea R1Pro（轮式） | 6.89% | 14.13% | 14.29% | 24.83% | 14.10% | 26.14% | 18.89% | 34.71% | **20.98%** | **35.40%** |
| Leju KUAVO 4 Pro（双足） | 3.26% | 11.75% | 6.45% | 18.66% | 12.91% | 26.35% | **17.59%** | **36.22%** | 15.60% | 34.40% |
| **平均** | 3.85% | 10.70% | 7.31% | 16.66% | 13.00% | 27.32% | 16.20% | 34.32% | **16.87%** | **35.16%** |

### 5.3 仿真评测对比

在 RoboTwin 2.0 的 **50** 项代表任务上评测（表 2）。自预训练检查点出发，在 RoboTwin 数据上微调：干净场景 50 任务 × 50 演示 = 2,500；随机化场景 × 500 演示 = 25,000。随机化包含背景、杂乱、桌面高度、光照等变化。

相对 $\pi_{0.5}$，LingBot-VLA（不含深度）在干净/随机化场景上绝对 SR 分别提升超过 **3.76%** 与 **8.58%**；含深度版本通过可学习 query 深度对齐，分别提升 **5.82%** 与 **9.92%**。逐任务结果见附录表 S9。

**表 2：RoboTwin 2.0 仿真评测（平均 SR）。**

| 场景 | $\pi_{0.5}$ | Ours w/o depth | Ours w/ depth |
|------|-------------|----------------|---------------|
| Clean | 82.74% | 86.50% | **88.56%** |
| Randomized | 76.76% | 85.34% | **86.68%** |

### 5.4 训练吞吐分析

选取 **StarVLA、Dexbotic、OpenPI** 为基线，在 **LIBERO** 上、统一 $\pi$-like 架构对比。我们在自有代码库复现 **Qwen2.5-VL-3B-$\pi$** 与 **PaliGemma-3B-pt-224-$\pi$**；本地 batch size 统一为 32。StarVLA/Dexbotic 默认 ZeRO，我们使用可比的 **FSDP2**；OpenPI 使用 DDP（通信开销更低）。指标：**样本吞吐（samples/s）**。

![图4：(a) Qwen2.5-VL-3B-$\pi$ 模型的训练吞吐分析。](../../../arxiv/vla/lingbot_vla/extracted/figures/experiment/infra_scaling/QwenPI.png)

![图4：(b) PaliGemma-3B-pt-224-$\pi$ 模型的训练吞吐分析。](../../../arxiv/vla/lingbot_vla/extracted/figures/experiment/infra_scaling/PaliGemmaPI.png)

图 4 显示：两种 VLM 设定下我们的代码库均最快；在 8/16/32/128/256 GPU 配置下吞吐紧密跟随理论线性扩展上限。8-GPU 集群上达到 **261 samples/s**（相对所依赖 VLM 不同，较基线约 1.5–2.8×）。

### 5.5 消融实验

#### 5.5.1 扩展实验

在评测子集 **25** 项代表任务上评估预训练数据扩展规律。图 5 显示：预训练时长从 **3,000** 增至 **20,000** 小时，PS 与 SR 持续上升，且 **20,000 小时处未见饱和**；Agibot G1、AgileX、Galaxea R1Pro 三条曲线与聚合趋势一致，说明扩展规律稳健且非单平台特有。

![图5：(a) 进度分（PS）随数据规模的扩展行为。](../../../arxiv/vla/lingbot_vla/extracted/figures/experiment/pre_training_data_scaling_law/Aggregated_Progress_Rate.png)

![图5：(b) 成功率（SR）随数据规模的扩展行为。](../../../arxiv/vla/lingbot_vla/extracted/figures/experiment/pre_training_data_scaling_law/Aggregated_Success_Rate.png)

#### 5.5.2 数据效率分析

在 Agibot G1 上，从 GM-100 选取 **8** 项代表任务做数据高效后训练（图 6）。仅 **80** 条演示/任务时，LingBot-VLA 即在 PS 与 SR 上超过使用完整 130 条演示的 $\pi_{0.5}$；随后训练数据增加，差距显著扩大，展现出更优的数据效率与可扩展性。

![图6：LingBot-VLA 后训练的数据效率。](../../../arxiv/vla/lingbot_vla/extracted/figures/experiment/post_training_data_scaling_law/Dual_Axis_Success_Progress.png)

---

## 6. 结论

我们提出 **LingBot-VLA**：通过大规模真实世界数据与优化代码库，在泛化能力与训练效率上取得突破。100 项任务的全面评测显示模型相对竞品具有明显优势。为促进开放科学，我们发布代码、模型与评测数据。未来工作将整合单臂与移动机器人数据，扩展模型能力，支持无约束环境中更多样、可移动的操作能力。

**致谢。** 感谢数据、评测、训练基础设施、机器人硬件与软件方面的众多贡献者；并感谢 Galaxea Team、AgileX Robotics、乐聚（深圳）机器人技术有限公司在数据采集与评测上的支持。

---

## 附录 A：实验详细结果

本附录给出正文聚合均值所依据的逐任务明细。GM-100 真实世界评测在 **AgileX、Agibot G1、Galaxea R1Pro、Leju KUAVO 4 Pro** 四个平台上各拆为两张表（Part I/II），共 **8 张表**；RoboTwin 2.0 仿真为 **1 张表**。完整 LaTeX 源见 [`sections/7_appendix.tex`](../../../arxiv/vla/lingbot_vla/extracted/sections/7_appendix.tex)；以下给出 RoboTwin 逐任务结果（对应正文 §5.3 与表 2 明细）。

**表 S9：RoboTwin 2.0 仿真评测（Clean / Randomized 设定下的 SR）。**

| 仿真任务 | $\pi_{0.5}$ Clean | $\pi_{0.5}$ Rand. | Ours w/o depth Clean | Ours w/o depth Rand. | Ours w/ depth Clean | Ours w/ depth Rand. |
|----------|-------------------|-------------------|----------------------|----------------------|---------------------|---------------------|
| Adjust Bottle | 100% | 99% | 100% | 100% | 100% | 100% |
| Beat Block Hammer | 96% | 93% | 87% | 91% | 92% | 89% |
| Blocks Ranking Rgb | 92% | 85% | 92% | 91% | 92% | 91% |
| Blocks Ranking Size | 49% | 26% | 66% | 73% | 76% | 70% |
| Click Alarmclock | 98% | 89% | 93% | 26% | 97% | 43% |
| Click Bell | 99% | 66% | 32% | 19% | 43% | 36% |
| Dump Bin Bigbin | 92% | 97% | 97% | 92% | 97% | 97% |
| Grab Roller | 100% | 100% | 100% | 99% | 100% | 100% |
| Handover Block | 66% | 57% | 80% | 83% | 83% | 95% |
| Handover Mic | 98% | 97% | 94% | 98% | 94% | 99% |
| Hanging Mug | 18% | 17% | 32% | 27% | 34% | 53% |
| Lift Pot | 96% | 85% | 100% | 99% | 100% | 100% |
| Move Can Pot | 51% | 55% | 79% | 84% | 89% | 87% |
| Move Pillbottle Pad | 84% | 61% | 93% | 94% | 92% | 90% |
| Move Playingcard Away | 96% | 84% | 96% | 99% | 98% | 100% |
| Move Stapler Pad | 56% | 42% | 74% | 49% | 74% | 48% |
| Open Laptop | 90% | 96% | 96% | 96% | 98% | 96% |
| Open Microwave | 34% | 77% | 91% | 75% | 91% | 92% |
| Pick Diverse Bottles | 81% | 71% | 79% | 86% | 88% | 85% |
| Pick Dual Bottles | 93% | 63% | 82% | 95% | 99% | 90% |
| Place A2b Left | 87% | 82% | 86% | 83% | 89% | 85% |
| Place A2b Right | 87% | 84% | 74% | 77% | 80% | 80% |
| Place Bread Basket | 77% | 64% | 92% | 93% | 95% | 93% |
| Place Bread Skillet | 85% | 66% | 90% | 89% | 90% | 92% |
| Place Burger Fries | 94% | 87% | 95% | 96% | 98% | 94% |
| Place Can Basket | 62% | 62% | 68% | 78% | 75% | 72% |
| Place Cans Plasticbox | 94% | 84% | 97% | 100% | 100% | 98% |
| Place Container Plate | 99% | 95% | 99% | 99% | 99% | 100% |
| Place Dual Shoes | 75% | 75% | 80% | 83% | 87% | 86% |
| Place Empty Cup | 100% | 99% | 100% | 100% | 100% | 100% |
| Place Fan | 87% | 85% | 91% | 79% | 92% | 87% |
| Place Mouse Pad | 60% | 39% | 82% | 78% | 86% | 79% |
| Place Object Basket | 80% | 76% | 90% | 91% | 90% | 88% |
| Place Object Scale | 86% | 80% | 84% | 90% | 90% | 88% |
| Place Object Stand | 91% | 85% | 97% | 93% | 93% | 88% |
| Place Phone Stand | 81% | 81% | 92% | 93% | 90% | 87% |
| Place Shoe | 92% | 93% | 99% | 94% | 99% | 99% |
| Press Stapler | 87% | 83% | 90% | 88% | 86% | 93% |
| Put Bottles Dustbin | 84% | 79% | 88% | 92% | 92% | 93% |
| Put Object Cabinet | 80% | 79% | 92% | 86% | 85% | 88% |
| Rotate Qrcode | 89% | 87% | 93% | 84% | 86% | 82% |
| Scan Object | 72% | 65% | 91% | 97% | 92% | 96% |
| Shake Bottle Horizontally | 99% | 99% | 100% | 100% | 99% | 98% |
| Shake Bottle | 99% | 97% | 99% | 100% | 100% | 99% |
| Stack Blocks Three | 91% | 76% | 92% | 99% | 96% | 95% |
| Stack Blocks Two | 97% | 100% | 100% | 100% | 100% | 99% |
| Stack Bowls Three | 77% | 71% | 72% | 83% | 71% | 77% |
| Stack Bowls Two | 95% | 96% | 92% | 95% | 90% | 97% |
| Stamp Seal | 79% | 55% | 76% | 86% | 74% | 77% |
| Turn Switch | 62% | 54% | 61% | 65% | 67% | 63% |

> **GM-100 逐任务表（表 S1–S8）：** 每表约 50 个任务编号（#001–#107 等），列结构与表 1 相同。因篇幅，此处不逐行誊录；请查阅原文 PDF 附录或 TeX 源 [`sections/7_appendix.tex`](../../../arxiv/vla/lingbot_vla/extracted/sections/7_appendix.tex)。
