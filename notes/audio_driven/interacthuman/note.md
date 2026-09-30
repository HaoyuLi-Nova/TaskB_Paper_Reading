# InterActHuman：动机、创新点与核心技术

- **Authors / Year / Venue**: Zhenzhi Wang, Jiaqi Yang, Jianwen Jiang（通讯）, Chao Liang, Gaojie Lin, Zerong Zheng, Ceyuan Yang, Yuan Zhang, Mingyuan Gao, Dahua Lin / 2026 / ICLR 2026（arXiv:2506.09984）
- **Link**: [https://arxiv.org/abs/2506.09984](https://arxiv.org/abs/2506.09984)
- **Project**: [https://zhenzhiwang.github.io/interacthuman/](https://zhenzhiwang.github.io/interacthuman/)
- **原文 TeX**: [`arxiv/audio_driven/interacthuman/extracted/iclr2026_conference.tex`](../../../arxiv/audio_driven/interacthuman/extracted/iclr2026_conference.tex)
- **官方代码（本仓库）**: [`code/audio_driven/interacthuman/`](../../../code/audio_driven/interacthuman/)（Wan2.1 + OmniAvatar 的 demonstration re-implementation，**不是** ByteDance 内部原实现；无 InterActHuman checkpoint）
- **批判精读**: 同目录 [`critique.md`](critique.md) · **问答**: [`QA.md`](QA.md)
- **一句话**: 在 MM-DiT 人体动画底座上，用去噪过程中逐步缓存的人物软掩码，把 wav2vec 音频只灌进这个人的时空区域；不做重叠说话、不改音色。

框架图：

![framework](../../../arxiv/audio_driven/interacthuman/extracted/figs/framework.png)

---

## 1. 动机

单人音频驱动人体动画（OmniHuman、FantasyTalking 等）默认：**所有条件描述的是画面里那一个主体**，参考图、文本、音频都对整幅 latent **全局注入**。

一旦同画面出现两个人、人或物，全局注入立刻错位：

- 参考外观还能靠 self-attn 或多概念定制（Video-Alchemist、ConceptMaster、Phantom）看起来像；
- **音频是局部模态**：这段语音只该驱动当前说话人的嘴和表情，不该驱动背景、听众、手里的杯子。

作者把条件分成两类：

| 模态                                   | 该打到哪         | 现有做法           |
| -------------------------------------- | ---------------- | ------------------ |
| 参考图、文本                           | 全局（整段视频） | 多概念定制已经能做 |
| 音频（以及任何只属于某个人的信号） | 该人的时空足迹   | 仍按视频级全局注入 |

痛点不是不会生成两人，而是 **没有成片就没有区域、没有区域就不能把音频注对地方**——后面第 4 节的 chicken-and-egg。这和 Bind-Your-Avatar / MultiTalk / HarmonizedVA 面对的是同一类**空间绑定**问题。

范围边界（读的时候要看清）：

- 目标停在这段声驱动对的那个人的口型 / 表情 / 身体；
- 评测默认 **两人一轮流说话**（一人 meaningful audio，一人 muted），**不是**重叠说话；
- **不改音色**，也不是音视频联合生成。

---

## 2. 创新点

相对换底座 / 换对时 / 再加一个 ID embedding这类组合，本文真正增量在 **layout 怎么来、音频怎么按 layout 注**。

### 2.1 显式 layout，而不是让注意力自己找人

多概念定制（Video-Alchemist、ConceptMaster）靠 feature fusion / attention **隐式**学哪张参考图对应画面哪一块。人体动画里音频必须对准嘴，隐式对应不够。InterActHuman 在每个 DiT block 上挂一个 **mask predictor**：用当前去噪视频 hidden 去 cross-attend 每张参考图，显式吐出该人的时空软掩码 $m_i$。

### 2.2 把 chicken-and-egg 拆成第 $k$ 步 mask 指导第 $k{+}1$ 步音频

这是全文最关键的调度，不是新算子。推理时没有 GT 视频，不能先分割再生成。利用扩散本身的多步：

- 第 $k$ 步：用当前噪声 latent + 参考图预测 mask，写入 cache；
- 第 $k{+}1$ 步：用 cache 里的 mask 做局部音频注入。

前 10 步 mask 不可靠，**关掉**按 mask 注入，避免早期错误把其他人压掉。附录消融：有 cache Sync-D $6.921$，无 cache $11.046$——跨步空间引导不是装饰。

### 2.3 局部音频 = 说话声与静音的软混合，而不是硬切

身份 $i$ 的音频 cross-attn 出说话特征 $\mathbf{p}_i$ 和静音特征 $\mathbf{p}_i^{\mathrm{mute}}$，按 mask 置信度混合：

$\mathbf{h}^{v} \leftarrow \mathbf{h}^{v} + m_i \odot \mathbf{p}_i + (1-m_i)\odot \mathbf{p}_i^{\mathrm{mute}}.$

边界软过渡，避免硬 mask 在 VAE 低分辨率格子上切出一圈伪影。听众输入 muted 轨，说话人输入真实语音。

### 2.4 不是新故事的部分

- 底座：内部是 Seaweed 系 MM-DiT + 3D VAE + flow matching；开源 demo 换 Wan2.1。
- 单人音频：先按 OmniHuman 做 **无 mask 的 wav2vec cross-attn** 与 mixed-conditions 预训练。
- 参考图：VAE 编码后与 noisy latent **拼进同一条序列，走原 DiT self-attn**，不另训外观网络。
- 数据：OpenHumanVid + Grounding-SAM2 + Qwen/Gemini caption，规模大但管线常规。

和 MultiTalk 的差：MultiTalk 绑定靠 amap → L-RoPE（软身份轴）；本文绑定靠可监督 mask 挡音频 cross-attn。一个在旋转里做人，一个在注意力支撑集上做人。

和 Bind-Your-Avatar 的差：Bind 是 **同一次 forward 里**用 face-attn 的 Q/K 长 3D mask，训练 teacher-forcing 用 SAM2 GT 注入；本文是 **跨去噪步** cache mask，且正文写局部音频 **主要在推理时**才按 mask 注入。Bind 显式支持重叠说话（$\mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\mathbf{A}^{\mathrm{cv}}$）；本文评测是轮流对话。

---

## 3. 问题形式与骨干数据流

### 3.1 任务

给定

- 文本 $T$（场景 / 动作 / 互动）；
- $N$ 张概念参考图 $\{X_i\}_{i=1}^{N}$（人头、半身、全身、物体、场景都可以）；
- $N$ 路身份音频 $\{Y_i\}_{i=1}^{N}$，

生成视频 $V$：每个人外观跟 $X_i$，口型跟 $Y_i$，构图跟 $T$。支持 **无首帧 I2V**（只要参考外观）、可选首帧、2–3 人对话、人–物组合。

### 3.2 底座

- **VAE**：时空压缩 $(4,8,8)$，latent 16 通道（附录 OmniHuman 底座）。
- **DiT**：MM-DiT 双流（视频 token × 文本 token），AdaSingle 做 timestep，深层 2/3 FFN 权重共享。
- **目标**：flow matching。线性路径 $z_t=(1-t)z_0+t\epsilon$，网络预测速度 $v_\Theta(z_t,t,c_{\mathrm{img}},c_{\mathrm{audio}})$，监督

$\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{t,z_0,\epsilon}\big\|v_\Theta-(z_1-z_0)\big\|_2^2.$

### 3.3 三条条件怎么进

| 模态   | 进法                                                                                                               | 是否局部             |
| ------ | ------------------------------------------------------------------------------------------------------------------ | -------------------- |
| 文本   | MM-DiT 文本流；推理用 Qwen2.5-VL 把每张参考图扩写成细 caption 再拼回$T$                                          | 全局                 |
| 参考图 | 与视频同一套 VAE → token 序列，**拼到 noisy video tokens 后面**，每个 DiT block 的 self-attn 里和视频互相看 | 全局外观、隐式占位   |
| 音频   | wav2vec 2.0，**不再加一层音频 Transformer**；预训练时每块 MM-DiT 之后做音频 cross-attn；多人时用 mask 把门   | 预训练全局，推理局部 |

参考图这条 **零额外参数**：复用 DiT 自己的 QKV。self-attn 序列长度从视频格子变成视频格子 + $N$ 张参考图的格子，代价是注意力平方项 $(1+n/F)^2$（$F$ 为视频时间格；表里 109 帧 → 28 个 VAE latent）。

音频时间：每个视频 latent 只看 wav2vec 的 **$\pm 5$ token 窗**，空间上不限制。

---

## 4. 核心技术 I：chicken-and-egg 与跨步 mask cache

### 4.1 循环依赖

若先有成片、再分割、再按区域注音频：

1. 推理时没有成片 → 不能得到每人的时空足迹；
2. 没有足迹 → 不能把 $Y_i$ 只打到人 $i$；
3. 音频打不对 → 成片更不可能按人说话。

同一步里先预测 mask 再立刻用这个 mask 注音频也有问题：当前步 latent 还很噪，mask 不可靠，用它去门控会把错误布局锁死（早期错误抑制其他人——论文原话）。

### 4.2 把循环改成序列

扩散有 $S$ 步（默认 50）。意图中的调度（正文 + 图注，**校正 Algorithm 1 的书写顺序**）：

```text
m_prev ← 0
for k = S downto 1:          # k 大 = 更噪
    参考图 self-attn 注入
    用当前 z_k 与 {X_i} 预测本步 mask m_k
    if k < S_mask:           # 前 10 步关掉；S_mask ≈ S-10 = 40
        用 m_prev 做局部音频注入   # 关键：用上一步，不是本步
    cache: m_prev ← m_k
    flow-matching / sampler 更新 z
```

对应关系：

| 符号                  | 含义                                                                             |
| --------------------- | -------------------------------------------------------------------------------- |
| $S=50$              | 总去噪步                                                                         |
| 前 10 步不用 mask | 循环开头$k=50,\ldots,41$，音频仍全局或干脆不按人门控                           |
| $S_{\mathrm{mask}}$ | Algorithm 里的阈值；与`if k < S_mask` 合起来应是噪声降到一定程度之后才开门 |
| $m^{\mathrm{prev}}$ | 上一步各层平均后的人物软掩码                                                     |

Algorithm 1 把cache ← 本步 mask写在音频注入**之前**，随后注入又用的是本步 $m_i$。这与正文第 $k$ 步指导第 $k{+}1$ 步、图注$t{-}1$ 的 mask 指导 $t$ 的音频 CA矛盾。**读方法时按正文/图注，不要按伪代码的赋值顺序。** 附录 Table 8 的 cache 消融只在跨步用旧 mask解释下才说得通。

### 4.3 为什么前 10 步必须关

附录 IoU（低运动 / 高运动）：

|                | step 1 | step 10 | step 20 | step 50 |
| -------------- | ------ | ------- | ------- | ------- |
| 低运动 combine | 0.376  | 0.738   | 0.881   | 0.956   |
| 高运动 combine | 0.741  | 0.916   | 0.932   | 0.937   |

step 1 的 mask 不能用。若此时按 mask 把音频只灌进模型瞎猜的那一块，其他人的 token 会长期吃 mute，布局被锁死。先走 10 步全局（或无门控）让各参考图在 self-attn 里占住位置，再开始局部注入。

深层 mask（layer 36）早期 IoU 明显高于浅层（layer 4）；最终用 **最后若干层平均**（正文）或全层平均（Algorithm）。IoU 随步数单调升，用来支撑迭代会收敛——这是对 cache 策略的证据，不是对同一步预测+注入的证据。

### 4.4 和 Bind Intra-Denoise 不是同一个时间轴

|               | InterActHuman                                                 | Bind-Your-Avatar                                           |
| ------------- | ------------------------------------------------------------- | ---------------------------------------------------------- |
| mask 何时更新 | **去噪步与去噪步之间**（50 步外循环）                   | **同一次 DiT forward 的层与层之间**（Intra-Denoise） |
| 训练时谁把门  | 正文：局部音频**主要在推理**；训练以 focal 监督 mask 头 | 训练用 SAM2 GT teacher-forcing 注入，router detach         |
| 早期噪声      | 前 10**步**不用 mask                                    | 靠 GT mask + 噪声增强，没有前 10 步关掉                |

两者都在解没有成片就没有区域，但 InterActHuman 赌的是 **ODE 轨迹上 mask 会越来越准**，Bind 赌的是 **当前层视觉 token 已经能对上脸**。

---

## 5. 核心技术 II：Mask Predictor

### 5.1 每层在算什么

每个 DiT block $\ell=1,\ldots,L$、每个参考 $i$ 挂一个轻量头（参数跨参考共享，约 56M vs 7B DiT）：

1. 线性把视频 hidden $\mathbf{h}^{v}$ 和参考 hidden $\mathbf{h}^{r}_i$ 投到 $Q,K,V$；
2. LayerNorm + **3D RoPE**（给格子时空位置）；
3. 视频 token 对**这一张**参考做 cross-attn：

$\mathbf{p}^{(l)}_i=\operatorname{softmax}\Big(\frac{\mathbf{Q}^{v}(\mathbf{K}^{r}_i)^{\top}}{\sqrt{d}}\Big)\mathbf{V}^{r}_i;$

4. 两层 MLP + sigmoid → 该层软掩码 $m^{(l)}_i\in[0,1]^{T\times H'\times W'}$（正文写成 $[0,1]^{T}$，维数是笔误；Algorithm 里 $m_i$ 与 $\mathbf{p}_i$ 做逐元素乘，必须是时空格）。

最后对层平均：

$m_i=\frac{1}{L}\sum_{l}m^{(l)}_i.$

这不是从 RGB 另训分割网络，而是 **在已经和参考图做过 self-attn 的 DiT 特征上**问：这个视频格子和参考 $i$ 有多像？

### 5.2 监督：要整个人，不要只跟参考图裁切走

GT 来自 Grounding-SAM2，query=`person`，得到逐帧人物（及物体）mask。监督目标是 **完整人体区域**，哪怕 $X_i$ 只是一张大头照。这样：

- mask 头不必在半身 / 头 / 全身之间切换语义；
- 音频可以灌进头肩乃至身体，而不是只打在脸那么小的一块（固定框的失败模式之一）。

损失是 **focal loss**（$\alpha=0.25,\gamma=2$），不是 BCE。理由：前景–背景极不平衡，SAM2 偶发烂 mask。再加 **frame-alignment flag**：SAM2 置信低于 $0.5$ 的帧不进 mask loss（flow matching 照算）。扩散损失与 focal **等权 $1:1$**。

随机把参考图裁成头 / 全身 / 衣服，face:full-body $=0.7:0.3$，减轻 DiT 把参考图 copy-paste 进画面、姿势永远不变。

### 5.3 失败模式（论文自己给的）

- 高度重叠：粉衣小女孩和黄猫叠在一起时，mask 会吞进错误物体、漏掉该排除的人；
- VAE 下采样太大，mask 格子粗，边界不准——这也是为什么注入要用软混合而不是 0/1 硬切。

---

## 6. 核心技术 III：局部音频注入

### 6.1 预训练：先会全局听话

多人之前，先按 OmniHuman 做单人：在每个 DiT block 的 MM-DiT 之后加 wav2vec cross-attn，**不加 mask**。混合条件训练（强条件任务少采样、弱条件任务多采样）：先参考图注入，再音频。这一步结束，模型已经会听到什么就动嘴，只是还不知道 **哪张嘴**。

### 6.2 推理：按上一步 mask 把门

对每个身份 $i$，同一套音频 CA 跑两遍（或 K/V 准备两套）：

- 说话：$K_i,V_i \leftarrow Y_i$（wav2vec）；
- 静音：$K_i^{\mathrm{mute}},V_i^{\mathrm{mute}} \leftarrow Y_i^{\mathrm{mute}}$（静音 / 零音频）。

$\mathbf{p}_i=\operatorname{softmax}\Big(\frac{\mathbf{Q}^{v}K_i^{\top}}{\sqrt{d}}\Big)V_i,\quad \mathbf{p}_i^{\mathrm{mute}}=\operatorname{softmax}\Big(\frac{\mathbf{Q}^{v}(K_i^{\mathrm{mute}})^{\top}}{\sqrt{d}}\Big)V_i^{\mathrm{mute}}.$

$\mathbf{h}^{v}\leftarrow\mathbf{h}^{v}+m_i\odot\mathbf{p}_i+(1-m_i)\odot\mathbf{p}_i^{\mathrm{mute}}.$

格子在人 $i$ 里面：$m_i\approx 1$，吃说话特征；在外面：吃静音，**不会跟着 $Y_i$ 动嘴**。$m_i$ 在 0.5 附近的边界格子两种都掺一点。

多人时对 $i=1\ldots N$ **依次加**：人 1 的说话只进 $m_1$ 高的地方；人 2 同理。轮流对话 = 同一时刻只有一路 $Y_i$ 是 meaningful，其余 muted。重叠说话在公式上也能写（两路 $m_i$ 同时非零），但测试集 **故意做成一人说一人听**，没有验证抢同一张嘴的情况。

### 6.3 CFG

共享一套 CFG 打在音频和文本上，scale $6.5$。**只在正分支做 masked 音频**；负分支不加按人门控的音频（或走无条件）。50 步采样。

### 6.4 消融：为什么必须是预测出来的动态 mask

多人测试（Tab. 4；指标越低越好的是 Sync-D / FVD）：

| 变体                     | 做法                                                     | Sync-D↓        | FVD↓          | 失败样子                 |
| ------------------------ | -------------------------------------------------------- | --------------- | -------------- | ------------------------ |
| Global audio             | OmniHuman 式整图灌音频                                   | 9.482           | 33.9           | 所有人都跟着同一路声说话 |
| ID Embedding             | 可学习 ID token 加到参考图+音频上，隐式配对，无 mask | 8.627           | 35.7           | 经常对错人               |
| Fixed Mask               | 用户给静态矩形，且人不能走                               | 7.068           | **40.2** | 口型还行，人一动 FVD 崩  |
| **Predicted Mask** | 本文                                                     | **6.670** | **22.9** | —                       |

ID embedding 的实现（附录）：DETR 式 $N$ 个可学习 query，同一向量加到配对的参考图和音频上，只用 flow matching、**不监督 mask**。作者自己写：这只证明这么实现的隐式匹配不行，不证明隐式一定差。

Fixed mask 的 Sync-D 已经接近预测 mask，但 FVD 最差——静态框在运动上把人钉死。这正是 AnyTalker最大脸框复制全视频、HarmonizedVA 首帧框复制全视频会踩的坑。

---

## 7. 训练、数据、推理配方

### 7.1 训练（内部原模型）

- 10k step，32×A800，FSDP，lr $3\times 10^{-5}$；
- 有效 batch = 8 条视频（4 node × 每 node 2 条）；
- $\mathcal{L}=\mathcal{L}_{\mathrm{FM}}+\mathcal{L}_{\mathrm{focal}}$，等权；
- 参考图随机遮挡（头 / 全身 / 衣）；
- 长视频：sliding window，窗口尾部若干 motion frame 接到下一窗头部（Loopy / EMO 同款）。

正文没有写清：**多人阶段训练时，音频 CA 是否已经用 GT mask 门控。** 只说局部注入primarily at inference。更稳妥的读法：mask 头是辅助分割任务；DiT 去噪在训练时仍主要见全局 / 混合音频；推理才把门插上。这比 Bind 的 teacher-forcing 更省事，也把门控分布的 train/test gap 留得更大。详见 [`critique.md`](critique.md) §3。

### 7.2 数据

- 来源：OpenHumanVid + 自采；丢掉短于 $4$ s、DWPose 检出多于 $3$ 个显著人。
- Caption：Qwen2-VL（从 Gemini-2.0-Pro 蒸馏）密标环境、外观、动作、表情、人–物、人–人；Gemini-2.0-Flash 抽结构化外观短语。
- 空间 GT：Grounding-SAM2，query=`person`；白底前景参考图 + mask 监督。
- 规模：**260 万** video–mask–caption triplet。
- 单人音频底座另有一套：切镜、OCR 去字幕、Q-align 画质/美学、RAFT 滤过猛运动、SyncNet 滤口型，最后 **2000 小时** 音频驱动数据。

### 7.3 推理

1. Qwen2.5-VL 把 $(X_i,T)$ 扩成细 prompt；
2. VAE 编码参考图，wav2vec 编码 $Y_i$；
3. $z_S\sim\mathcal{N}(0,I)$，mask cache 置 0；
4. 50 步：前 10 步不按 mask 注音频，之后用 cache；
5. VAE 解码。

开销（720p、109 帧、单 A100、一次 forward）：1/2/3 张参考图时 mask 头 $0.4/0.8/1.2$ s，相对 7B DiT 的 $6.5$–$7.7$ s 可接受。

---

## 8. 开源代码在做什么（对照论文）

路径：[`OmniAvatar/models/wan_video_dit.py`](../../../code/audio_driven/interacthuman/OmniAvatar/models/wan_video_dit.py)。作者声明：这是 OmniAvatar 上的示意，**不保证与论文等价，无预训练权重。**

`MaskPredictor`：

```python
q = RMSNorm(Linear(dim→dim)(vid))     # Q：视频 token
k = RMSNorm(Linear(16→dim)(ref))      # K：参考图 VAE latent（16 通道），不是 h^r
v = Linear(dim→dim)(vid)              # V：视频，不是参考
q = rope_apply(q, freqs)              # 视频格子的 3D RoPE
k = rope_apply(k, freqs)              # 论文有 freqs_ref；这里仍用 freqs
x = flash_attn(q, k, v)
m = Sigmoid(Linear(512→1)(GELU(Linear(dim→512)(o(x)))))
```

`WanModel.forward` 里音频是 **全局 residual 相加**（只在 layer $2\ldots L/2$）：

```python
x = audio_cond_tmp + x          # 没有 m ⊙ p + (1-m) ⊙ p_mute
pred_mask = mask_predictor(x, ref * mask, ...)
# pred_mask 只 return，不回门控本步音频
```

对照清单：

| 论文                            | 开源 demo                             |
| ------------------------------- | ------------------------------------- |
| $Q=$视频, $K/V=$参考 hidden | $V=$视频；$K$ 来自 16 通道 latent |
| 3D RoPE 分视频 / 参考两套频率   | `freqs_ref` 算了但 `k` 没用       |
| 跨步 cache + 前 10 步关闭       | 没有外循环 mask cache                 |
| 软混合说话 / 静音 CA            | `AudioPack` 后直接加到 $x$ 上     |
| 内部 Seaweed MM-DiT + OmniHuman | Wan2.1 + OmniAvatar 单人流程          |

精读方法以 TeX Algorithm / §3 为准；demo 只用来看 mask 头长什么样。更细的入口表见 [`README_LOCAL.md`](../../../code/audio_driven/interacthuman/README_LOCAL.md)。

---

## 9. 和同期绑定方法怎么摆

|                         | 空间绑定                | 重叠说话                                   | 无首帧                         | 音频注入何时局部化 |
| ----------------------- | ----------------------- | ------------------------------------------ | ------------------------------ | ------------------ |
| MultiTalk               | 注意力图 → L-RoPE      | 弱，需预分轨                               | 否（I2V 首帧）                 | 训练+推理同一套    |
| AnyTalker               | 静态最大脸框            | 需预分轨                                   | 否                             | 框在去噪前就定死   |
| Bind-Your-Avatar        | Intra-Denoise 3D 软掩码 | 结构上支持（$\mathbf{A}^{\mathrm{ac}}$） | 可以                           | 训练 GT、推理预测  |
| **InterActHuman** | 跨步预测人体区域        | 评测为轮流                                 | **可以**（只要参考外观） | 正文：主要在推理   |

能力表（论文附录）把 MultiTalk / HunyuanVideo-Avatar 写成要首帧、不要多参考图；本文同时打多参考定制 + 多人说话。实验表 1 的 MultiTalk 是有数字的（Sync-D 7.671 vs 6.670），但 **没有 Bind**，多人测试也不是重叠说话。

---

## 10. 对 HarmonizedVA 的读法

- **可迁：** 第 $k$ 步 layout 指导第 $k{+}1$ 步条件；前若干步不开门；边界用 $m$ 与 $1-m$ 软混合而不是硬切；mask 监督整个人而不是参考图裁切那么小。
- **不要迁：** 轮流对话的评测设定（V1 要重叠说话）；局部音频主要在推理才门控（联合训练时门必须进计算图）；内部 7B Seaweed 配方本身。
- 本文不回答音色。mask 跟人走了，声仍可以不像这个人。把 InterActHuman 的动态区域接到 **独立音频潜变量 / 实体槽** 上，才是课题增量，而不是再实现一个 focal 分割头。
