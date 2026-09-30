# InterActHuman — 考察理解深度

配合 [`note.md`](note.md)（方法逐步拆开）和 [`critique.md`](critique.md)（六维批判）。下面每问都要求能讲清为什么这样设计，不是复述摘要。

---

## Q1. 论文真正要解的循环依赖是什么？为什么先分割再生成和同一步预测 mask 立刻注入都不够？

**答：**

循环是：

1. 局部音频需要每人的时空足迹 $m_i$；
2. $m_i$ 要拿生成视频（或至少不太噪的 latent）和参考图匹配才能算准；
3. 推理时没有 GT 视频，音频若注错，成片更不可能按人说话。

先分割再生成把 2 当成已发生的事，推理不成立。

同一步：用当前 $z_k$ 预测 $m_k$，立刻用 $m_k$ 门控本步音频看起来打破了循环，但早期 $z_k$ 接近纯噪，IoU 很低（附录 combine mask 在 step 1 只有 0.376 / 0.741）。用这张错 mask 把门，等于把错误布局写进 ODE：其他人的 token 开始长期吃 mute，后面很难翻盘。论文原话是早期不可靠 mask **会抑制其他参考概念**。

所以他们把时间轴拉到 **去噪步之间**：本步只负责把相对还能看的 $m_k$ 写入 cache，**下一步**再用。前 10 步连这个门都关着，先让参考图 self-attn 把人占进画面。

---

## Q2. 写出推理时 mask 与音频的调度。Algorithm 1 和正文为什么打架？你信哪一个？

**答：**

意图中的调度（正文 §3 + 图 2 注）：

```text
m_prev = 0
for k = S .. 1:                 # S=50，k 大更噪
    参考图 → self-attn 注入
    由 z_k 与 {X_i} 预测 m_k    # 本步只预测
    if k < S_mask:              # 前 10 步：k=50..41 不开门
        用 m_prev 做局部音频     # 用的是上一步
    m_prev = m_k
    sampler 更新 z
```

$S_{\mathrm{mask}}\approx 40$ 才能和前 10 步关闭同时成立。若误读成 $S_{\mathrm{mask}}=10$ 且 `if k < 10`，变成只在最后 9 步才局部注入，和 IoU 在 step 10 已经可用的事实相反。

Algorithm 1 写成：先 `cache ← 本步 m_i`，再 `if k < S_mask` 用 **本步** $m_i$ 注入。这是同一步预测+注入，否定了图注里的 $t{-}1\to t$。

**信正文和 cache 消融，不信伪代码赋值顺序。** 无 cache 时 Sync-D 从 6.921 掉到 11.046，只有音频门控依赖跨步旧 mask才解释得通。工程上最干净的实现是：`mask_in = cache` 作为 DiT 输入，forward 同时出 `(velocity, mask_out)`，再 `cache = mask_out`。

---

## Q3. Mask predictor 的输入 / 注意力 / 输出各是什么？它为什么能在参考图只是大头时仍标出全身？

**答：**

每层 $\ell$、每个身份 $i$：

- 视频 hidden $\mathbf{h}^{v}$ → $Q$；
- 该参考图在 DiT 里的 hidden $\mathbf{h}^{r}_i$ → $K,V$（论文公式；开源 demo 的 $V$ 改成了视频，$K$ 来自 16 通道 VAE latent，**不要混**）；
- 3D RoPE 后做

$\mathbf{p}^{(l)}_i=\mathrm{softmax}(Q^{v}(K^{r}_i)^{\top}/\sqrt{d})\,V^{r}_i;$

- MLP + sigmoid → $m^{(l)}_i\in[0,1]^{\text{时空格}}$；
- 后几层（或全层）平均得 $m_i$。

这是这个视频格子和参考 $i$ 有多像的外观匹配，不是检测框。参数跨 $i$ 共享（56M），所以 $N$ 增加只线性重复 cross-attn，不复制一份头。

监督的 GT 是 Grounding-SAM2 的 **整个人**（query=`person`），不是参考图裁切那么小的一块。因此即使用大头照当 $X_i$，头仍被要求预测头+肩+身体的足迹。好处：音频可以灌进会动的身体，而不是只打在脸上几格；坏处：语义类 `person` 在重叠、遮挡时会吞错物体（Fig. 6 黄猫 / 粉衣女孩）。

Focal $\alpha=0.25,\gamma=2$ 是因为格子里背景远多于人，BCE 早期不稳。SAM2 置信低于 $0.5$ 的帧用 alignment flag 从 mask loss 里拿掉，避免烂 GT 带偏头。

---

## Q4. 局部音频公式里为什么要 mute 分支？直接 $m_i\odot\mathbf{p}_i$ 行不行？听众那路音频在测试里是什么？

**答：**

$\mathbf{h}^{v}\leftarrow\mathbf{h}^{v}+m_i\odot\mathbf{p}_i+(1-m_i)\odot\mathbf{p}_i^{\mathrm{mute}}.$

$\mathbf{p}_i$：wav2vec($Y_i$) 的 cross-attn 输出（说话）；$\mathbf{p}_i^{\mathrm{mute}}$：同一套 CA，K/V 换成静音轨。

只做 $m_i\odot\mathbf{p}_i$ 等于：mask 外的 token **这一层完全吃不到音频残差**。多层叠下来，背景和听众的特征范数、AdaLN 统计会和训练时全局音频 CA 总在加东西不一致，边界格子还会因为 VAE 粗分辨率被一刀切出一圈伪影。加上 $(1-m_i)\odot\mathbf{p}^{\mathrm{mute}}$：

- 人 $i$ 外面仍走一条合法的音频残差（静音），分布更接近预训练 OmniHuman；
- $m_i\in(0,1)$ 的边界是软混合，不是硬切。

测试协议（正文评测段，必须记住）：**固定两人，一人 meaningful audio，一人 muted**。听众那路 $Y_j$ 本身就是静音，所以 $m_j$ 高的区域也在吃 mute——这才是听的人闭嘴。用户研究甚至写谁在说无所谓，只要不要大家一起说。因此表 1 的 Sync-D 更多证明 **不要全员说话**，弱于证明 **对的那个人在说**（没有对调两路音频看口型是否跟着换的实验）。

重叠说话：公式可以对两个 $i$ 都加 $m_i\odot\mathbf{p}_i$，支撑集不重叠时结构成立；支撑集重叠（两人凑近、抢同一张侧脸）没有评测。Bind 用 $\mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\mathbf{A}^{\mathrm{cv}}$ 显式留了重叠的位置。

---

## Q5. 前 10 步关掉 mask 注入，和 Bind 的 teacher-forcing，分别在防什么？能互相替代吗？

**答：**

| | InterActHuman 前 10 步关门 | Bind teacher-forcing |
| --- | --- | --- |
| 发生在 | **推理**，去噪步的外循环 | **训练**，注入用 SAM2 GT，router 的预测只吃 BCE |
| 防的是 | 早期错 mask 把其他人 token 锁进 mute | 联合训 router+去噪时，网络为了好分类把视觉内容洗掉；以及错 mask 让条件注不进去、扩散越训越废 |
| 推理时 | 第 11 步起用预测 cache | 全程用预测 mask（训练从没让 DiT 见过预测 mask） |

不能替代：一个管推理 ODE 的开头，一个管训练目标的可分性。InterActHuman 正文还说局部音频 **primarily at inference**——训练时 DiT 可能仍见全局音频，mask 头只是辅助损失。那样的话，它既没有 Bind 那种去噪见过正确门控，也没有训练见过预测门控，train/test 的音频条件分布差一截。论文没有 ablate训练也按 GT mask 门控。

---

## Q6. 四档消融（Global / ID embedding / Fixed mask / Predicted）各自对应哪种失败？Fixed mask 的 Sync-D 已经不错，为什么还说它不行？

**答：**

- **Global：** OmniHuman 整图灌同一路声 → 所有人一起说。Sync-D 最差（9.482），IQA 最好（4.768）：画质可以很好，只是口型绑错。说明好看和绑对人是两件事。
- **ID embedding：** 给每个身份一个可学习 token，加到配对的参考图和音频上，让 DiT 自己配对，不监督 mask。Sync-D 8.627。附录承认这只是 DETR 式的朴素实现，**不能**据此宣称隐式匹配理论更差。
- **Fixed mask：** 用户给静态矩形，且默认人少动、框内只有一个人。Sync-D 7.068，已经接近预测 mask 的 6.670，但 **FVD 最差（40.2）**：口型还行，运动和时序分布崩了——人一旦走出框，音频还在旧地方灌，或者人被框钉死。这就是首帧框复制全视频（AnyTalker / 原 HarmonizedVA）的定量样子。
- **Predicted：** Sync-D 和 FVD 同时最好。动态区域既跟上位置，又不把运动锁死。

所以不能只看 Sync-D 宣布固定框也行。绑定方法必须同时报口型和运动/FVD，否则会选出嘴对了但人像贴纸的解。

---

## Q7. 参考图是怎么进 DiT 的？为什么说零额外参数？这和 mask predictor 是不是同一条路？

**答：**

参考图 $X_i$ 走 **和视频相同的 3D VAE**，展成 token，**拼在** noisy video tokens 后面，每个 block 的 **self-attn** 里和视频互相看。QKV 都是原 DiT 的，所以外观注入零新参数。序列变长，self-attn 按 $(F+n_{\mathrm{ref}})^2$ 涨；表里 28 个视频 latent + $n$ 张参考，开销写成 $(1+n/28)^2$。

Mask predictor 是 **另挂的 cross-attn 头**（56M），读的是已经交互过的 $\mathbf{h}^{v},\mathbf{h}^{r}_i$，专门为了吐 $m_i$。两条路分工：

- self-attn concat：让外观长在视频里（全局、隐式占位）；
- mask 头：把占位变成可用的区域图，给音频这种必须局部的模态用。

没有 mask 头时，外观可以对，音频仍会全局漏；没有 concat、只靠 mask 灌音频，人可能根本不出现在该位置。消融没把关掉参考 concat、只留 mask单独做。

---

## Q8. 开源 `MaskPredictor` 和论文公式差在哪三处？精读时为什么必须以 TeX 为准？

**答：**

[`wan_video_dit.py`](../../../code/audio_driven/interacthuman/OmniAvatar/models/wan_video_dit.py)：

1. **$V$ 的来源：** 论文 $V\leftarrow$ 参考；代码 `v = Linear(vid)`，注意力变成用参考相似度去门控视频特征再出 mask，不是把参考特征拉到视频格子上再出 mask。
2. **$K$ 的来源：** 论文是参考 **hidden** $\mathbf{h}^{r}$；代码 `Linear(16, dim)` 打在 VAE **16 通道 latent** 上（还乘了 `mask`），没有用 DiT 里已经和视频交互过的参考 token。
3. **RoPE：** 算了 `freqs_ref`（参考只有 1 帧时间维），但 `rope_apply(k, freqs)` 仍用视频频率。音频路径则是 `x = audio_cond_tmp + x` 全局加，**完全没有** $m\odot p+(1-m)\odot p^{\mathrm{mute}}$，也没有 50 步外循环 cache。

作者 README 写明：ByteDance 政策下这是 OmniAvatar 上的 demonstration，无 checkpoint、不保证正确。所以：

- 讲论文方法→ TeX §3 + 校正后的 Algorithm；
- 讲能跑的开源→ OmniAvatar 单人推理 + 一个示意分割头。

把 demo 当成可复现的 InterActHuman 会得出错误的架构印象（尤其是音频居然不乘 mask）。

---

## Q9. 和 MultiTalk 的 L-RoPE、Bind 的 $\mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\mathbf{A}^{\mathrm{cv}}$ 比，InterActHuman 的绑定写在哪一层？

**答：**

- **MultiTalk：** 自注意力里用目标区域汇总出 amap，再把身份编进 1D RoPE 的额外轴，音频 CA 按这根轴对齐。绑定发生在 **旋转位置编码**，是软的、不可直接监督的身份轴。要首帧 I2V。
- **Bind：** 预测人物–视觉 $\mathbf{A}^{\mathrm{cv}}$，音频–人物 $\mathbf{A}^{\mathrm{ac}}$ 另给，乘成 $\mathbf{A}^{\mathrm{av}}$。绑定发生在 **cross-attn 的支撑集**，重叠说话 = 两路音频乘到不同空间。训练 GT、推理预测。
- **InterActHuman：** 绑定发生在 **音频残差进 $\mathbf{h}^{v}$ 之前的逐格门控**。没有单独的 $\mathbf{A}^{\mathrm{ac}}$：人 $i$ 的音频自然对应 mask $i$（身份与音频在输入里已经配对好）。重叠没有矩阵语言，只是两个门同时开。

对 HarmonizedVA：若既要人在哪（动态）又要哪路声、甚至哪段音色（可对调、可重叠），Bind 的矩阵分解更完整；InterActHuman 的跨步 cache 更适合当 **$m_i$ 怎么随去噪变准** 的调度，两者可以叠：cache 产 $\mathbf{A}^{\mathrm{cv}}$，$\mathbf{A}^{\mathrm{ac}}$ 仍由说话人分配给出。

---

## Q10. 论文说自己是unified interface for all modalities through the layout。布局真的统一注入了参考图、文本和音频吗？

**答：**

没有。Layout 目前 **只门控音频**（局部模态）。参考图走全局 self-attn concat，文本走 MM-DiT 文本流 + Qwen 扩写，都不乘 $m_i$。

统一接口是叙事：将来任何局部模态（动作、另一路音效、手里物体的外观变化）都可以复用同一张 $m_i$。实现上只兑现了音频。BlobGen-Vid 那种用户给时空 mask 去管文本概念反而是 layout 管外观；本文 layout 不管外观、管声。读 contribution (2) 时不要理解成 ControlNet 式的万用条件接口已经做完。

---

## Q11. 数据 2.6M triplet 和 2000 小时音频驱动各服务哪一段训练？没有它们方法还成立吗？

**答：**

- **2000 小时（SyncNet 过滤后）：** OmniHuman 式单人音频预训练——先学会全局听话。没有它，后面的 mask 门控没有会动嘴的先验可门。
- **2.6M video–mask–caption：** 多人 / 多概念阶段，给 mask 头提供 SAM2 GT，给参考图注入提供多样外观和人–物组合。过滤短于 $4$ s、多于 $3$ 人，所以任意 $N$ 的声称先天偏 2–3 人。

方法的 **算法**（cache + 软门控）不依赖这个规模；**论文里的 SOTA 数字**高度依赖内部数据和 7B Seaweed。开源 demo 两者都没有，所以可复现只复现了模块形状，不复现表 1。这和 Ovi 类似：配方可抄，壁垒在数据。

---

## Q12. 若把 InterActHuman 接到联合音视频生成（Ovi 式双塔）上，哪三件事必须改，哪一件可以原样搬？

**答：**

**可以搬：** 跨步 $m^{\mathrm{prev}}\to$ 下一步条件；前若干步不开门；边界 $m$ 与 $1-m$ 软混合。这与视频塔还是音频塔无关，是 ODE 上的空间绑定调度。

**必须改：**

1. **一条共享音频潜变量不够。** 本文 $Y_i$ 是外给的 wav2vec 条件，不是生成出来的。联合生成若仍是单轨音频，mask 只能决定画面谁动嘴，不能决定声带像谁。要对齐 HarmonizedVA，需要每人一路 speech stem，再用 $m_i$（或 Bind 的 $\mathbf{A}^{\mathrm{av}}$）把 stem $i$ 送到人 $i$ 的视觉 token，并反向让视觉约束该 stem 的音色。
2. **局部门控要进训练图。** 联合生成早期两路都极噪（Harmony 说的对应漂移）。只在推理插 mask，音频塔从未见过按人切开的速度场，会碎。应 teacher-forcing 或对预测 mask 直通，不能只 focal 一个分割头。
3. **评测协议。** 轮流 + 谁说都行测不出换位、重叠、音色绑错。至少要：对调两路声看口型是否换人；重叠说话；换位置后 mask 是否跟人走且声仍像该人。

细节逐步推导见 [`note.md`](note.md) §4–6、§10。
