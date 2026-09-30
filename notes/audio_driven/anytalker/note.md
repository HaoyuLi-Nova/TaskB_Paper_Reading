# AnyTalker：动机、创新点与核心技术

- **Authors / Year / Venue**: Zhizhou Zhong, Yicheng Ji, Zhe Kong, Yiying Liu（项目负责人）, Jiarui Wang, Jiasun Feng, Lupeng Liu, Xiangyi Wang, Yanjia Li, Yuqing She, Ying Qin, Huan Li, Shuiyang Mao, Wei Liu, Wenhan Luo（通讯） / 2025 / arXiv:2511.23475（CVPR 2026 投稿模板）
- **Link**: [https://arxiv.org/abs/2511.23475](https://arxiv.org/abs/2511.23475)
- **Project**: [https://hkust-c4g.github.io/AnyTalker-homepage](https://hkust-c4g.github.io/AnyTalker-homepage)
- **原文 TeX**: [`arxiv/audio_driven/anytalker/extracted/main.tex`](../../../arxiv/audio_driven/anytalker/extracted/main.tex)
- **官方代码（本仓库）**: [`code/audio_driven/anytalker/`](../../../code/audio_driven/anytalker/)（Wan2.1 Fun-1.3B-InP / I2V-14B；`main` @ `e5490b6`）
- **批判精读**: 同目录 [`critique.md`](critique.md) · **问答**: [`QA.md`](QA.md) · **AFCA 形状与空间门**: [`afca.md`](afca.md)
- **一句话**: 在 Wan I2V 上用共享参数的音–脸交叉注意力对每对 $(\mathrm{face},\,\mathrm{audio})$ 循环，空间上乘**全视频最大脸框**；用单人水平拼接学多人说话，再用约 $12$ 小时真实双人精炼听者反应。不改音色。

框架图：

![framework](../../../arxiv/audio_driven/anytalker/extracted/fig/method.png)

---

## 1. 动机

单人音频驱动（FantasyTalking、OmniAvatar、OmniHuman）已经能把**一张参考图 + 一路语音**做成口型同步的视频。一旦同画面要两个人对话，现成路线会撞上两堵墙：

1. **多人数据贵。** 轮流说话、角色切换、眼神等非语言线索难标注。MultiTalk / Bind-Your-Avatar / InterActHuman 都写了数百到数千小时多人数据。单人独白语料（HDTF、VFHQ、VoxCeleb2）规模大但学不到听的人该怎么动。
2. **多流控制不好扩。** MultiTalk 的 L-RoPE 要预先规定身份标签取值范围；人一多、标签集合一变，绑定轴就得重定义。作者要的是：**ID 数目在推理时任意增加，同一套参数循环即可**。

叙事上拆成两句：

- 用便宜的单人数据把谁的嘴跟哪路声先学会；
- 只用很少真实双人把听的人转头、抬眉补上。

范围边界（读的时候要看清）：

- 目标停在口型 / 表情 / 头姿 / **听者眼动**跟对的那路声。
- **不改音色**，也不是音视频联合生成。HarmonizedVA声要像这个人是另一条问题。
- 空间绑定是**静态最大脸框**，人物大幅走动、换位会失效——这和原 HarmonizedVA 的首帧框复制是同一类失败。

---

## 2. 创新点

相对换底座 / 再加一个 ID embedding这类组合，本文真正增量在 **多流怎么循环** 和 **单人数据怎么伪造成多人**。

### 2.1 AFCA：同一层、对每对 $(\mathrm{face},\,\mathrm{audio})$ 循环再求和

这是架构上相对 MultiTalk L-RoPE 的 delta。不做身份标签轴，而是：

$$
H_{i}' = H_{i} + \sum_{k=1}^{n} \mathrm{AFCA}_{\mathrm{out}}^{(k)},
$$

所有 $\mathrm{AFCA}_{\mathrm{out}}^{(k)}$ 来自**同一套** $W_{K},W_{V}$。$n$ 增加只线性重复 cross-attn，不复制一份头，也不改 RoPE 的 label 集合。这是最接近可变人数集合的开源设计。

### 2.2 音频与脸拼成一条 KV，再用静态框挡输出

论文公式：

$$
\begin{aligned}
K_{\mathrm{af}} &= \mathrm{Concat}(f_{\mathrm{audio}},\, f_{\mathrm{face}})\cdot W_{K}, \\
V_{\mathrm{af}} &= \mathrm{Concat}(f_{\mathrm{audio}},\, f_{\mathrm{face}})\cdot W_{V}, \\
\mathrm{Attn}_{\mathrm{out}} &= \mathrm{MHCA}(Q_{\mathrm{video}},\, K_{\mathrm{af}},\, V_{\mathrm{af}},\, M_{\mathrm{temporal}}), \\
\mathrm{AFCA}_{\mathrm{out}} &= M_{\mathrm{token}} \odot \mathrm{Attn}_{\mathrm{out}}.
\end{aligned}
$$

两件事要分开记：

- **对时**在 $M_{\mathrm{temporal}}$：首个 VAE 时间格看全部音频 token，之后每格只看 $4$ 个音频 token（与 Wan 的 $4$ 帧/格对齐）。
- **空间**在 $M_{\mathrm{token}}$：离线算整段视频脸框的**并集（最大框）**，patchify 后与 $\mathrm{Attn}_{\mathrm{out}}$ 逐元素相乘。框外的视觉 token 吃不到这路音频残差。

代码里并不是先 concat 再投影，而是脸、音频**分别投影后再在序列维拼接**（见 §4）。脸 token 在注意力掩码里始终可见，音频部分才跟 $M_{\mathrm{temporal}}$。

### 2.3 两阶段数据：拼接伪双人 + $12$ 小时真双人

| 阶段 | 数据 | 学什么 |
| --- | --- | --- |
| Stage 1 | $\approx 1000$ 小时单人；$50\%$ 概率把 batch 内相邻两条水平拼接 | 局部区域内的唇–音映射；伪双人说话模式 |
| Stage 2 | $\approx 12$ 小时真实双人（InsightFace 两张脸、diarization、光流、SyncNet $2\times 2$ 对角） | 听者反应：转头、抬眉、对视 |

拼接不是简单 resize 再并排：480P 目标 $(H,W)=(480,832)$，每人从脸中心裁 $(480,416)$ 再随机放大，避免 2K/4K 原图缩完脸只剩几格。完全去掉真实单人、只训拼接，口型会崩（表 4 第一行 Sync-C$^{*}$ $3.21$）。

### 2.4 不是新故事的部分

- 底座：Wan I2V（1.3B Fun-InP / 14B），3D VAE、T5、CLIP 参考交叉注意力、Wav2Vec2，全部冻结编码器。
- 音频注入位置：仍是每块 cross-attn，和 EMO / OmniHuman 系同一套路，只是 KV 从一路音频变成循环 $n$ 路。
- 脸框：InsightFace 离线最大框，**不是** Bind 的去噪中 3D mask，也不是 InterActHuman 的跨步 cache mask。
- 评测指标 Interactivity：把 CyberHost 的手部关键点方差搬到眼部，再限制在听者时段。新在听者眼动这个切片，不是新的运动统计。

和 MultiTalk 的差：MultiTalk 绑定靠 amap $\to$ L-RoPE（软身份轴）；本文绑定靠静态框挡 AFCA 输出。一个在旋转里做人，一个在注意力支撑集上做人。

和 Bind-Your-Avatar 的差：Bind 是**同一次 forward 里**预测 3D mask 再乘 $\mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\mathbf{A}^{\mathrm{cv}}$，明确留了重叠说话；本文框是视频级静态的，重叠说话依赖预分轨。

和 InterActHuman 的差：InterActHuman 评测默认一人说话一人 mute，mask 是去噪过程中预测的全身足迹；本文 InteractiveEyes 的 $80\%$ 两人都会说话，但空间仍钉在最大框上。

---

## 3. 技术方案（逐步拆开）

### 3.1 输入怎么进 DiT

沿用 Wan I2V：

$$
f_{\mathrm{input}} = [f_{\mathrm{video}},\, f_{\mathrm{text}},\, f_{\mathrm{ref}},\, f_{\mathrm{audio}}].
$$

- $f_{\mathrm{video}}$：3D VAE $\to$ patchify $\to$ flatten。
- $f_{\mathrm{text}}$：T5，截断/填充到 $512$ token。
- $f_{\mathrm{ref}}$：CLIP 编码**首帧**（以及每人裁出的脸，作为 $f_{\mathrm{face}}$）。
- $f_{\mathrm{audio}}$：Wav2Vec2；多人时是长度为 $n$ 的列表。

每块：self-attn（视频）$\to$ 文本 cross-attn $\to$ 参考图 cross-attn $\to$ AFCA $\to$ FFN。CFG $=4.0$，无条件分支把文本和音频置零。

### 3.2 时间对齐：$M_{\mathrm{temporal}}$

Wan 的时间压缩是 $4$ 帧 $\to$ $1$ 个 latent 格。音频 token 更密。图 3(a)：

- 第 $0$ 个视频格：可以看到**全部**音频 token（让第一帧吃到整句的韵律/身份）。
- 第 $t\ge 1$ 个视频格：只看见对应的 $4$ 个音频 token。

这和 MultiTalk 把 wav2vec 按 VAE 格子打包是同一类时钟，不是新时钟。

### 3.3 空间绑定：$M_{\mathrm{face}}$ 是全局最大框

训练：InsightFace 在**整段视频**上取脸框并集。推理：首帧检测后再**均匀膨胀**。然后

$$
M_{\mathrm{token}} = \mathrm{Patchify}(\mathrm{Flatten}(M_{\mathrm{face}})),
$$

与 $\mathrm{Attn}_{\mathrm{out}}$ 同形状，直接 $\odot$。

作者给的理由：人会在框内晃头，若用单帧小框，reshape 之后会激活错 token。这只覆盖框内位移，**不覆盖换位、走近、一人穿过另一人**。换位失败是结构决定的，不是超参没调好。

多人时每人一张 $M_{\mathrm{face}}^{(k)}$。两框重叠（凑近、侧脸抢同一片格子）时，两路 AFCA 残差会加到同一组视觉 token 上——结构上没有 Bind 那种 $\mathbf{A}^{\mathrm{ac}}$ 把重叠说话写清楚。

### 3.4 代码里的 AFCA（`WanAF2VCrossAttention`）

张量形状、各编码器 I/O、以及「空间门为何乘在 `Attn_i` 而不是 softmax见同目录 [`afca.md`](afca.md)。下面只保留对照论文时必须记住的代码差。

路径：[`wan/modules/model.py`](../../../code/audio_driven/anytalker/wan/modules/model.py)。一次 forward：

1. 文本 CA、参考图 CA 照旧。
2. 对 $k=1\ldots n$：
   - $K_{\mathrm{face}}^{(k)}=\mathrm{Linear}(f_{\mathrm{face}}^{(k)})$，$K_{\mathrm{audio}}^{(k)}=\mathrm{Linear}(f_{\mathrm{audio}}^{(k)})$（value 同理）；
   - 沿序列维 $\mathrm{cat}$ 成 $K_{\mathrm{concat}}$；
   - 注意力掩码：脸那一段全 `True`，音频那一段拷 $M_{\mathrm{temporal}}$；
   - 输出 $\times$ `face_mask_list[k]`。
3. 所有人的输出与文本/参考残差相加，过输出投影。

与论文式 (2) 的差别：论文写 $\mathrm{Concat}(f_{\mathrm{audio}},f_{\mathrm{face}})\cdot W_{K}$（先拼特征再投）；代码是**两套 Linear 后再拼 KV 序列**。功能等价于视频 $Q$ 同时看见这张脸和这路声，梯度路径不同。对照论文时以代码为准。

同文件还有 `LegacyA2VCrossAttention`：把视频按时间格拆开再做 CA，14B 路径相关。精读优先看 `WanAF2VCrossAttention`。

入口：[`generate_a2v_batch_multiID.py`](../../../code/audio_driven/anytalker/generate_a2v_batch_multiID.py) $\to$ [`wan/audio2video_multiID.py`](../../../code/audio_driven/anytalker/wan/audio2video_multiID.py)。脸框膨胀在 [`wan/utils/infer_utils.py`](../../../code/audio_driven/anytalker/wan/utils/infer_utils.py) 的 `expand_face_mask_flexible` / `gen_inference_masks`。

### 3.5 训练配方（数字要能对上）

编码器全冻；DiT + AFCA 全开。AdamW。

| | 1.3B | 14B |
| --- | --- | --- |
| 卡 | $8\times$ H200 | $32\times$ H200 |
| Stage 1 | lr $2\times 10^{-5}$，全局 batch $48$，$2.4$M step | 全局 batch $32$，$2.4$M step |
| Stage 2 | lr $5\times 10^{-6}$，batch $48$，$50$K step | batch $16$，$50$K step |

双人清洗：多数帧恰好两张脸；diarization 只允许两人说或一人说；左右顺序全程不变（InsightFace 拒换位）；SyncNet $2\times 2$ 矩阵最大两项必须在对角线。这套过滤本身就把走动换位的难例扔了，所以静态框在训练集里几乎不会被打脸。

### 3.6 评测：InteractiveEyes 与 Interactivity

数据集：约 $10$ 秒、稳定相机、**每一帧恰好两张脸**；$80\%$ 两人都说话，$20\%$ 一人说一人听。说话区间**人工**标，不用 diarization。每个单人基准各 $20$ 条，身份不在训练集。

$Motion$ 是对齐到脸上的眼部关键点，帧间 $L1$ 位移平均：

$$
\mathrm{Motion}=\frac{1}{|S|-1}\sum_{j}\frac{1}{|E|}\sum_{i}\lvert E_{i,j+1}-E_{i,j}\rvert.
$$

Interactivity 只在听者时段 $L_{2},L_{3}$ 上加权。Sync-C$^{*}$ 只在说话时段 $L_{1},L_{4}$ 上加权——听者闭嘴不再稀释口型分。

异常钳制：脸对齐到 $256\times 256$ 后，连续帧平均关键点位移 $>10$ px 则冻结，直到位移回到阈值下。这是为了把 Bind 那种突然躺倒从 $Motion$ 里拿掉。

---

## 4. 实验里该记住的数字

**单人（表 1，14B）。** HDTF Sync-C $9.05$ 超过 MultiTalk $8.91$；VFHQ FVD $290.73$ **差于** MultiTalk $243.66$。1.3B 口型明显弱于 14B 和 MultiTalk。结论应写成：14B 单人口型有竞争力，不是全面 SOTA。

**多人（表 2）。** Interactivity：GT $0.77$，MultiTalk $0.49$，Bind $0.45$，AnyTalker-1.3B $0.97$，14B $1.01$。生成视频**高于真值**。Sync-C$^{*}$：1.3B $4.56$ 明显低于 MultiTalk $6.88$；14B $6.99$ 略超。FVD 14B 最好。

**组件消融（表 3，HDTF，仅 Stage 1）。** 去掉 mask token，Sync-C $6.97\to 5.84$；去掉拼接数据 $6.97\to 6.21$；Baseline（只有普通音频 CA）Sync-C $5.42$ 但 FID 最好——作者解释为 Baseline 更呆，动得少所以 FID 好看。

**数据消融（表 4，InteractiveEyes）。** 三件套 RS+CM+RM 才把 Interactivity 拉到 $0.97$。有拼接无真双人：口型最好（$4.89$）但交互只有 $0.58$。有真双人无拼接：交互 $0.71$、口型掉到 $3.63$。拼接是口型的前置，真双人是交互的前置；二者不能互相替代。

**EMTD 半身（表 A1）。** 14B Sync-C $8.45$、FID $50.61$、FVD $664.58$ 优于 MultiTalk；ID $0.77$ 略低于 MultiTalk $0.79$。

**四 ID。** 只在 teaser / 主页视频里展示，定量表始终是双人。从双人训练泛化到四人是作者的推测（AFCA 学到了一般交互模式），没有四人测试集。

---

## 5. 对 HarmonizedVA 的直接含义

| 要素 | AnyTalker | 本课题目标 |
| --- | --- | --- |
| 音色 | 复制输入音轨 | 画面/参考调和音色 |
| 生成单位 | 多路音频挤进**一条**视频去噪 | 每人一条语音轨 + 共享视频 |
| 空间绑定 | 静态最大脸框 | 动态软路由 + 实体槽 |
| 重叠说话 | 预分轨；框重叠时残差相加 | 结构隔离 |
| 换位 | 训练集过滤掉；推理会灌错脸 | 身份跟随 |

该借鉴：

- **单人拼接当伪多人**是目前最省数据的多人说话预训练，Stage 1 的 $50/50$ 混合有消融撑着。
- **共享参数循环 $n$ 对**是可变人数最便宜的实现，不必预定义 L-RoPE 标签集。
- InteractiveEyes 的听者时段标注可以直接拿来评听的人像不像在听；指标本身要校准（见 critique）。

不该重复：

- 静态最大框当路由（原项目已经在换位上失败过）。
- 把 Interactivity $>$ GT 当成更自然而不做真人偏好校准。
- 期望 AFCA 自动解决串音：它只挡空间，不拆音频潜变量。
