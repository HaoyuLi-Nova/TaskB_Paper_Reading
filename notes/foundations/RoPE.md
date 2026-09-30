# RoPE 学习笔记

- **原论文**: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864))；苏剑林博客 [让研究人员绞尽脑汁的 Position Embedding](https://kexue.fm/archives/8265)
- **缩放 / 外推**: Chen et al., Position Interpolation ([arXiv:2306.15595](https://arxiv.org/abs/2306.15595))；Peng et al., YaRN ([arXiv:2309.00071](https://arxiv.org/abs/2309.00071))；Hugging Face [`modeling_rope_utils`](https://huggingface.co/docs/transformers/en/internal/rope_utils)
- **跨模态对齐**: Cheng et al., MMAudio ([arXiv:2412.15322](https://arxiv.org/abs/2412.15322))；Ovi ([arXiv:2510.01284](https://arxiv.org/abs/2510.01284))
- **本仓库代码**: Wan `rope_params` / `rope_apply_3d`；Ovi `freqs_scaling`；HarmonizedVA `ref_freqs`；MultiTalk `rope_1d`
- **一句话**: RoPE 不把位置向量加进 token，而是把 Q/K 在成对通道上转一个与位置成正比的角；注意力分数因此只依赖**相对角** $\varphi_{m,i}-\varphi_{n,i}$。所谓缩放 RoPE，就是改这个角怎么随位置增长——LLM 用来把更长序列压回训练过的角域，音视频用来把两套不同帧率压到同一根时间轴上。

---

## 0. 符号表

RoPE 永远有两根轴，不要混：

| 轴                   | 符号                                                                                                                                                     | 含义 |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| **token 位置** | $m$（Q 侧）、$n$（K 侧） | 这个 token 在序列 / 网格 / 身份轴上的坐标。注意力相对量是$m-n$。                                                        |      |
| **通道对下标** | $i$                        | 一头内部第$i$ 对实数通道（也就是第 $i$ 个复通道）。$i=0,1,\ldots,d/2-1$。第 $i$ 对占用实数维 $2i$ 与 $2i+1$。 |      |

$m$ 随 token 变；$i$ 随这个向量里哪两个维度变。转角 $\varphi_{m,i}$ 两个下标都要带。

### 位置 $m$ 在本文里的几种取值

$m$ **不是**第几个 token 的整数下标这一种东西，而是 **RoPE 用来转角的坐标**。同一套公式，坐标语义可以换：

| 记号                                                                                                                            | 取值                                                                                              | 语义                               |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------- |
| $m,n$                                                                                                                         | $0,1,\ldots,L-1$                                                                                | 1D 序列下标（语言模型、标准 RoPE） |
| $m^{\mathrm{v}}=(t,h,w)$                                                                                                      | $t\in\{0,\ldots,T_v-1\}$，$h,w$ 为空间格 | 视频 token 的 3D 网格坐标；时间维上的位置就是$t$ |                                    |
| $m^{\mathrm{a}}$                                                                                                              | $0,1,\ldots,T_a-1$                                                                              | 生成音频 token 的 1D 时间下标      |
| $m^{\mathrm{r}}$                                                                                                              | $0,1,\ldots$                                                                                    | Whisper 参考 token 的 1D 下标      |
| $m$                      | 如$[0,4]$ / $12$ / $[20,24]$           | MultiTalk 的身份坐标（仍叫$m$，只是不再是序列号） |                                                                                                   |                                    |

线性扫过视频网格时用 $\ell=t\cdot H_{\mathrm{tok}}\cdot W_{\mathrm{tok}}+h\cdot W_{\mathrm{tok}}+w$ 表示 **token 序号**；$\ell$ 不进 RoPE，$\varphi$ 用的是 $(t,h,w)$。

### 向量、频率、转角

| 符号                                                                                                                   | 含义                                                              |
| ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| $x_m$                                             | 位置$m$ 的内容向量（词嵌入或已投影的一头）                     |                                                                   |
| $q_m,k_n,v_n$                                     | 位置$m$ 的 query、位置 $n$ 的 key / value                    |                                                                   |
| $d$ | 一头的实数维（head dim，偶数）。$d=D/n_h$                                                                    |                                                                   |
| $D$                                                                                                                  | Transformer 隐维                                                  |
| $n_h$                                                                                                                | 头数。**不要**和 K 的位置 $n$ 混淆                        |
| $c=d/2$                                                                                                              | 一头里的复数通道数（也就是通道对的个数）                          |
| $x_{m,2i},\; x_{m,2i+1}$                          | 位置$m$ 的向量里，第 $i$ 对的两个实数分量                    |                                                                   |
| $b=10000$                                                                                                            | 频率底（代码里的`theta`）                                       |
| $\omega_i=b^{-2i/d}$                              | 第$i$ 对的角频率。$i$ 小 → $\omega_i$ 大 → 转得快        |                                                                   |
| $\varphi_{m,i}=m\cdot\omega_i$                    | 位置$m$、第 $i$ 对的转角。缩放后为 $m\cdot s\cdot\omega_i$ |                                                                   |
| $\lambda_i=2\pi/\omega_i$                         | 第$i$ 对转满 $2\pi$ 所需的位置跨度（波长）                   |                                                                   |
| $e_m$                                                                                                                | 加法位置嵌入（Vaswani / BERT 那种），**不是** RoPE 的 $m$ |
| $s$                                               | 缩放系数。LLM 外推时常$s=L'/L>1$；Ovi 音频 $s_a<1$           |                                                                   |
| $L,L'$                                                                                                               | 预训练最大位置、外推后的最大位置                                  |
| $\tau$                                                                                                               | YaRN 的注意力温度（**不是**视频时间 $t$）                 |

### 本仓库音视频（§3.3）

| 符号                                                                                                                                                       | 含义                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| $T_v,T_a$                                                                                         | 视频 / 生成音频的时间格数（Ovi 约$31$ 与 $157$） |                                                                |
| $H_{\mathrm{tok}},W_{\mathrm{tok}}$                                                                                                                      | 视频空间 token 网格高、宽                                      |
| $c_t,c_h,c_w$ | 一头里分给时间 / 高 / 宽的复数通道数，$c_t+c_h+c_w=c$                                                                                  |                                                                |
| $\omega^{(t)}_i,\omega^{(h)}_i,\omega^{(w)}_i$ | 三段各自的频率表，$i$ 仍是该段内部的通道对下标                                                        |                                                                |
| $s_a\approx T_v/T_a$                                                                                                                                     | 生成音频的时间缩放（`temporal_rope_scaling_factor`）         |
| $s_r$                                                                                                                                                    | Whisper 参考的时间缩放（`ref_temporal_rope_scaling_factor`） |

对照代码：`pos` / 网格坐标 → $m$（或 $t,h,w$）；`arange(0, dim, 2)` 的下标 → $i$；`freqs[m, i]` → $e^{\mathrm{j}\varphi_{m,i}}$。

---

## 1. 动机

Self-attention 对置换等变：把序列打乱，只要内容不变，分数矩阵只是跟着置换。语言 / 视频都需要谁在谁前面。Transformer 原论文用**加法绝对位置**：

$$
x_m \leftarrow x_m + e_m
$$

$e_m$ 可以是 $\sin/\cos$ 或可学习表。问题有三：

1. **绝对与相对打架**：模型真正需要的往往是 $m-n$（相隔几个 token），但加法把 $e_m$ 和内容缠在一起，相对关系只能靠网络自己从绝对表里抠。
2. **长度不灵活**：可学习表长度锁死；正弦表理论上可外推，实测过训练长度后迅速崩。
3. **和线性注意力不兼容**：相对位置的常见写法是在 $q^\top k$ 展开式里加 bias / 额外项，依赖 softmax 注意力的分解，套不到 kernelized linear attention 上。

相对位置（Shaw、Transformer-XL、T5 bias）直接往注意力里塞 $m-n$，相对性好，但实现重、和 FlashAttention 不好融，也通常**不旋 V**——位置只调制权重。

RoPE 要同时满足三件事（论文 §3.1 的约束）：

$$
\langle f_q(x_m,m),\; f_k(x_n,n)\rangle = g(x_m,x_n,m-n)
$$

即：Q、K 各自只看自己的绝对位置 $m$、$n$，点积却只依赖相对位置 $m-n$。旋转矩阵是满足这条的解：正交（不改范数）、可叠加（$R(m)^\top R(n)=R(n-m)$）、无额外参数。

---

## 2. 计算公式

### 2.1 二维：一对通道就是一次平面旋转

头维 $d$ 必须为偶数。先看只有一对的情形：$i=0$，两个实数分量 $(x_{m,0},x_{m,1})$，频率记 $\omega_0$（论文二维节写作 $\theta$，本文统一用 $\omega_i$）。位置为 $m$：

$$
f_{\{q,k\}}(x_m,m) = \begin{pmatrix} \cos(m\omega_0) & -\sin(m\omega_0) \\ \sin(m\omega_0) & \cos(m\omega_0) \end{pmatrix} W_{\{q,k\}} x_m
$$

等价复数写法（论文式 12）：

$$
(W_q x_m)\,e^{\mathrm{j} m\omega_0},\qquad (W_k x_n)\,e^{\mathrm{j} n\omega_0}
$$

点积取出实部后只剩相对角 $(m-n)\omega_0$：

$$
q_m^\top k_n = \mathrm{Re}\big[(W_q x_m)\,(W_k x_n)^{*}\,e^{\mathrm{j}(m-n)\omega_0}\big]
$$

几何图像：Q 转 $m\omega_0$、K 转 $n\omega_0$，两者夹角是 $(m-n)\omega_0$。夹角小 → 点积大。**位置从不进入 $v_n$。**

### 2.2 一般形式：$d/2$ 个互不干扰的 2D 子空间

把 $d$ 维切成 $d/2$ 对。第 $i$ 对用自己的频率（0-based，与代码一致；论文式 15 的 $i$ 从 1 起，差一个换元）：

$$
\omega_i = b^{-2i/d} = 10000^{-2i/d},\qquad i=0,1,\ldots,d/2-1
$$

块对角旋转：第 $i$ 块是 $2\times 2$，转角 $\varphi_{m,i}=m\omega_i$。

$$
R_{\omega,m}^{d} = \mathrm{diag}\Big( \begin{pmatrix}\cos(m\omega_0) & -\sin(m\omega_0) \\ \sin(m\omega_0) & \cos(m\omega_0)\end{pmatrix}, \;\ldots,\; \begin{pmatrix}\cos(m\omega_{d/2-1}) & -\sin(m\omega_{d/2-1}) \\ \sin(m\omega_{d/2-1}) & \cos(m\omega_{d/2-1})\end{pmatrix} \Big)
$$

于是

$$
q_m = R_{\omega,m}^{d} W_q x_m,\qquad k_n = R_{\omega,n}^{d} W_k x_n
$$

$$
q_m^\top k_n = x_m^\top W_q^\top R_{\omega,\,n-m}^{d}\, W_k x_n
$$

第 $i$ 对的转角：

$$
\varphi_{m,i} = m\cdot\omega_i = m\cdot 10000^{-2i/d}
$$

低频（$i$ 小、$\omega_i$ 大）对近邻敏感；高频通道的波长

$$
\lambda_i = \frac{2\pi}{\omega_i}
$$

可以比训练长度还长，几乎只编码还在序列哪一段这种粗位置。YaRN / Llama 3 缩放时会按 $\lambda_i$ 分段，根源在这里。

论文还证明：按 $10000^{-2i/d}$ 取频，内积随 $|m-n|$ **长期衰减**——离得远的 token 平均上更难对准。这是启发式，不是硬约束；后面会看到身份 RoPE、跨模态 RoPE 都在**故意违反**下标差 = 距离。

### 2.3 两种等价实现

**复数乘**（Wan / Ovi `rope_params`）。表的形状是 $[\text{max } m,\; d/2]$，第 $(m,i)$ 项是 $e^{\mathrm{j}\varphi_{m,i}}$：

```python
freqs = 1.0 / theta ** (torch.arange(0, dim, 2).float() / dim)  # ω_i，i = 0 … d/2-1
angs  = torch.outer(pos, freqs)                                  # φ_{m,i} = m · ω_i
cis   = torch.polar(torch.ones_like(angs), angs)                 # e^{j φ_{m,i}}
x_rot = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
y     = torch.view_as_real(x_rot * cis).flatten(-2)
```

**`rotate_half` + cos/sin**（GPT-NeoX / LLaMA 风格，MultiTalk `rope_1d`）。对每一对 $(x_{m,2i},x_{m,2i+1})$：

$$
y_{m,2i} = x_{m,2i}\cos\varphi_{m,i} - x_{m,2i+1}\sin\varphi_{m,i}
$$

$$
y_{m,2i+1} = x_{m,2i}\sin\varphi_{m,i} + x_{m,2i+1}\cos\varphi_{m,i}
$$

向量写法：$y = x\cos\varphi + \mathrm{rotate\_half}(x)\sin\varphi$，其中 $\mathrm{rotate\_half}:(x_{2i},x_{2i+1})\mapsto(-x_{2i+1},x_{2i})$，就是乘 $\mathrm{j}$。两套公式与 $e^{\mathrm{j}m\omega_i}$ **完全相同**，不是简化版。

配对方式有一个无伤大雅的分叉（$i$ 仍是第几对，只是哪两个实数维算一对不同）：

| 风格                    | 第$i$ 对占用的实数维 | 代表                                          |
| ----------------------- | ---------------------- | --------------------------------------------- |
| GPT-J / 论文相邻维      | $(2i,\; 2i+1)$       | Wan`view_as_complex`、MultiTalk `rope_1d` |
| GPT-NeoX / LLaMA 对半切 | $(i,\; i+d/2)$       | Hugging Face Llama                            |

频率公式一样。对照官方实现时不要强行改配对。

### 2.4 和加法正弦的关系

Vaswani 已经用同一套频率：加法嵌入的第 $2i$ 维是 $\sin(m\omega_i)$。RoPE 用同一套 $\omega_i$，但**乘到已经投影好的 $q_m/k_n$ 上**，而不是加到 $x_m$ 上。因此：

- 没有 `pos_emb[token]` 这张表；
- $v_n$ 不带位置；
- 相对性是旋转的代数性质，不是额外 bias。

---

## 3. 缩放 RoPE（重点）

缩放不是第三种位置编码，只是改转角里的 **$m$ 或 $\omega_i$**。统一写成

$$
\varphi_{m,i}= g(m)\cdot h(\omega_i)
$$

标准 RoPE：$g(m)=m,\; h=\mathrm{id}$。所有变体都在动 $g$ 或 $h$。

社区里其实是**两族用途、同一类算子**：

|              | 族 A：LLM 上下文外推                                                                     | 族 B：跨模态时间对齐                       |
| ------------ | ---------------------------------------------------------------------------------------- | ------------------------------------------ |
| 问题         | 训练长度$L$，推理要 $L'\gg L$          | 视频$T_v$ 格 vs 音频 $T_a$ 格，两套时钟 |                                            |
| $s$ 的含义 | $s=L'/L>1$：把更长下标压进旧角域                                                       | $s=T_v/T_a<1$：把更密的序列压进更疏的轴  |
| 典型实现     | HF`rope_type=linear/dynamic/yarn/llama3`                                               | Ovi`freqs_scaling`；MMAudio aligned RoPE |
| 改哪一侧     | 整条语言模型的 RoPE                                                                      | 通常只改**非预训练 / 更密**的那一路  |

Ovi 的 `freqs = freqs_scaling * freqs` 与 HF linear 的 `inv_freq /= factor` 是同一件事，只是 $s$ 与 HF 的 `factor` 互为倒数。

### 3.1 族 A — 把更长上下文塞回训练过的角域

预训练时位置 $m=0,\ldots,L-1$，最大转角约 $L\cdot\omega_i$。推理到 $L'$ 若仍用 $m=0,\ldots,L'-1$（**裸外推**），低频通道会转到训练从未见过的区域，PPL 陡升。

#### 线性 Position Interpolation（PI）

Chen et al. / kaiokendev：不外推下标，而把下标**插值**回 $[0,L)$：

$$
g(m)=\frac{m}{s},\qquad s=\frac{L'}{L},\qquad h(\omega_i)=\omega_i
$$

即 $\varphi_{m,i}= (m/s)\cdot\omega_i = m\cdot(\omega_i/s)$。HF `rope_type="linear"` 直接 `inv_freq /= factor`，`factor` 就是这里的 $s$。

效果：长度 $L'$ 的最后一个 token 转角仍是 $L\cdot\omega_i$，和训练时最后一个 token 一样。需要少量长文本微调。

**代价（NTK 视角）**：所有 $i$ 的频率被同等拉长。$i$ 小、$\omega_i$ 大的那几对本用来分辨差 1 个 token，被压扁后，邻近 token 的相对角太小，网络分不清谁挨着谁。短上下文微调后反而可能变差。

#### NTK-aware：改 $b$，而不是所有 $\omega_i$ 除以同一个 $s$

bloc97；YaRN 论文写成正式定义。目标：

- 最低频（$i$ 大、波长最长）按 PI 那样缩放；
- 最高频（$i$ 小）尽量不动，保住局部分辨率。

做法是换 RoPE 的底（YaRN 原文用 $|D|$ 表示参与旋转的偶数维，本文即 $d$）：

$$
b' = b\cdot s^{d/(d-2)}
$$

然后仍用 $\omega_i'=(b')^{-2i/d}$。HF `rope_type="dynamic"` 属于这一支（再按当前序列长动态更新 $s$）。Code Llama 把 `rope_theta` 直接加大到 $10^6$，是手工版 NTK-aware。

零样本外推优于 PI；微调时部分 $i$ 会略微出界，不如 PI / YaRN 稳。

**Dynamic NTK**：每步 $s=\max(1,\; L_{\mathrm{cur}}/L)$，短序列不缩放，超过 $L$ 再逐渐拉。适合自回归、KV cache（cache 必须存在 **RoPE 之前** 的 $k_n,v_n$，因为 $s$ 一变每个 $m$ 的转角都变）。

#### NTK-by-parts 与 YaRN

按波长 $\lambda_i=2\pi/\omega_i$ 把通道对分成三区（$\rho_i=L/\lambda_i$）：

- $\lambda_i \ll L$（$i$ 小，高频 / 局部）：**不插值**，外推原频率；
- $\lambda_i \gtrsim L$（$i$ 大，低频 / 全局）：只做 PI，避免出界；
- 中间：在 $\omega_i$ 与 $\omega_i/s$ 之间用 ramp 混合。

YaRN = NTK-by-parts + 注意力温度。拉长后相对角变小，softmax 更尖；乘一个

$$
\tau \approx 0.1\ln s + 1
$$

（或把 $\sqrt{1/\tau}$ 乘进 RoPE 的复因子，不动 attention 内核）。HF `rope_type="yarn"`，额外参数 `beta_fast=32`、`beta_slow=1`、`attention_factor`。

#### Llama 3.1 的分段缩放

HF `rope_type="llama3"`。Llama 3.1：预训练 8k，`factor=8` → 128k；`low_freq_factor=1`、`high_freq_factor=4`。对每个 $i$ 看波长 $\lambda_i=2\pi/\omega_i$：

- 波长很短（高频，$i$ 小）：$\omega_i$ 不变；
- 波长很长（低频，$i$ 大）：$\omega_i\leftarrow\omega_i/\mathrm{factor}$；
- 中间：在两者间平滑。

和 YaRN 同一哲学（高频留给局部、低频拿去覆盖长距离），实现是 Meta 自己的平滑，不是 YaRN 的 $\alpha/\beta$ ramp。

#### Hugging Face `rope_type` 对照

| `rope_type` | 动的是什么                           | 典型用途        |
| ------------- | ------------------------------------ | --------------- |
| `default`   | $g(m)=m$                           | 原版 LLaMA RoPE |
| `linear`    | $\omega_i\leftarrow\omega_i/s$     | PI              |
| `dynamic`   | 改$b$，且随当前长度变              | Dynamic NTK     |
| `yarn`      | 按$\lambda_i$ 分区 + 温度 $\tau$ | 长上下文微调    |
| `llama3`    | 按$\lambda_i$ 三段平滑             | Llama 3.1 128k  |
| `longrope`  | 每个$i$ 不同的长短因子             | Phi-3 等        |

### 3.2 族 B — 把两套时间分辨率拧到同一根角轴上

LLM 缩放解决的是 **一条序列变长**。音视频联合生成解决的是 **两条序列一开始就不同步**。

RoPE 的分数只看 $\varphi_{m,i}-\varphi_{n,i}$。若视频 token 的时间坐标用 $t=0,\ldots,T_v-1$，音频 token 的位置用 $m^{\mathrm{a}}=0,\ldots,T_a-1$，两边都 $s=1$，则时间段上

$$
\varphi^{\mathrm{v}}_{t,i}=t\cdot\omega_i,\qquad \varphi^{\mathrm{a}}_{m^{\mathrm{a}},i}=m^{\mathrm{a}}\cdot\omega_i
$$

第 $0$ 帧视频会对上第 $0$ 个音频格，第 $T_v-1$ 帧会对上第 $T_v-1$ 个音频格——但 $T_a>T_v$（Ovi 里 $157>31$），后面那些音频格在视频时间轴上没有对应的 $t$，亲和矩阵的对角线是斜的、错位的。Ovi 论文图 2 画的就是这件事。

**对齐条件**：希望同一物理时刻在同一对 $i$ 上转同一只角，即

$$
t\cdot\omega_i \approx (m^{\mathrm{a}}\cdot s_a)\cdot\omega_i \quad\Longleftrightarrow\quad s_a \approx \frac{T_v}{T_a}
$$

Ovi：$31/157\approx 0.197$，配置写成 `0.19676`。于是 $m^{\mathrm{a}}\approx 5t$ 时两边时间段转角相等——大约一帧视频对上 $5$ 个音频格。**没有把 $157$ reshape 成 $31$**，只是让音频转得更慢，整段 $T_a$ 才铺满约 $T_v$ 个视频时间单位。

MMAudio 做的是镜像：视觉 $8$ fps、音频 latent $31.25$ fps，他们把**视觉**频率乘 $31.25/8>1$（让疏的那边转快），而不是把音频乘 $8/31.25$。相对对齐相同；选哪一侧，取决于哪一路的预训练 RoPE 不能动。Ovi 视频塔是 Wan，必须 $s=1$，所以只能缩放音频。

这和 PI 的同构：

$$
s_a=0.19676 \;\equiv\; \mathrm{HF\;factor}\approx 5.08
$$

都是把更密 / 更长的下标压进另一套已经训练好的角域。差别只在叙事：LLM 是长度外推，这里是**跨模态时钟校准**。YaRN 那种按 $\lambda_i$ 分区，在这条线上通常**不必上**——你要的是整根时间轴对齐，不是保住差 $1$ 个音频 token的局部分辨率去和视频做 cross-attn。

MMAudio 自己也写：aligned RoPE **有用但不够**，他们另外加了 sync 模块。Ovi 把同步完全交给缩放 RoPE + 双向 cross-attn + SyncNet 滤数据，没有显式 sync loss。

### 3.3 落到本仓库：Ovi / HarmonizedVA 的转角

公式都写在 `rope_params` / `rope_apply_*` 和 `audio.json` 里。先定一头里怎么切通道（$i$ 的分段），再写每种 token 的位置 $m$ 怎么进转角。

#### 头维怎么切（$i$ 的三段）

$d=D/n_h$。Ovi 按 Wan 的 3D RoPE 把复数通道 $c=d/2$ 切成三段（与 `rope_apply_3d` 的 `split` 一致）：

$$
c_t=c-2\lfloor c/3\rfloor,\quad c_h=c_w=\lfloor c/3\rfloor
$$

建表时 `rope_params` 的 `dim` 是实数维（`arange(0, dim, 2)`，所以表长是该段的通道对数）：

| 段   | 通道对个数 | `dim` 实参                               | 该段内部的$i$      |
| ---- | ---------- | ------------------------------------------ | -------------------- |
| 时间 | $c_t$    | $d-4\lfloor d/6\rfloor$                  | $i=0,\ldots,c_t-1$ |
| 高   | $c_h$    | $2\lfloor d/6\rfloor$   | 各自从$0$ 起 |                      |
| 宽   | $c_w$    | $2\lfloor d/6\rfloor$   | 各自从$0$ 起 |                      |

未缩放的基频（`theta=10000`，$i$ 是**该段表**的下标）：

$$
\omega_i=\;10000^{-2i/\mathrm{dim}},\quad i=0,1,\ldots,\mathrm{dim}/2-1
$$

代码是先乘缩放再和位置做外积：

$$
\varphi_{m,i}(s)= m\cdot s\cdot\omega_i
$$

复数因子 $e^{\mathrm{j}\varphi_{m,i}}$。`s=freqs_scaling`。

Ovi `rope_params` 原句：

```python
freqs = 1.0 / theta ** (arange(0, dim, 2) / dim)   # ω_i
freqs = freqs_scaling * freqs                       # ω_i ← s · ω_i
freqs = outer(arange(max_seq_len), freqs)           # φ_{m,i} = m · s · ω_i
freqs = polar(ones, freqs)                          # e^{j φ_{m,i}}
```

#### 视频 token：位置 $m^{\mathrm{v}}=(t,h,w)$，$s=1$

`video.json` 不设 scaling。第 $\ell$ 个视频 token 对应网格 $(t,h,w)$，

$$
\ell=t\cdot H_{\mathrm{tok}}\cdot W_{\mathrm{tok}}+h\cdot W_{\mathrm{tok}}+w
$$

$\ell$ 只是展平序号。一头里三组通道 **各自** 用网格的一维当位置：

$$
\begin{aligned}
\varphi^{\mathrm{v}}_{t,i} &= t\cdot\omega^{(t)}_i \\
\varphi^{\mathrm{v}}_{h,i} &= h\cdot\omega^{(h)}_i \\
\varphi^{\mathrm{v}}_{w,i} &= w\cdot\omega^{(w)}_i
\end{aligned}
$$

$t\in\{0,\ldots,T_v-1\}$（约 $31$），$h,w$ 是空间格下标。
时间段的通道对数 $c_t$ 必须和下面音频转的那一段相同，fusion 才能对上。

#### 生成音频 token：位置 $m^{\mathrm{a}}$，时间缩放 $s_a$

官方 Ovi `audio.json`：

```json
"temporal_rope_scaling_factor": 0.19676
```

约等于 $T_v/T_a=31/157\approx 0.197$。

只转 **前 $c_t$ 个复数维**（和视频的时间段一样宽），后面 $c_h+c_w$ 维不转（`rope_apply_1d` 的 passthrough）。下面用 **一个位置上的一头** 把这句话算开。

##### 例子：位置 $m^{\mathrm{a}}=5$ 的一个音频 token，一头 $d=128$

Ovi 的 $D=3072$，$n_h=24$，所以一头 $d=128$ 个实数 $=64$ 个复数对。视频 3D RoPE 把这 $64$ 对切成：

$$
c_t=22,\quad c_h=c_w=21
$$

| 全局通道对$i$  | 实数维          | 视频拿来编码                                                        | 这个音频 token 做什么                   |
| ---------------- | --------------- | ------------------------------------------------------------------- | --------------------------------------- |
| $0,\ldots,21$  | $0$–$43$   | 时间$t$ | **转**：$\varphi=5\cdot s_a\cdot\omega^{(t)}_i$ |                                         |
| $22,\ldots,42$ | $44$–$85$  | 高$h$                                                             | **不转**：乘 $1$（passthrough） |
| $43,\ldots,63$ | $86$–$127$ | 宽$w$                                                             | **不转**：乘 $1$                |

代码就是把向量看成 $64$ 个复数后切开：

```python
c_rope = freqs.shape[1]          # 22，只有时间段那张表
x_i_rope = x_i[:, :, :c_rope] * freqs[m]   # 前 22 对 × e^{j φ}
x_i_passthrough = x_i[:, :, c_rope:]       # 后 42 对原样拼回去
```

取第 $0$ 对（时间段第一对，$\omega^{(t)}_0=1$，$s_a=0.19676$）：

$$
\varphi^{\mathrm{a}}_{5,0}=5\cdot 0.19676\cdot 1=0.9838
$$

若这两个实数原来是 $(x_0,x_1)$，转完是

$$
\begin{pmatrix} x'_0 \\ x'_1 \end{pmatrix}
=
\begin{pmatrix} \cos 0.9838 & -\sin 0.9838 \\ \sin 0.9838 & \cos 0.9838 \end{pmatrix}
\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
$$

同一头的第 $22$ 对（视频眼里的高）**转角为 $0$**：

$$
(x_{44},x_{45}) \leftarrow (x_{44},x_{45})
$$

没有 $e^{\mathrm{j}\varphi}$，向量方向不变。后面 $i=23,\ldots,63$ 全是这样。

对比：同一物理时刻附近的一个视频 token，例如网格 $(t,h,w)=(1,2,3)$（因为 $5\cdot 31/157\approx 1$）。它的 $64$ 对**全转**，但三段用不同位置：

| 全局$i$     | 视频转角       | 音频转角（上面那个 token）                                |
| ------------- | -------------- | --------------------------------------------------------- |
| $0$（时间） | $t\cdot 1=1$ | $0.9838$ → 相对角 $\approx 0$，对得上                |
| $22$（高）  | $h\cdot 1=2$ | $0$ → 相对角就是视频的 $h$，**与音频下标无关** |
| $43$（宽）  | $w\cdot 1=3$ | $0$ → 同上                                             |

点积 $q^\top k$ 是 $64$ 对加起来的。只有前 $22$ 对的相对角带着第几帧对第几个音频格；后 $42$ 对音频侧是单位旋转，不会在高/宽子空间里假装自己有个 $h$ 或 $w$。

若音频也把后 $42$ 对按 $m^{\mathrm{a}}=5$ 去转，视频会把那两段读成高度 $=5$、宽度 $=5$，空间 RoPE 和音频时间下标缠在一起，fusion 对的就不是时间了。

$m^{\mathrm{a}}\in\{0,\ldots,T_a-1\}$（$0$ 到 $156$）：

$$
\varphi^{\mathrm{a}}_{m^{\mathrm{a}},i}= m^{\mathrm{a}}\cdot s_a\cdot\omega^{(t)}_i
\qquad s_a=0.19676\approx\frac{T_v}{T_a}
$$

和视频第 $t$ 帧时间段相比：

$$
\varphi^{\mathrm{a}}_{m^{\mathrm{a}},i}\approx \varphi^{\mathrm{v}}_{t,i}
\quad\text{当}\quad
m^{\mathrm{a}}\cdot\frac{T_v}{T_a}\approx t
$$

即 $m^{\mathrm{a}}\approx t\cdot T_a/T_v\approx 5t$。

#### Whisper 参考 token：位置 $m^{\mathrm{r}}$，另一套缩放 $s_r$

本课题在 Ovi 音频塔上加了参考音频 K。同一段时间通道，另一张表 `ref_freqs`：

```json
"ref_temporal_rope_scaling_factor": 0.11670
```

$$
\varphi^{\mathrm{r}}_{m^{\mathrm{r}},i}= m^{\mathrm{r}}\cdot s_r\cdot\omega^{(t)}_i
\qquad s_r=0.11670
$$

$s_r\neq s_a$ 是有意的：**不要**让 $m^{\mathrm{r}}$ 和 $m^{\mathrm{a}}$ 格点一一对应，避免 cross-attn 按位置抄参考音色。`k_ref` 用 `ref_freqs`，生成音频 $q$ 用 `freqs`（$s_a$）。这是缩放 RoPE 的第三种用法——不是对齐、也不是外推，而是**把两条音频故意错开**，让注意力去对内容而不是对下标。

#### Fusion 里谁对谁

相对角仍是 $\varphi_{m,i}-\varphi_{n,i}$。这里 $m$ 取 Q 的位置，$n$ 取 K 的位置：

| Q        | K        | Q 的位置$m$                         | K 的位置$n$ |                                       |
| -------- | -------- | ----------------------------------------------------- | ------------------------------------- |
| 视频     | 生成音频 | $(t,h,w)$，$s=1$，三段都转                        | $m^{\mathrm{a}}$，只时间段，$s_a$ |
| 生成音频 | 视频     | $m^{\mathrm{a}}$，只时间段，$s_a$                 | $(t,h,w)$                           |
| 生成音频 | Whisper  | $m^{\mathrm{a}}$，$s_a$                           | $m^{\mathrm{r}}$，$s_r$           |

视频时间维和音频只有 $s_a$ 对齐；高/宽两段音频侧是 $0$ 旋转，不参与对哪一帧。同一对 $i$ 才能比角；时间段的 $i$ 与高、宽段的 $i$ 不是同一组通道。

---

## 4. 其它变种（位置语义，不只是缩放）

缩放改的是 $\varphi_{m,i}$ 随 $m$ 增长的速度。下面这些改的是 **$m$ 代表什么**，频率仍是 $\omega_i=b^{-2i/d}$，$i$ 仍是通道对。

### 4.1 3D RoPE（Wan / CogVideo 一路）

把一头的 $i=0,\ldots,c-1$ 切成 $T/H/W$ 三段，各自用网格坐标当位置：

$$
\varphi_{m^{\mathrm{v}},i} = \big(t\cdot\omega^{(t)}_{i_t},\; h\cdot\omega^{(h)}_{i_h},\; w\cdot\omega^{(w)}_{i_w}\big)
$$

同一套 $\omega$ 表按通道拼起来（见 §3.3）。自注意力里差一帧和差一行走不同的 $i$ 子空间。Ovi 音频只借用其中时间段，就是为了和这段对得上。

### 4.2 M-RoPE（Qwen2-VL 等）

多模态把位置拆成时间 / 高度 / 宽度三维 ID，文本则三维取同一个 $m$。和 3D RoPE 同类：一块头维、多根位置轴，通道对 $i$ 仍分段。

### 4.3 身份 1D RoPE（MultiTalk 音频 cross-attn）

$m$ 不是序列下标，而是说话人坐标：人 1 ∈ $[0,4]$，人 2 ∈ $[20,24]$，背景 $=12$；音频 $n$ 钉在区间中点 $2$ 与 $22$。公式仍是 $\varphi_{m,i}=m\cdot\omega_i$，相对角 $\varphi_{m,i}-\varphi_{n,i}$ 编码的是是不是同一个人。详见 `code/mycode/multitalk_layers.py` 的 `rope_1d`。

这再次说明：RoPE 不规定 $m$ 必须是 $0,1,2,\ldots$。任何你希望接近就对得上的标量轴都可以当 $m$。缩放 RoPE 只是给这根轴乘了一个常数 $s$。

### 4.4 和 RoPE 并列的位置方案（不是 RoPE 变种）

| 方法             | 位置怎么进注意力                                                                  | 和 RoPE 的关系                         |
| ---------------- | --------------------------------------------------------------------------------- | -------------------------------------- |
| ALiBi            | 分数上加与$m-n$ 成正比的线性 bias | 不旋$q_m/k_n$；外推好，表达力不如多频旋转 |                                        |
| T5 relative bias | 可学习的$b_{m-n}$                                                               | 相对、有参数、和 FlashAttention 摩擦大 |
| xPos             | RoPE + 随$\|m-n\|$ 衰减的稳定因子                                               | 旋转基础上再压长距离                   |
| NoPE             | 什么都不加                                                                        | 近层靠因果掩码偷位置，长程弱           |

---

## 5. 和加法位置嵌入的区别（对照本仓库）

这里没有另加一套 $e_m$。位置只进入转角 $\varphi_{m,i}$，再乘到已经投影好的 $q_m/k_n$ 的第 $i$ 对通道上。

同一套 $\omega_i$：

- 视频：位置取 **整数网格** $m^{\mathrm{v}}=(t,h,w)$，$s=1$
- 生成音频：位置取 **整数下标** $m^{\mathrm{a}}$，$s=s_a<1$（同样的 $m^{\mathrm{a}}$ 转得更慢，整段 $T_a$ 才铺满约 $T_v$ 个视频时间单位）
- Whisper：位置 $m^{\mathrm{r}}$，$s=s_r$ 更小，刻意不对齐生成音频

这就是不同种类、不同位置在代码里的全部差别：$s$、用哪一段 $i$、以及位置取 $t$ 还是 $m^{\mathrm{a}}$ 还是 $m^{\mathrm{r}}$。

---

## 6. 读代码地图

| 文件                                                                                                    | 看什么                                                               |
| ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `code/base_models/Wan2.1/wan/modules/model.py` 的 `rope_params` / `rope_apply`                    | 原版 3D RoPE，$s=1$；$m=(t,h,w)$，$i$ 分三段                   |
| `code/base_models/ovi/ovi/modules/model.py` 的 `rope_params(..., freqs_scaling)`、`rope_apply_1d` | $\varphi_{m,i}=m\cdot s\cdot\omega_i$；音频只转时间段的 $i$      |
| `code/base_models/ovi/ovi/configs/model/dit/audio.json`                                               | $s_a$ ← `temporal_rope_scaling_factor`                          |
| `HarmonizedVisualAudio/code/ovi_mask/.../audio.json`                                                  | 另加$s_r$ ← `ref_temporal_rope_scaling_factor`                  |
| `code/mycode/multitalk_layers.py` 的 `rope_1d`                                                      | $m$ 是身份轴，不是时间轴；$i$ 仍是通道对                         |
| HF`transformers/modeling_rope_utils.py`                                                               | `linear` / `dynamic` / `yarn` / `llama3` 怎么改 $\omega_i$ |

---

## 7. 容易混的三件事

1. **$\varphi_{m,i}=m\omega_i$ 就是标准 RoPE**，不是山寨。和论文的差别只在 $m$ 的语义（下标 / 网格 / 身份 / 缩放后的下标）。$i$ 始终是通道对，不换含义。
2. **Ovi 的 $s_a<1$ 和 HF 的 `factor>1` 互为倒数**，都是线性插值。不要看到 `freqs_scaling=0.19676` 就去套 YaRN 的 `factor`。
3. **缩放对齐的是角 $\varphi_{m,i}$，不是 token 个数。** $T_a$ 不会变成 $T_v$；cross-attn 仍是 $T_v\times$空间 对 $T_a$。只是第 $t$ 帧和第 $m^{\mathrm{a}}\approx 5t$ 个音频格在同一对 $i$ 上转角接近，softmax 才容易走对角。
