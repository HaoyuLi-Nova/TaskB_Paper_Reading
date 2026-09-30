# AdaLN 学习笔记

- **源头**: Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT, ICCV 2023) [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- **AdaLN-Single**: Chen et al., PixArt-α [arXiv:2310.00426](https://arxiv.org/abs/2310.00426) §2.3
- **跨模态 AdaLN**: LTX-2 [arXiv:2601.03233](https://arxiv.org/abs/2601.03233) §3.1 / §3.1.2
- **祖先**: Perez et al., FiLM (AAAI 2018)；U-Net 扩散里的 AdaGN
- **本仓库代码**: `code/foundations/adaln/`（玩具）；Wan `wan/modules/model.py` 的 `WanAttentionBlock`；LTX-2 `ltx_core/model/transformer/adaln.py` + `transformer.py` + `ops.py`
- **一句话**: AdaLN **不是**新的标准化算法。统计量仍按 token、沿通道维 $D$ 算；它只把 LayerNorm 里固定的 $\gamma,\beta$，换成由条件 $c$（时间步 / 类别 / 另一模态）**线性映射**出来的 $\mathrm{scale}(c),\mathrm{shift}(c)$，再在残差上乘一条 **gate**。块内真正的矩阵乘法只发生在$c\mapsto 6D$ 向量这一步；作用到 token 上的全是逐元素乘加。

先分清两种乘法，再走一个 DiT 块。否则 $Wc$ 和 $x\odot(1+s)$ 会混成一件事。

---

## 0. 两种乘法，不要混

AdaLN 里同时出现两种乘，运算完全不同。

### 0.1 矩阵乘法（只发生在条件支路）

把条件向量 $c$ 变成一长串仿射系数，用的是 **`nn.Linear`**，也就是矩阵乘。

约定（与代码里 batch-first 张量一致）：一个样本是**行向量**。

$$
c \in \mathbb{R}^{B \times D_c},\qquad
W \in \mathbb{R}^{D_c \times (K D)},\qquad
\mathbf{b} \in \mathbb{R}^{K D}
$$

$$
u = c\, W + \mathbf{1}_B\,\mathbf{b}^{\top}
\quad\in\mathbb{R}^{B \times (K D)}
$$

- $B$：batch。每一行独立乘同一份 $W$。
- $D_c$：条件维（时间步嵌入的宽度，常等于隐维 $D$）。
- $K$：要切出来的组数。标准 DiT 块 $K=6$；LTX 若给文本交叉注意力也加 AdaLN 则 $K=9$。
- $D$：token 隐维。切完后每组都是 $D$ 维，才能和 $x$ 的最后一维对齐。

下标写法（第 $b$ 个样本、输出第 $j$ 维）：

$$
u_{b,j} = \sum_{i=1}^{D_c} c_{b,i}\, W_{i,j} + b_j
$$

这是 **对 $i$ 求和** 的真矩阵乘。$W$ 的每一列对应输出的一个通道；不同输出通道会**混合** $c$ 的所有分量。

**PyTorch 存的是 $W$ 的转置。** `nn.Linear(in_features=D_c, out_features=K*D)`：

| 对象       | 形状                        | 对应上面哪一个          |
| ---------- | --------------------------- | ----------------------- |
| `weight` | $(KD,\; D_c)$             | $W^{\top}$            |
| `bias`   | $(KD,)$                   | $\mathbf{b}$          |
| 前向       | `u = x @ weight.T + bias` | $u = cW + \mathbf{b}$ |

读 checkpoint 时不要把 `weight.shape[0]` 当成 $D_c$。

SiLU **不是**矩阵乘，是按元素：

$$
\mathrm{SiLU}(z)_{b,i} = z_{b,i}\cdot \sigma(z_{b,i}),\qquad
\sigma(\alpha)=\frac{1}{1+e^{-\alpha}}
$$

典型条件 MLP 是SiLU → Linear，所以完整的是

$$
u = \mathrm{SiLU}(c)\, W + \mathbf{b}
$$

先逐元素非线性，再一次矩阵乘。没有 $W_1 W_2$ 那种两层都在 AdaLN 投影里的必要；时间步嵌入 MLP 另算（§3.1）。

### 0.2 逐元素乘 / Hadamard 积（作用在 token 上）

仿射作用到隐状态时，**没有**对通道求和：

$$
(x \odot s)_{b,n,d} = x_{b,n,d}\cdot s_{b,n,d}
$$

PyTorch 里就是 `x * scale`。$s$ 往往是 $(B,1,D)$，靠广播补上 $N$ 这一维：

$$
x \in \mathbb{R}^{B\times N\times D},\quad
s,b,g \in \mathbb{R}^{B\times 1\times D}
$$

$$
\bigl(x \odot (1+s)\bigr)_{b,n,d} = x_{b,n,d}\cdot (1 + s_{b,1,d})
$$

同一个样本的 $N$ 个 token **共用**一套通道增益；不同通道 $d$ 的增益不同。这里**不对 $d$ 求和**，所以不是用 $s$ 再做一次线性层。

残差上的 gate 同样是 Hadamard：

$$
\bigl(x + g \odot y\bigr)_{b,n,d} = x_{b,n,d} + g_{b,1,d}\cdot y_{b,n,d}
$$

### 0.3 一句话对照

|              | 矩阵乘$cW$                       | Hadamard$x\odot s$              |
| ------------ | ---------------------------------- | --------------------------------- |
| 对谁求和     | 对条件维$D_c$ 求和               | 不对任何维求和                    |
| 混不混合通道 | 混合$c$ 的各维，产出 $KD$ 个数 | 第$d$ 通道只乘第 $d$ 个 scale |
| 出现位置     | 条件 MLP /`time_projection`      | `modulate`、残差 gate           |
| 形状变化     | $(B,D_c)\to(B,KD)$               | $(B,N,D)$ 保持不变              |

---

## 1. 符号与形状

全程默认 token 张量

$$
x \in \mathbb{R}^{B \times N \times D}
$$

| 符号                                                            | 形状                       | 含义                                                    |
| --------------------------------------------------------------- | -------------------------- | ------------------------------------------------------- |
| $B$                                                           | 标量                       | batch                                                   |
| $N$                                                           | 标量                       | token 数（图像是 patch 数，视频是$T\cdot H\cdot W$）  |
| $D$                                                           | 标量                       | 隐维 / 通道维。**Norm 的统计量只沿这一维算**      |
| $x_{b,n,:}$                                                   | $(D,)$                   | 第$b$ 个样本、第 $n$ 个 token 的向量                |
| $t$                                                           | $(B,)$ 或 $(B,N)$      | 扩散时间步。多数模型样本级；Wan / LTX 可 per-token      |
| $c$                                                           | $(B,D_c)$                | 条件行向量，通常$D_c=D$，由正弦时间嵌入再经 MLP 得到  |
| $K$                                                           | 标量                       | 切块组数，标准块为$6$                                 |
| $u$                                                           | $(B,KD)$                 | 一次 Linear 的输出，尚未切块                            |
| $\mu,\sigma^2$                                                | $(B,N,1)$                | 每个 token 自己的均值 / 方差，**不是** batch 统计 |
| $\gamma,\beta$                                                | $(D,)$                   | 普通 LN 的固定仿射，对所有 token、所有样本共用          |
| $s,b,g$（或 $\mathrm{scale},\mathrm{shift},\mathrm{gate}$） | $(B,1,D)$ 或 $(B,N,D)$ | AdaLN 的条件仿射；沿$N$ 广播（除非 per-token $t$）  |

不要把 $N$（序列长）和 LayerNorm 的normalized shape搞混：`nn.LayerNorm(D)` 的意思是**只对最后一维 $D$ 标准化**，$B$ 与 $N$ 都当独立样本轴。

记号：$s$ 是 scale，$b$ 是 shift（偏置），$g$ 是 gate。下文公式里写 $s_{\mathrm{msa}}(c)$ 表示由 $c$ 经线性层切出来、给 Self-Attn 用的那一组 scale，形状 $(B,1,D)$。

---

## 2. 一个 DiT 块里的数据流

### 2.1 对照：普通 Pre-Norm Transformer 块

Vaswani / GPT 风格（Pre-LN）只有**两条残差**，仿射与条件无关：

$$
\begin{aligned}
x &\leftarrow x + \mathrm{Attn}\bigl(\mathrm{LN}(x)\bigr) \\
x &\leftarrow x + \mathrm{FFN}\bigl(\mathrm{LN}(x)\bigr)
\end{aligned}
$$

$t$ 若要进来，早期做法是把时间步 embedding **加到** $x$ 上，或拼到序列里。那是一次 $(B,N,D)$ 上的加法，每个通道加同一个偏置；Attn / FFN **内部**看到的尺度并不随 $t$ 改。

### 2.2 标准 DiT 块（AdaLN-Zero，无交叉注意力）

条件 $c$ 在进块之前就算好，**块内不再看原始标量 $t$**，只反复用切出来的 6 组通道向量。

```
c = MLP(sin(t)) + class_emb          # (B, D_c)
                │
                ▼
         Linear: (B, D_c) → (B, 6D)     ← 本块唯一的矩阵乘
                │
                ├─ 切 3 组给 Self-Attn：shift / scale / gate（Wan、官方 DiT）
                └─ 切 3 组给 FFN：     shift / scale / gate

x  (B, N, D) ─────────────────────────────────────────┐ skip
 │                                                     │
 ├─ Norm（无仿射）  →  (B, N, D)                      │
 ├─ ⊙ (1 + scale_msa) + shift_msa                     │  ← Hadamard，不是矩阵乘
 ├─ Self-Attention                                    │
 ├─ ⊙ gate_msa ───────────────────────────────────────┤
 ▼                                                     ▼
 x                                                     │ skip
 │                                                     │
 ├─ Norm（无仿射）                                    │
 ├─ ⊙ (1 + scale_mlp) + shift_mlp                     │
 ├─ FFN                                                │
 ├─ ⊙ gate_mlp ───────────────────────────────────────┤
 ▼                                                     ▼
 out (B, N, D)
```

与官方 DiT / Wan 一致的前向（scale 写成 $1+\cdot$；切块顺序是 **shift, scale, gate**）：

$$
\begin{aligned}
h_{\mathrm{msa}}
  &= \mathrm{Norm}(x)\odot\bigl(1 + s_{\mathrm{msa}}(c)\bigr) + b_{\mathrm{msa}}(c) \\
x
  &\leftarrow x + g_{\mathrm{msa}}(c)\odot \mathrm{Attn}(h_{\mathrm{msa}}) \\[6pt]
h_{\mathrm{mlp}}
  &= \mathrm{Norm}(x)\odot\bigl(1 + s_{\mathrm{mlp}}(c)\bigr) + b_{\mathrm{mlp}}(c) \\
x
  &\leftarrow x + g_{\mathrm{mlp}}(c)\odot \mathrm{FFN}(h_{\mathrm{mlp}})
\end{aligned}
$$

三件必须同时记住：

1. **Pre-Norm**：先 Norm 再进 Attn / FFN；残差加的是**未**经过这次 Norm 的 $x$。
2. **gate 在残差支路上**：skip 永远是 $x$ 本身。$g=0$ 时整条支路消失，块变成恒等。
3. **仿射是通道级、通常是样本级**：$s,b,g$ 形状 $(B,1,D)$，同一条样本的 $N$ 个 token 共用一套。它们不编码第几个 patch，只编码当前噪声有多大 / 当前类别是什么。

### 2.3 文本条件 DiT 块（Wan / LTX 单流）

在 Self-Attn 与 FFN **之间**插入文本交叉注意力。Wan 的交叉注意力用普通 LN，**不**走 AdaLN：

$$
\begin{aligned}
x &\leftarrow x + g_{\mathrm{msa}}\odot \mathrm{SelfAttn}(\mathrm{AdaLN}(x; c)) \\
x &\leftarrow x + \mathrm{CrossAttn}\bigl(\mathrm{LN}(x),\; \mathrm{text}\bigr) \\
x &\leftarrow x + g_{\mathrm{mlp}}\odot \mathrm{FFN}(\mathrm{AdaLN}(x; c))
\end{aligned}
$$

对应 `WanAttentionBlock.forward`：`e[0:3]` 管 Self-Attn，`e[3:6]` 管 FFN，中间的 `cross_attn` 只吃 `norm3(x)`。

LTX 把交叉注意力也纳入 AdaLN 时，$K=6$ 扩成 $K=9$，多出来的 $(b_q, s_q, g_q)$ 调图像侧 Q，文本侧另有 $(b_{kv}, s_{kv})$。见 §6.1。

### 2.4 LTX-2 双流块（再加音视频交叉注意力）

单流内部仍是 §2.3；两流之间多一段 **A↔V Cross-Attn**，发生在文本交叉注意力之后、FFN 之前：

```
视频流 vx                         音频流 ax
   │                                 │
   ├ Self-Attn + 本模态 AdaLN        ├ 同左
   ├ Text Cross-Attn                 ├ 同左
   │                                 │
   ├──── A2V / V2A Cross-Attn ───────┤   ← 跨模态 AdaLN（§6.2）
   │                                 │
   ├ FFN + 本模态 AdaLN              ├ 同左
   ▼                                 ▼
```

### 2.5 形状跟踪（一次前向）

以 $B=2,\; N=8,\; D=16,\; D_c=16,\; K=6$ 为例：

| 步骤       | 张量                         | 形状                                                  | 运算                       |
| ---------- | ---------------------------- | ----------------------------------------------------- | -------------------------- |
| 输入       | $x$                        | $(2,8,16)$                                          | —                         |
| 条件       | $c$                        | $(2,16)$                                            | —                         |
| SiLU       | $\mathrm{SiLU}(c)$         | $(2,16)$                                            | 逐元素                     |
| Linear     | $u=cW+\mathbf{b}$          | $(2,96)$ | 矩阵乘，$W\in\mathbb{R}^{16\times 96}$ |                            |
| chunk      | $s_{\mathrm{msa}}$ 等 6 组 | 各$(2,16)$ 再 `[:,None,:]` → $(2,1,16)$        | 切片，不是乘法             |
| Norm       | $\mathrm{Norm}(x)$         | $(2,8,16)$                                          | 沿$D$ 统计，见 §4.1     |
| modulate   | $h$                        | $(2,8,16)$                                          | Hadamard + 广播            |
| Attn / FFN | 同$x$                      | $(2,8,16)$                                          | 内部另有 QKV / 两层 Linear |
| 残差       | $x + g\odot y$             | $(2,8,16)$                                          | 又一次 Hadamard            |

条件从进块到出块**不会**变成 $(B,N,D)$（除非时间步本身 per-token）。AdaLN 没有这个 patch 用另一套 $\gamma$的能力；空间结构仍由 Attention / RoPE 负责。

---

## 3. 从标量 $t$ 到 6 组仿射：逐步矩阵乘

这是笔记里最容易写错的一段。按官方 DiT / Wan的切块顺序写；玩具代码顺序不同，§5.3 对照。

### 3.1 时间步 → 条件向量 $c$

标量（或 per-token 标量）先变成正弦向量，再经过两层 Linear。以 Wan 为例（`time_embedding`）：

$$
t \in \mathbb{R}^{B}
\quad\xrightarrow{\text{sinusoidal}}
\quad
e_{\sin}\in\mathbb{R}^{B\times D_{\mathrm{freq}}}
$$

$$
c = \mathrm{SiLU}\!\bigl(e_{\sin}\, W_1 + \mathbf{b}_1\bigr)\, W_2 + \mathbf{b}_2
\in\mathbb{R}^{B\times D}
$$

| 矩阵    | 形状                         | 代码                                             |
| ------- | ---------------------------- | ------------------------------------------------ |
| $W_1$ | $(D_{\mathrm{freq}},\, D)$ | `time_embedding[0]`：`Linear(freq_dim, dim)` |
| $W_2$ | $(D,\, D)$                 | `time_embedding[2]`：`Linear(dim, dim)`      |

这一步与 AdaLN 的自适应还无关，只是把 $t$ 编成能做矩阵乘的向量。类条件 DiT 还会 **加** 上 `class_emb`（同样是 $(B,D)$ 的加法，不是 concat）。

### 3.2 条件向量 → 长向量 $u$（AdaLN 投影）

**AdaLN-Zero（每块一个 Linear，官方 DiT）**

$$
u = \mathrm{SiLU}(c)\, W_{\mathrm{ada}} + \mathbf{b}_{\mathrm{ada}}
\in\mathbb{R}^{B\times 6D}
$$

$$
W_{\mathrm{ada}}\in\mathbb{R}^{D\times 6D},\qquad
\mathbf{b}_{\mathrm{ada}}\in\mathbb{R}^{6D}
$$

零初始化：`nn.init.zeros_(W)` 且 `zeros_(b)`，于是开训 $u=\mathbf{0}$。

**AdaLN-Single / Wan（全网共享 Linear + 每块表）**

Wan 把上面这个 Linear **提到网络级**（`time_projection`），每块只留一张表：

$$
e_0 = \mathrm{SiLU}(c)\, W_{\mathrm{shared}} + \mathbf{b}_{\mathrm{shared}}
\in\mathbb{R}^{B\times 6D}
$$

reshape 成 $(B,6,D)$ 后，第 $\ell$ 块：

$$
e^{(\ell)} = E_{\mathrm{table}}^{(\ell)} + e_0
\in\mathbb{R}^{B\times 6\times D}
$$

$$
E_{\mathrm{table}}^{(\ell)}\in\mathbb{R}^{1\times 6\times D}
\quad\text{（Wan 的 ``modulation``；LTX / PixArt 的 ``scale_shift_table`` 是 }(6,D)\text{）}
$$

加法是逐元素的，**不是**又一次矩阵乘。共享 Linear 全网一份；表吸收这一层喜欢怎样调制。

LTX 的 `AdaLayerNormSingle.forward` 就是

```text
embedded_t = TimestepEmbedding(t)          # (B, D)
u = Linear(SiLU(embedded_t))               # (B, K*D)
```

块内再 `table[k] + u 切出来的第 k 段`。

### 3.3 切块：把 $6D$ 还原成 6 个 $D$ 维向量

$u$ 在最后一维上相邻切开，**没有重叠、没有再乘矩阵**：

$$
u = \bigl[\;
\underbrace{u_{0:D}}_{\text{组 0}}\ \big|\
\underbrace{u_{D:2D}}_{\text{组 1}}\ \big|\
\cdots\ \big|\
\underbrace{u_{5D:6D}}_{\text{组 5}}
\;\bigr]
$$

官方 DiT、Wan、LTX 单流（`chunk(6)` / `slice(0,3)` 再 `slice(3,6)`）的约定是：

| 下标 | 切片            | 符号                          | 用在                           |
| ---- | --------------- | ----------------------------- | ------------------------------ |
| 0    | $u_{:,0:D}$   | $b_{\mathrm{msa}}$（shift） | `Norm(x) * (1+s) + b` 的加项 |
| 1    | $u_{:,D:2D}$  | $s_{\mathrm{msa}}$（scale） | 乘在 Norm 后的$x$ 上         |
| 2    | $u_{:,2D:3D}$ | $g_{\mathrm{msa}}$（gate）  | 乘在 Self-Attn 输出上          |
| 3    | $u_{:,3D:4D}$ | $b_{\mathrm{mlp}}$          | FFN 前的 shift                 |
| 4    | $u_{:,4D:5D}$ | $s_{\mathrm{mlp}}$          | FFN 前的 scale                 |
| 5    | $u_{:,5D:6D}$ | $g_{\mathrm{mlp}}$          | 乘在 FFN 输出上                |

然后 `[:, None, :]`：$(B,D)\to(B,1,D)$，以便和 $(B,N,D)$ 广播。

**切块顺序是约定，不是数学。** 玩具 `AdaLNZero` 是 `scale, shift, gate`（scale 在前），与 Wan / 官方 DiT / LTX **相反**。对着实现读，不要凭记忆对 `e[0]`。

### 3.4 广播：$(B,1,D)$ 如何乘到 $(B,N,D)$

取 $B=1,N=2,D=2$，已经切好

$$
s = \begin{pmatrix} 0.1 & -0.2 \end{pmatrix},\quad
b = \begin{pmatrix} 0 & 1 \end{pmatrix}
$$

（写成 $(1,1,2)$ 后）以及

$$
\hat{x} =
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
\in\mathbb{R}^{1\times 2\times 2}
$$

$$
h = \hat{x}\odot(1+s)+b
=
\begin{pmatrix}
1\cdot 1.1 + 0 & 2\cdot 0.8 + 1 \\
3\cdot 1.1 + 0 & 4\cdot 0.8 + 1
\end{pmatrix}
=
\begin{pmatrix}
1.1 & 2.6 \\
3.3 & 4.2
\end{pmatrix}
$$

两个 token 的第 0 通道都乘 $1.1$、都加 $0$；第 1 通道都乘 $0.8$、都加 $1$。**没有** $s$ 与 $\hat{x}$ 之间的 $\sum_d$。

若误写成矩阵乘 $\hat{x}\, \mathrm{diag}(1+s)$，数值在 $N=1$ 时碰巧对，但那是每个 token 各自乘对角阵，实现里就是 Hadamard，不要再引入一个 $(D,D)$ 矩阵。

---

## 4. Norm 与 `modulate` 的公式

### 4.1 LayerNorm

对每一个 $(b,n)$，只沿 $D$。令 $x_{b,n}\in\mathbb{R}^{D}$：

$$
\begin{aligned}
\mu_{b,n}
  &= \frac{1}{D}\sum_{d=1}^{D} x_{b,n,d}
  = \frac{1}{D}\,\mathbf{1}^{\top} x_{b,n} \\
\sigma^{2}_{b,n}
  &= \frac{1}{D}\sum_{d=1}^{D}(x_{b,n,d}-\mu_{b,n})^{2} \\
\hat{x}_{b,n,d}
  &= \frac{x_{b,n,d}-\mu_{b,n}}{\sqrt{\sigma^{2}_{b,n}+\varepsilon}} \\
y_{b,n,d}
  &= \gamma_d\,\hat{x}_{b,n,d}+\beta_d
\end{aligned}
$$

向量形式（最后一个仿射才是对角的）：

$$
\hat{x}_{b,n}
  = \frac{x_{b,n}-\mu_{b,n}\mathbf{1}}{\sqrt{\sigma^{2}_{b,n}+\varepsilon}},
\qquad
y_{b,n}
  = \gamma \odot \hat{x}_{b,n} + \beta
$$

- $\mu,\sigma^2$ 形状 $(B,N,1)$：一共 $B\cdot N$ 套统计，互不影响。
- $\gamma,\beta$ 形状 $(D,)$：同一通道上所有 token 共用。这不是 per-token 仿射。
- 方差用总体方差（分母 $D$，`unbiased=False`），与 `nn.LayerNorm` 一致。
- **不是 BatchNorm**：BN 沿 $(B,N)$ 对每个通道求均值。LN 推理不依赖 batch 统计。
- $\mathbf{1}^{\top}x / D$ 是一次点积（求和），但权重固定为 $1/D$，不是学出来的 $W$。

手写实现见 `code/foundations/adaln/layernorm_demo.py`。DiT / AdaLN 路径上会把 `elementwise_affine=False`，只用 $\hat{x}$，把 $\gamma,\beta$ 的位置让给 $s(c),b(c)$。

### 4.2 RMSNorm

Zhang & Sennrich (2019)。不做均值中心化：

$$
\mathrm{RMS}(x)_{b,n}
  = \sqrt{\frac{1}{D}\sum_{d=1}^{D} x_{b,n,d}^{2}+\varepsilon},
\qquad
\hat{x}_{b,n}
  = \frac{x_{b,n}}{\mathrm{RMS}(x)_{b,n}}
$$

可选再乘固定 $\gamma$（无 $\beta$）。LTX-2 Transformer 的 AdaLN **前半步是 RMSNorm**，不是 LayerNorm：

```python
# ltx_core/.../ops.py  PytorchAdaZeroFunction
return rms_norm(x, eps=eps) * (1 + scale) + shift
```

与 LayerNorm 的差别只在标准化；后面的条件仿射公式完全一样。SD3 的 MM-DiT 还在 Q/K 上额外加 RMSNorm 控注意力熵，那是另一处，不是 AdaLN。

### 4.3 无仿射 Norm（AdaLN 的前半截）

$$
\mathrm{Norm}(x) \in \bigl\{\mathrm{LN}_{\gamma=\beta=\emptyset}(x),\; \mathrm{RMS}(x)\bigr\}
$$

```python
def layer_norm(x, eps=1e-6):
    mean = x.mean(-1, keepdim=True)                 # (B, N, 1)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps)

def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
```

`rsqrt` 是按元素的 $1/\sqrt{\cdot}$，不是矩阵逆。

### 4.4 FiLM / `modulate`（AdaLN 的后半截）

Perez et al. 的 Feature-wise Linear Modulation：

$$
\mathrm{FiLM}(x; s, b) = s\odot x + b
$$

DiT 系为了零初始化写成

$$
\mathrm{modulate}(x; s, b) = x\odot(1+s)+b
$$

```python
def modulate(x, scale, shift):
    return x * (1.0 + scale) + shift   # scale/shift: (B, 1, D)
```

$s=0$ 时仿射为只平移 $b$；$s=b=0$ 时就是恒等（作用在已经 Norm 过的 $x$ 上）。论文公式有时写 $\gamma\odot\mathrm{LN}(x)+\beta$，官方代码一律 $(1+\mathrm{scale})$。读代码以 $(1+\mathrm{scale})$ 为准。

把 $1+s$ 看成对角阵 $S=\mathrm{diag}(1+s_{b,1,:})\in\mathbb{R}^{D\times D}$，则对单个 token

$$
h_{b,n} = S\, \hat{x}_{b,n} + b_{b,1,:}
$$

这是 **对角** 线性变换，参数只有 $D$ 个，不是满的 $(D,D)$。实现里不会真的实例化 $S$。

---

## 5. AdaLN 家族

### 5.1 一般形式（还没有 Zero / gate）

$$
\mathrm{AdaLN}(x \mid c)
  = \mathrm{Norm}(x)\odot\bigl(1+s(c)\bigr)+b(c)
$$

其中 $s(c),b(c)$ 来自 §3.2–3.3 的线性层切块。名字里的自适应指的是仿射自适应于 $c$，不是统计量自适应于 $c$。

### 5.2 AdaLN-Zero（DiT 原版）

在 5.1 之上加两件事：

1. 每个子层再回归一个 **gate** $g(c)\in\mathbb{R}^{B\times 1\times D}$，Hadamard 乘在残差支路上。
2. 投影 $W_{\mathrm{ada}},\mathbf{b}_{\mathrm{ada}}$ **全零初始化**。

开训时：

$$
s=b=g=0
\;\Rightarrow\;
h=\mathrm{Norm}(x),\quad
x \leftarrow x + 0\cdot\mathrm{SubLayer}(h) = x
$$

深层从恒等残差打开，和 ResNet 的零初始化分支、ControlNet 的零卷积同类。玩具 `AdaLNZeroBlock` 在 `python demo.py` 第 2 节打印 $\|y-x\|_\infty=0$。

**为什么叫 Zero**：指初始化，不是没有仿射。

### 5.3 切块顺序对照（必须对着代码）

| 来源                                        | $K=6$ 的顺序                                                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 官方 DiT`adaLN_modulation` + `chunk(6)` | `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`                                     |
| Wan`WanAttentionBlock`                    | 同上：`e[0]=shift_msa, e[1]=scale_msa, e[2]=gate_msa, e[3]=shift_mlp, e[4]=scale_mlp, e[5]=gate_mlp` |
| LTX`get_ada_values(..., slice(0,3))`      | `shift_msa, scale_msa, gate_msa`；`slice(3,6)` 为 MLP 三组                                         |
| 本仓库玩具`AdaLNZero`                     | `scale_msa, shift_msa, gate_msa, ...`（**scale 在前**）                                        |

Wan 前向（与公式逐项对应）：

```python
e0 = self.time_projection(time_embedding(sin(t))).unflatten(1, (6, self.dim))
# e0: (B, 6, D)  ← 共享 Linear，矩阵乘发生在这里
e = (self.modulation + e).chunk(6, dim=1)          # modulation: (1, 6, D)
y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], ...)
x = x + y * e[2]
# ... cross_attn 用普通 LN ...
y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
x = x + y * e[5]
```

### 5.4 AdaLN-Single（PixArt-α / LTX）

动机：28 层各挂一个 `Linear(D, 6D)` 太贵。改成

$$
\begin{aligned}
e_{\mathrm{shared}}
  &= \mathrm{SiLU}(\mathrm{Embed}(t))\, W_{\mathrm{shared}} + \mathbf{b}
  && \in \mathbb{R}^{B\times KD} \\
\mathrm{ada}^{(\ell)}_{k,:}
  &= \mathrm{table}^{(\ell)}_{k,:} + e_{\mathrm{shared}}\text{ 的第 }k\text{ 段}
  && \mathrm{table}^{(\ell)}\in\mathbb{R}^{K\times D}
\end{aligned}
$$

- $W_{\mathrm{shared}}$ **全网络一份**（LTX 的 `AdaLayerNormSingle.linear`）。
- $\mathrm{table}^{(\ell)}$ 每块一张。
- 块内前向与 AdaLN-Zero **相同**，只是 $s,b,g$ 的来源换了。
- LTX 默认 $K=6$；`cross_attention_adaln=True` 时 $K=9$。

`adaln.py` 顶部注释写2 params × 3 norms，容易理解成shift/scale × {SA, FFN, output}。**单流块实际用法不是这样**：`slice(0,3)` / `slice(3,6)` 是 (shift, scale, gate) × {MSA, MLP}。最终输出头另有一张只有 2 行的 `scale_shift_table`（shift, scale，无 gate），不要和块内 6 组合并计数。

玩具：`AdaLNSingle`（共享）+ `AdaLNSingleBlock.scale_shift_table`（每块）。`demo.py` 第 3 节：两块吃同一份 `shared(t)`，只因 table 不同而输出分叉。

### 5.5 为什么扩散要用这一套

同一套权重必须服务整条轨迹：$t\approx 1$ 时 $x$ 几乎是噪声，网络该做大尺度、低细节的运动；$t\approx 0$ 时接近干净样本，该修纹理与对齐。U-Net 用 AdaGN（GroupNorm 的 $\gamma,\beta$ 由 $t$ 生成）解决这件事。DiT 把主干换成 Transformer 之后，对应接口就是 AdaLN：让 **Attn / FFN 入口处的通道增益随 $t$ 变**。

只把 $t$ 加到 token 上做不到同等强度：加法是平移；AdaLN 是先标准化再按 $t$ 重新标定。后者直接改每一层子模块看到的尺度。这就是 DiT 把 in-context 条件、交叉注意力、AdaLN 三种注入比完后留下 AdaLN-Zero 的原因。

> **AdaLN =（无仿射）LayerNorm / RMSNorm + 由条件线性映射生成的对角 FiLM +（Zero 变体里）残差 gate。**

---

## 6. 交叉注意力上的 AdaLN

### 6.1 文本交叉注意力（LTX `cross_attention_adaln`）

Q 来自图像 / 视频 token，KV 来自文本。打开该开关时，块表从 6 行扩到 9 行，`slice(6,9)` 给出 $(b_q,s_q,g_q)$。

LTX `apply_cross_attention_adaln`（此时 $x$ **已经**在 `post_sa_function` 里 RMSNorm 过，这里只做仿射）：

$$
\begin{aligned}
Q_{\mathrm{in}}
  &= x_{\mathrm{norm}}\odot(1+s_q)+ b_q \\
K_{\mathrm{in}},V_{\mathrm{in}}
  &= \mathrm{text}\odot(1+s_{kv})+ b_{kv} \\
\Delta
  &= \mathrm{CrossAttn}(Q_{\mathrm{in}}, K_{\mathrm{in}}, V_{\mathrm{in}}) \\
x
  &\leftarrow x + g_q \odot \Delta
\end{aligned}
$$

- $s_q,b_q,g_q$：该块 `scale_shift_table[6:9] +` 图像侧时间步经共享 MLP 的输出。
- 文本侧只有 2 组 $(b_{kv}, s_{kv})$，来自 `prompt_scale_shift_table` 形状 $(2,D)$。若启用 `prompt_adaln_single`，再叠 `prompt_timestep`（用 $\sigma$ 而不是去噪 $t$）；关掉则 K/V **与时间步无关**，可以跨去噪步做 KV cache。

Wan **没有**这一段：文本交叉注意力前面只是普通 `norm3`。

### 6.2 Cross-modality AdaLN（LTX-2 音视频交叉注意力）

论文 §3.1.2 的文字是：投影成 $Q,K,V$ **之后**，用本流时间步调 $Q$、用对方时间步调 $(K,V)$，再用对方时间步生成输出 gate。代码里实际是：先对**隐状态**做 RMSNorm + scale/shift，再进 Attention（Attention 内部才做 QKV 线性层）。先写代码真实在算的东西。

每流一张 5 行的表（`scale_shift_table_a2v_ca_video` 形状 $(5,D_v)$，音频侧 $(5,D_a)$）。A2V 用两表的 `slice(0,2)`，V2A 用两表的 `slice(2,4)`：

| 行   | 视频表$(5,D_v)$                  | 音频表$(5,D_a)$                  |
| ---- | ---------------------------------- | ---------------------------------- |
| 0–1 | A2V 时视频 Q 的`(scale, shift)`  | A2V 时音频 KV 的`(scale, shift)` |
| 2–3 | V2A 时视频 KV 的`(scale, shift)` | V2A 时音频 Q 的`(scale, shift)`  |
| 4    | A2V 残差 gate（加在$x_v$ 上）    | V2A 残差 gate（加在$x_a$ 上）    |

注意：这里 `get_av_ca_ada_values` 对前两维解包成 **`scale, shift`**，与单流 MSA 的 **`shift, scale, gate`** 顺序不同。

两套共享 MLP（`embedding_coefficient` 分别为 4 和 1）：

- `cross_scale_shift_adaln`：吃**本模态**时间步 → 长度 $4D$ 的向量，供行 0–3。
- `cross_gate_adaln`：吃**对方模态**的 $\sigma$（噪声水平）→ 长度 $D$ 的向量，供行 4。

A2V（视频吸收音频）代码对应的公式：

$$
\begin{aligned}
\tilde{x}_v
  &= \mathrm{RMS}(x_v)\odot(1+s_v(t_v))+ b_v(t_v) \\
\tilde{x}_a
  &= \mathrm{RMS}(x_a)\odot(1+s_a(t_a))+ b_a(t_a) \\
x_v
  &\leftarrow x_v + g(t_a)\odot \mathrm{Attn}_{A2V}(\tilde{x}_v,\; \tilde{x}_a)
\end{aligned}
$$

V2A 对称。$g(t_a)=0$ 时视频这一层完全不吸收音频。这就是两流扩散步或时间分辨率不一致时仍能同步的接口：不是把 $t_v$ 与 $t_a$ 加成一个向量再做一次 $W$，而是让对方的 $\sigma$ 去拧本侧残差的阀门。

论文还写过scale/shift 由另一流**隐状态**条件化。实现里条件是时间步 / $\sigma$，不是把对方 $x$ 拿去做 Linear。以代码为准。

论文写先 QKV 投影再 AdaLN、代码写先 AdaLN 再 QKV，二者一般**不等价**：

$$
\bigl((1+s)\odot \mathrm{RMS}(x)+b\bigr)W_Q
\quad\neq\quad
(1+s)\odot (x W_Q)+b
$$

左边是 LTX 实现；右边是 §3.1.2 的字面阅读。对角缩放与满矩阵 $W_Q$ 不可交换。

玩具 `CrossModalityAdaLN` 把这件事缩成三个小 MLP；`demo.py` 第 4 节：gate 置零后 $\|out-video\|_\infty=0$。

---

## 7. 对照总表

| 名称                 | 标准化            | 仿射怎么来（矩阵乘在哪）                                               | gate           | 谁在用                                       |
| -------------------- | ----------------- | ---------------------------------------------------------------------- | -------------- | -------------------------------------------- |
| LayerNorm            | 减均值 / 除标准差 | 固定$\gamma,\beta\in\mathbb{R}^{D}$，无 $c$                        | 无             | 普通 Transformer；AdaLN 里关掉仿射后当前半截 |
| RMSNorm              | 只除 RMS          | 可选固定$\gamma$                                                     | 无             | LLaMA；LTX AdaLN 的前半截；SD3 的 QK-Norm    |
| AdaGN                | GroupNorm         | $t$ 生成 $\gamma,\beta$                                            | 无             | 扩散 U-Net（SD 1.x / 2.x）                   |
| AdaLN                | LN 或 RMS         | $c\mapsto(s,b)$ 一次 Linear                                          | 可无           | 概念总称                                     |
| AdaLN-Zero           | 无仿射 LN         | **每块** `Linear(D, 6D)`，零初始化                             | 有             | 原版 DiT                                     |
| AdaLN-Single         | LN 或 RMS         | **共享** `Linear(D, KD)` + 每块 table 加法                     | 有             | PixArt-α、LTX、Wan 的`modulation` 表      |
| 文本 CA AdaLN        | 同上              | Q 用图像$t$（表的第 6–8 行）；KV 用 prompt 表                       | Q 侧有         | LTX`cross_attention_adaln`                 |
| Cross-modality AdaLN | RMS               | Q/KV 用本流$t$（4 段 Linear）；gate 用对方 $\sigma$（1 段 Linear） | 对方$\sigma$ | LTX-2 双流                                   |

MM-DiT（SD3）仍是 AdaLN 家族：两套权重（文本流 / 图像流），调制输入是 $t$ 的嵌入加上 **pooled** 文本向量 $c_{\mathrm{vec}}$；细粒度文本走拼接后的联合注意力，不靠 AdaLN 传 token 级文本。

---

## 8. 易混点

1. **AdaLN 不混合 token。** 统计量与普通 LN 一样沿 $D$。空间混合只发生在 Attention 里。
2. **$cW$ 不是 $x\odot s$。** 前者对 $D_c$ 求和、改变长度 $D_c\to KD$；后者不对任何维求和、保持 $(B,N,D)$。
3. **$(1+\mathrm{scale})$ 是实现约定。** 论文可能写 $\gamma\odot\mathrm{LN}(x)$。零初始化时必须保证起步乘项为 1，所以代码用 $1+s$。
4. **6 组切块顺序不统一。** Wan / 官方 DiT / LTX 单流是 `shift, scale, gate`；玩具 `AdaLNZero` 是 `scale, shift, gate`。跨模态表前两维又是 `scale, shift`。
5. **gate ≠ Attention 内部的 sigmoid gate。** AdaLN 的 gate 乘在残差上，形状 $(B,1,D)$。LTX 另有 `PytorchGatedAttention`（按头 `sigmoid`，形状 $(B,N,H)$），是另一条支路。
6. **条件通常不是 per-token。** 默认 $(B,1,D)$ 广播。Wan 在 Diffusion Forcing / per-token $t$ 时会把 $e$ 做成 $(B,N,6,D)$，那是时间步本身 per-token，不是 AdaLN 给了每个 patch 独立的满 $\gamma$ 矩阵。
7. **AdaLN-Zero 与 AdaLN-Single 前向公式相同。** 差的是 $s,b,g$ 从每块 MLP还是共享 MLP + 表来。不要把 Single 理解成只有一组仿射、没有 gate。
8. **不要用隐状态 $x$ 去乘 $W_{\mathrm{ada}}$。** $W_{\mathrm{ada}}$ 只乘条件 $c$。有人把 AdaLN 想成又一层作用于 $x$ 的 Linear，那会多出 $(B,N,KD)$ 的错误形状。

---

## 9. 代码锚点

| 要看的东西                       | 路径                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| 手写 LayerNorm 逐步形状          | `code/foundations/adaln/layernorm_demo.py`                                                   |
| AdaLN-Zero / Single / 跨模态玩具 | `code/foundations/adaln/adaln.py`、`demo.py`                                               |
| Wan 一块的 6 组 modulate         | `code/base_models/Wan2.1/wan/modules/model.py` → `WanAttentionBlock`                      |
| Wan 共享`time_projection`      | 同文件`WanModel`：`Linear(dim, dim*6)`                                                     |
| Wan 数据流展开                   | `code/base_models/Wan2.2/mycode/dit_trace.py`                                                |
| LTX AdaLN-Single 定义            | `code/base_models/ltx2/.../transformer/adaln.py`                                             |
| LTX 切块 + 跨模态前向            | `.../transformer/transformer.py` → `get_ada_values` / `BasicAVTransformerBlock.forward` |
| LTX 真正的 RMS×(1+s)+b          | `.../transformer/ops.py` → `PytorchAdaZeroFunction`                                       |

运行玩具：

```bash
cd code/foundations/adaln
python layernorm_demo.py   # LN 的 (B,N,D) 统计
python demo.py             # AdaLN-Zero 恒等、Single、跨模态门控
```
