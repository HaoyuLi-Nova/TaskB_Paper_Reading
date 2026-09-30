# Bind-Your-Avatar：动机、创新点与代码实现

- **Authors / Year / Venue**: Yubo Huang, Weiqiang Wang, Sirui Zhao, Tong Xu, Lin Liu, Enhong Chen / 2025 / ICCV 2025（arXiv:2506.19833）
- **Link**: [https://arxiv.org/abs/2506.19833](https://arxiv.org/abs/2506.19833)
- **Project**: [https://yubo-shankui.github.io/bind-your-avatar/](https://yubo-shankui.github.io/bind-your-avatar/)
- **官方代码（本仓库）**: [`code/audio_driven/bindyouravatar/`](../../../code/audio_driven/bindyouravatar/)（目录名跟 GitHub，slug 是 `bind_your_avatar`）
- **批判精读**: 同目录 [`critique.md`](critique.md)
- **一句话**: 在 CogVideoX MM-DiT 上，用去噪中途预测的 3D 人物软掩码把谁和哪路音频乘开，做同场景双人说话视频；不改音色。

---

## 1. 动机

音频驱动 talking head 已经能把**一张参考脸 + 一路语音**做成口型同步的单人视频，但主流设定还有两层限制：

1. **构图依赖 inpainting 首帧**：第一帧把人钉在画面里，后面只是让这张脸说话。
2. **两人对话常被做成左右拼接**：各生成一条单人视频再拼，没有共享背景、站位、遮挡和近距离互动，不是同场景。

作者把任务收成 **Multi-Talking-Character Video Generation**：

- 输入：文本（环境 / 站位 / 动作）+ \(n\) 张参考脸 + \(n\) 路语音 + 可选 inpainting 首帧 + 可选音频–人物矩阵 \(\mathbf{A}^{\mathrm{ac}}\)。
- 输出：同一空间里同时在场的多人说话视频。
- 实验固定 \(n=2\)。额外希望：**重叠说话**、**可以没有 inpainting 首帧**（靠文本构图）。

叙事上要绑的是两件事：

- **谁**（身份跟哪张参考脸走）
- **说什么**（哪路音频打到哪个视觉区域）

静态首帧框复制到所有 \(t\) 会在走动、换位、两人靠近时把音频灌进错的脸。这和 MultiTalk / AnyTalker / HarmonizedVA 面对的是同一类**空间绑定**问题，不是人造题目。

范围边界（读的时候要看清）：

- 目标停在口型 / 表情 / 头姿跟**对的那路声**。
- **不改音色**，也不是音视频联合生成。HarmonizedVA声要像这个人是另一条问题。

---

## 2. 创新点

相对换底座 / 换对时这类组合，本文真正增量在绑定怎么写、掩码从哪来。

### 2.1 把 who 和 what 拆成两个矩阵再乘

预测时空软掩码 \(\mathbf{M}\)，展平为人物–视觉矩阵 \(\mathbf{A}^{\mathrm{cv}}\in\mathbb{R}^{n\times S}\)。音频–人物 \(\mathbf{A}^{\mathrm{ac}}\in\mathbb{R}^{n\times n}\) 单独给出（单位阵 = 第 \(i\) 路声对人 \(i\)）。音频–视觉路由：

\[
\mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\,\mathbf{A}^{\mathrm{cv}}.
\]

重叠说话 = 两路 \(\mathbf{e}_a\) 同时在，乘到不同空间支撑集上。对调两路音频只改 \(\mathbf{A}^{\mathrm{ac}}\)，不必重训分割。

注入分两步（先脸后声）：

\[
\mathbf{v}'=\mathbf{v}+\mathbf{A}^{\mathrm{cv}}\cdot\mathrm{FaceCrossAttn}(\mathbf{e}_c,\mathbf{v}),
\]
\[
\mathbf{v}''=\mathbf{v}'+\mathbf{A}^{\mathrm{av}}_{\mathrm{inflate}}\cdot\mathrm{AudioCrossAttn}(\mathbf{e}_a,\mathbf{v}').
\]

### 2.2 Intra-Denoise：去噪中途预测 3D mask

三种路由是方法进化，不是三个并列产品：

| 名称 | 掩码从哪来 | 失败 / 代价 |
| --- | --- | --- |
| Pre-Denoise | 首帧检脸，2D 框复制所有 \(t\) | 人一动就偏；框太粗，靠近会重叠 |
| Post-Denoise | 先去噪出无声粗视频 → 每帧检脸 → 再去噪一遍 | 绑定准，推理两轮，贵 |
| **Intra-Denoise（采用）** | 当前层视觉 token + 脸 embedding 预测 3D mask，一轮前向 | 要训 router；早期噪声大，靠 teacher-forcing |

不是从 RGB 另训一个分割网络，而是复用**已预训练的 face cross-attn 的 Q/K**（画面格子已经能对上脸），在这组相关性上长出时空图。

### 2.3 几何先验，而不是每格独立分类

对照 Ingredients 的 token–人物相关性：本文加 3D 位置、空间 / 时间 / 跨人注意力，再用光滑损失和层间一致损失。意图是 mask 成块、沿时间跟着人走、同一格不要同时是两个人。

### 2.4 不是新故事的部分

通道拼接参考 / inpainting、CLIP + InsightFace + Q-Former、Wav2Vec 再对齐到 \(T'\)、LoRA 插音频，都是 talking-head / ConsisID 常规组合。底座是 CogVideoX，不是 Wan。

和 MultiTalk 的差：MultiTalk 绑定靠 amap → L-RoPE（软身份轴）；本文绑定靠可监督掩码挡 cross-attn。一个在旋转里做人，一个在注意力支撑集上做人。

---

## 3. 骨干数据流（实现口径）

代码入口：[`models/transformer.py`](../../../code/audio_driven/bindyouravatar/models/transformer.py) 的 DiT `forward`；router：[`models/router.py`](../../../code/audio_driven/bindyouravatar/models/router.py)。

默认分辨率（注释与硬编码一致）：RGB \(480\times 720\)、约 49 帧 → 3D VAE → latent \(T'=13,\ H'=60,\ W'=90\)；patch \(\tau=2\) → token 网格 \(30\times 45\)。

\[
S = 13\times 30\times 45 = 17550.
\]

条件怎么进：

| 模态 | 实现 | 形状（一条视频、两人） |
| --- | --- | --- |
| 文本 | T5，与视觉 token 拼进 MM-DiT | `[B, 226, 3072]` |
| 视觉 | patch_embed 后丢掉文本段 | `[B, 17550, 3072]` |
| 脸 | `LocalFacialExtractor`（ConsisID 式 Q-Former） | 每人 `[32, 2048]`，拼成 `[2, 32, 2048]` |
| 音频 | Wav2Vec → sliding window → `proj_in`，时间对齐到 13 格 | `[B, 2, 13, 32, 768]` |
| 音频–人物 | `af_matrix` | `[B, 2, 2]` |

Face / audio cross-attn **每 2 层 DiT 插一次**（`cross_attn_interval=2`）。CogVideoX-5B 约 42 层，故 `num_ca=21`，router 为这 21 层各备一套 `to_q` / `to_k`。

**Batch=16 的含义：** 训练论文写 batch 16，是 16 条视频。Router **不把 16 条合成一个大 batch**，而是 `for j in range(B)` 逐条调用。注释里的 `16` 是 **attention 头数**，不是视频 batch。下面形状全部是16 次里的一次。

---

## 4. Intra-Denoise 实现（按调用顺序）

### 4.1 调用点：画面复制到两人

[`transformer.py`](../../../code/audio_driven/bindyouravatar/models/transformer.py) 在第 `i` 层 DiT 之后（`i % cross_attn_interval == 0`）：

```python
cur_valid_face_emb = valid_face_emb[j]          # [2, 32, 2048]
# 2=人, 32=该人 face query, 2048=Q-Former 维
sub_img = sub_img.repeat(2, 1, 1)               # [17550, 3072] → [2, 17550, 3072]
# 两人看到同一套视觉 token
id_feat, weight_output, q_out, k_out = \
    self.perceiver_cross_attention[ca_idx](cur_valid_face_emb, sub_img)
routing_logit_pred = self.router(
    weight_output, q_out, k_out, ca_idx, self.is_teacher_forcing
)
# routing_logit_pred [1, 17550, 2]
```

`valid_face_emb` 在 `forward` 开头就算好，是 `list` 长 \(B\)，每个元素两人的 face token。`ca_idx` 是第几个 face-ca 层（0…20）。

### 4.2 Face Cross-Attn：产出 Q/K 和按脸加权的视觉特征

[`PerceiverCrossAttention.forward(x, latents)`](../../../code/audio_driven/bindyouravatar/models/router.py)。**调用时 `x=脸`，`latents=视觉 token`**（类 docstring 写反了）。

```python
x = self.norm1(x)            # LayerNorm(2048): [2, 32, 2048]
latents = self.norm2(latents)  # LayerNorm(3072): [2, 17550, 3072]

q = self.to_q(latents)       # Linear(3072→2048): [2, 17550, 2048]
# Q 来自画面：这个格子长什么样
k, v = self.to_kv(x).chunk(2, dim=-1)
# Linear(2048→4096) 再切两半: K/V 各 [2, 32, 2048]
# K/V 来自脸：这个人长什么样

q = reshape_tensor(q, 16)    # [2, 16, 17550, 128]  人×头×视觉格×头维
k = reshape_tensor(k, 16)    # [2, 16, 32, 128]
v = reshape_tensor(v, 16)

q_out = q.clone().detach()   # router 用这份，梯度不回传到 face attn
k_out = k.clone().detach()

scale = 1 / dim_head ** 0.25
weight = (q * scale) @ (k * scale).transpose(-2, -1)
# [2, 16, 17550, 32]：每格对 32 个 face token 的未归一化分数
weight_out = weight.clone().detach()   # 传给 router，但 router.forward 不用
weight = softmax(weight, dim=-1)       # 在 32 个 face token 上归一化
out = weight @ v                       # [2, 16, 17550, 128]
out = out.permute(0, 2, 1, 3).reshape(2, 17550, 2048)
id_feat = self.to_out(out)             # Linear(2048→3072): [2, 17550, 3072]
```

`reshape_tensor`：`[B, L, 2048] → view(B, L, 16, 128) → transpose → [B, 16, L, 128]`。

返回四件套：

| 张量 | 形状 | 谁用 |
| --- | --- | --- |
| `id_feat` | `[2, 17550, 3072]` | 注入：按 mask 把两人脸特征混回画面 |
| `weight_output` | `[2, 16, 17550, 32]` | 传入 router，**未使用** |
| `q_out` | `[2, 16, 17550, 128]` | router 真正输入（画面侧） |
| `k_out` | `[2, 16, 32, 128]` | router 真正输入（脸侧） |

Q/K detach 对应论文：router 图从去噪路上断开，避免为了好分类把视觉内容洗掉。

### 4.3 Router：从 Q/K 收到每格 512 维相关性

[`MultiIPRouter.forward(weight, q_out, k_out, layer_idx, is_teacher_forcing)`](../../../code/audio_driven/bindyouravatar/models/router.py)。`weight` 与 `is_teacher_forcing` **forward 内均未使用**；`layer_merge` 建了但没接到计算图。实现比附录更窄：就是复用 face-attn 的 Q/K，再投影、再点积。

```python
num_id = q_out.size(0)                              # 2

q = q_out.permute(0, 2, 3, 1).reshape(2, 17550, -1) # [2, 17550, 2048]
# 16 头×128 拼回；每个格子、每个人一条完整 Q
k = k_out.permute(0, 2, 3, 1).reshape(2, 32, -1)    # [2, 32, 2048]

q = self.to_q[layer_idx](self.norm_q(q))            # 该层自己的 Linear(2048→2048)
k = self.to_k[layer_idx](self.norm_k(k))            # 21 层各一套，不是 face attn 的 to_q

q = reshape_tensor(q, 16)                           # [2, 16, 17550, 128]
k = reshape_tensor(k, 16)                           # [2, 16, 32, 128]

q_k_weight = q @ k.transpose(-2, -1)                # [2, 16, 17550, 32]
# 在 router 的 Q/K 空间里重新算格子 vs 32 个 face token
q_k_weight = q_k_weight.permute(0, 2, 3, 1)         # [2, 17550, 32, 16]
q_k_weight = q_k_weight.reshape(2, 17550, -1)       # [2, 17550, 512]
# 32×16=512：每格、每人一条我和这张脸有多像的描述
q_k_weight = self.norm(q_k_weight)                  # LayerNorm(512)
```

被注释掉的 `softmax(dim=0)` 原本会在**人物维**归一化，现已关掉。同一格两人分数独立，背景两侧都可以接近 0。

### 4.4 折回 3D 体积 + 加性位置（代码叫 3D rope，实际是正弦加法）

```python
q_k_weight = q_k_weight.reshape(num_id, 13, 45, 30, -1)
# [2, 13, 45, 30, 512]
q_k_weight = q_k_weight + self.pos_emb
# pos_emb [13, 45, 30, 512]：t / h / w 三路正弦拼到 512，不够零填充
```

CogVideoX token 顺序实际是 \(T\times 30\times 45\)（高 30、宽 45），router 硬编码 `height=45, width=30`。进出 flatten 顺序一致，**第 \(s\) 个输出仍对应第 \(s\) 个视觉 token**；空间注意力的邻居按转置网格算。

### 4.5 四层时空注意力

[`SpatialTemporalAttentionBlock`](../../../code/audio_driven/bindyouravatar/models/router.py)，输入输出都是 `[2, 13, 45, 30, 512]`，`mlp_ratio=1`。每层四步残差：

**空间。** `(人, 帧)` 收成 batch，序列 = 全部空间格：

```python
x_space = x.reshape(2 * 13, 45 * 30, 512)   # [26, 1350, 512]
x = x + spatial_attn(norm1(x_space)).reshape(2, 13, 45, 30, 512)
```

同一人、同一帧里格子互相看，把碎响应连成区域。

**时间。** 固定人、固定像素，沿 13 格：

```python
x_temp = x.permute(0, 2, 3, 1, 4).reshape(2 * 45 * 30, 13, 512)  # [2700, 13, 512]
x = x + temporal_attn(norm2(x_temp)).reshape(2, 45, 30, 13, 512).permute(0, 3, 1, 2, 4)
```

人走动时 mask 沿时间连续。

**跨人。** 同一时空格上，人 0 与人 1 互看（序列长度 = 2）：

```python
x_id = x.permute(2, 3, 1, 0, 4).reshape(45 * 30 * 13, 2, 512)  # [17550, 2, 512]
x = x + multi_id_attn(norm3(x_id)).reshape(45, 30, 13, 2, 512).permute(3, 2, 0, 1, 4)
```

这是 Identity Exclusivity 在实现里的位置：同一格两人特征交互，逼这块别同时是两个人。

**FFN。** 每个 `(人, t, h, w)` 独立 `Linear(512→512)→GELU→Linear(512→512)`。

四层堆叠后，512 维已带几何 / 时间 / 跨人信息，不再是孤立的 \(QK^\top\)。

### 4.6 投影成软掩码 \(\mathbf{A}^{\mathrm{cv}}\)

```python
q_k_weight = q_k_weight.reshape(2, -1, 512)   # [2, 17550, 512]
output = self.final_proj(q_k_weight)          # Linear(512→1)+Sigmoid → [2, 17550, 1]
return output.permute(2, 1, 0)                # [1, 17550, 2]
```

最后一维是人 0 / 人 1 的 \((0,1)\) 分数。**人物维没有 softmax**。论文写背景当第 \(n+1\) 类，代码里背景就是 \((0,0)\)，不是单独一类。

Batch=16 时 16 次结果 `cat` 成 `[16, 17550, 2]`。

---

## 5. 掩码如何改数据

真正的路由是两处按 token 的 batched 矩阵乘：`[S, 1, 2] @ [S, 2, 3072]`。前面全部在造这个 `[S, 2]` 的 \(M\)。

### 5.1 人脸：按 \(M\) 混合两路 `id_feat`

```python
# routing_logit: [1, 17550, 2]   id_feat: [2, 17550, 3072]
mask_id_feat = routing_logit.transpose(1, 0) @ id_feat.transpose(1, 0)
# [17550, 1, 2] @ [17550, 2, 3072] → [17550, 1, 3072]
mask_id_feat = mask_id_feat.transpose(1, 0)   # [1, 17550, 3072]
hidden_states = hidden_states + local_face_scale * cat(mask_id_feats)
```

格子 \(s\)：

\[
\mathbf{v}'[s]=\mathbf{v}[s]+\alpha\big(m_0[s]\,\mathrm{Face}_0[s]+m_1[s]\,\mathrm{Face}_1[s]\big).
\]

人 0 的脸上 \(M\approx[1,0]\) 只加自己的脸特征；背景 \([0,0]\) 两边都不加。

### 5.2 音频：先 \(\mathbf{A}^{\mathrm{ac}}\mathbf{A}^{\mathrm{cv}}\)，再 inflate

```python
batch_routing_logits = torch.cat(routing_logits, dim=0)        # [B, 17550, 2]
av_matrix = af_matrix @ batch_routing_logits.transpose(-2, -1) # [B, 2, 17550]
av_matrix = av_matrix.transpose(-2, -1)                        # [B, 17550, 2]
# 即 A_av = A_ac A_cv

audio_feat = self.audio_model(...)   # 画面同样 repeat 到 2 人: [2, 17550, 3072]

routing_logit = routing_logit.unsqueeze(0)   # [1, 17550, 2]
routing_logit = routing_logit[:, :, [1, 0]]  # 人物维对调 [m0,m1]→[m1,m0]
routing_logit = 1 - routing_logit            # → [1-m1, 1-m0]  即 inflate

mask_audio_feat = routing_logit.transpose(1, 0) @ audio_feat.transpose(1, 0)
hidden_states = hidden_states + cat(mask_audio_feats)   # 无额外 scale
```

inflate 后：人 0 的音频打到 \(1-m_1\) 高的地方 = **不是人 1**（人 0 + 背景 + 头肩）；人 1 同理。口型仍落在自己脸上，身体也能跟着动，同时尽量不灌进另一张嘴。被注释掉的 `<0.3` 置零未开。

单位阵 `af_matrix` 时，乘 \(\mathbf{A}^{\mathrm{ac}}\) 不换人，只是把布局变成格子 × 人；对调两路音频就改这个 \(2\times 2\)。

---

## 6. 训练：teacher-forcing 与损失

### 6.1 注入用 GT，预测只吃损失

联合训 router 和去噪容易塌成token 很好分类但没有视觉内容，早期 mask 错还会让条件注不进去。因此：

| | 挡住 face / audio attn 的 mask | 给 router 的监督 |
| --- | --- | --- |
| 训练 `is_teacher_forcing=True` | SAM2 转成的 `index_mask`（加噪 / dropout） | `routing_logit_pred` 对 GT 做 BCE |
| 推理 | `routing_logit_pred` | 无 |

`index_mask[b, s] ∈ {0, 1, -1}`（人 0 / 人 1 / 背景）→ one-hot 式 `[1,0]` / `[0,1]` / `[0,0]`。然后：

1. `view(1, 13, 30, 45, 2)`，**时间维 max 再复制回 13 帧**：监督更偏这个空间位置曾经是谁的并集，不是严格逐帧。
2. 10% 元素改成均匀随机，再加 \(\mathcal{N}(0, 0.1^2)\)，clamp 到 \([0,1]\)。
3. 以 `index_mask_drop_prob` 整表置零（学无路由时的退化）。

DiT 学的是GT 路由下如何用脸和声；router 是另训的分割头。测试才第一次用预测 mask，存在 train/test 路由分布差。详见 [`critique.md`](critique.md) §3.5。

Q/K 在 face attn 里已经 detach，所以 BCE **不会**经 Q/K 回传到 DiT 主干。

### 6.2 Router 损失（代码口径）

论文写 \(\mathcal{L}_r+\lambda_{\mathrm{st}}\mathcal{L}_{\mathrm{st}}+\lambda_l\mathcal{L}_{\mathrm{layer}}\)，权重 \(1 / 0.001 / 8\)。实现对应：

- \(\mathcal{L}_r\)：每层 `bce_loss(pred, teacher)`，再对 21 个 face-ca 层平均。论文说交叉熵（背景一类）；代码是 **逐人 sigmoid BCE**，背景靠两侧都接近 0。
- \(\mathcal{L}_{\mathrm{layer}}\)：各层 mask 在 dim=层 上的方差（`consistency_loss`）。
- \(\mathcal{L}_{\mathrm{st}}\)：把 `[17550, 2]` 折成 `[13, 45, 30, 2]`，时间差分 L2 + 高/宽差分 L2。

另有 `spatial_distribution_loss` / `id_distribution_loss`，鼓励空间不要塌成一点、两人占用别完全重叠。

### 6.3 三阶段（论文）

1. 身份 / inpainting，无音频；50% 丢首帧。
2. 加音频，DiT 上 LoRA。
3. 加 router，只训 router + 两个 cross-attn + LoRA。单人数据把条件复制 \(n\) 份冒充多人。步数 1万 / 4万 / 1万，batch 16，lr \(10^{-5}\)。

---

## 7. 用一个格子把数字对上

取第 \(j\) 条视频、token \(s\)（例如 \(t=5,\ h=10,\ w=20\)，\(s=5\cdot 30\cdot 45+10\cdot 45+20\)）：

1. `repeat` 后，人 0 和人 1 都拿到同一个 \(\mathbf{v}[s]\in\mathbb{R}^{3072}\)。
2. Face attn：Q 来自 \(\mathbf{v}[s]\)，K/V 来自该人 32 个 face token → `id_feat[0,s]`、`id_feat[1,s]`，以及 detach 的 `q_out[:, :, s, :]`。
3. Router：该格 16 头 Q 与 32 个 K 点积 → 512 维 → 加 \((t,h,w)\) → 看空间邻居、时间邻居、另一个人在同一格上的 512 维 → sigmoid 得 \((m_0, m_1)\)。
4. 人脸：\(\mathbf{v}[s]\mathrel{+}=\alpha(m_0\cdot\mathrm{id\_feat}_0[s]+m_1\cdot\mathrm{id\_feat}_1[s])\)。
5. 音频：先按 `af_matrix` 换轨，再变成 \(1-m_1\)、\(1-m_0\) 去乘两路 `audio_feat[s]`。

---

## 8. 论文表述 vs 代码

| 论文 | 代码 |
| --- | --- |
| 3D RoPE | 加性 3D 正弦位置，不是旋转 |
| mask 人物维 softmax / 背景第 \(n+1\) 类 | 每人物独立 Sigmoid；背景 = \((0,0)\) |
| 复用 face attn 的注意力权重 | 只复用 **detach 后的 Q/K**，在 router 里重新 \(QK^\top\)；传入的 `weight` 未用 |
| \(\mathcal{L}_r\) 交叉熵 | 逐人 BCE |
| 推理 inflate + 低置信丢掉 + 聚类光滑 | 只实现了对调再 `1-M`；阈值置零被注释掉 |
| token 网格 \(H'\times W'\) | CogVideoX 为 \(30\times 45\)；router reshape 写成 \(45\times 30\) |

这些不影响输入输出是什么，但复现或往 Wan/Ovi 上迁时要对着代码而不是附录。

---

## 9. 对 HarmonizedVA 的读法

- 可迁：\(\mathbf{A}^{\mathrm{av}}=\mathbf{A}^{\mathrm{ac}}\mathbf{A}^{\mathrm{cv}}\)、Intra-Denoise在去噪中途用 Q/K 长 3D mask、音频 inflate（\(1-M_{\mathrm{other}}\)）。
- 不要迁：teacher-forcing 把预测 mask 是否真能驱动去噪留到推理——若自己做联合训练，这是隐患不是细节。
- 本文不回答音色；绑定准了，声仍可以不像这个人。
