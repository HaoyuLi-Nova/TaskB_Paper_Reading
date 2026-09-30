# Wan: Open and Advanced Large-Scale Video Generative Models

- **Authors / Year / Venue**: Wan Team (Alibaba) / 2025 / arXiv 2503.20314
- **Link**: https://arxiv.org/abs/2503.20314 · 代码 `code/base_models/Wan2.1/`（公开仓只有推理）
- **One-line summary（一句话概括）**: 本文只整理 **Wan-I2V** 的训练与推理数据流；I2V 在 T2V DiT 上通道拼接噪声视频 + 帧 mask + 条件图 latent，再用 CLIP/T5 做 decoupled cross-attention。

记号（论文 §4.2 Rectified Flow）：$x_1$ 干净视频 latent，$x_0\sim\mathcal{N}(0,I)$ 噪声，$t\in[0,1]$，$X_t=t\,x_1+(1-t)\,x_0$。Wan-VAE 因果压缩：像素 $T=4n+1$（常用 81）→ latent $t_{\mathrm{lat}}=1+(T-1)/4$（21），通道 $c=16$，时序 stride $s=4$。

代码锚点：`wan/image2video.py` 造 $y$；`wan/modules/model.py` 里 `cat(X_t,y)` 与 CLIP∥T5 cross-attn。示意脚本：`code/mycode/wan_i2v_train_step.py`。

---

## 共享：条件 $y$（训练造一次 / 推理造一次，之后冻结）

1. 条件图 $f_0$（训练=视频第 0 帧；推理=用户图）。
2. $z_c=\mathrm{VAE}([f_0,\,0,\,0,\ldots])$，形状 `[16, t_lat, h, w]`。后面帧是 0-padding，不是真视频。
3. 像素时钟上手写二元 mask：I2V 仅 $f_0=1$，其余 0。因 81 不能整除 4，把第 0 帧 mask 重复 4 次再 `view` 成 `[4, t_lat, h, w]`（每个 latent 格对应打包进该格的 4 个像素帧）。语义祖先是 SEINE（ICLR 2024）；4 通道 rearrange 是适配 MagViT-v2 式因果 VAE。
4. 官方 $y=\mathrm{cat}(\mathrm{mask},\,z_c)$ → `[20, t_lat, h, w]`（`image2video.py`）。论文示意图顺序是 $z_t,z_c,m$；**加载权重必须跟官方：`[X_t | mask | z_c]`**。
5. $\mathrm{CLIP}(f_0)$、$\mathrm{T5}(\mathrm{caption})$ **不进 36 通道**，在 `WanI2VCrossAttention` 里与文本分开做 cross-attn。

I2V 时各 latent 格的 4 通道相同（第 0 格 `1111`，其余 `0000`）。首末帧 / 续写会在同一格内出现 `0001`、`1100`，所以不能压成每格 1 bit。联合训练（论文 §5.1）只改哪些像素帧 keep=1，拼法不变。

---

## 训练（每个样本只抽一个 $t$，没有时间步连环）

公开仓无 train loop，按 §4.2 + 与推理相同的 DiT 入口还原：

1. $x_1=\mathrm{VAE}(V)$：**整段真视频**，要还原的目标。不要和 $z_c$ 混：$x_1$ 后续格有运动，$z_c$ 后续格是 0-pad。
2. 按上节造 $y$、CLIP、T5 → **本 step 固定**。
3. 抽 $x_0\sim\mathcal{N}(0,I)$，再抽一个 $t$（论文 logit-normal）。
4. $X_t=t\,x_1+(1-t)\,x_0$（代数混合，**不是上一次 DiT 输出**）。
5. DiT 吃 $\mathrm{cat}(X_t,y)$（36 通道）+ CLIP + T5 + $t$，出速度 $u$，形状仍 `[16, t_lat, h, w]`。
6. $\mathcal{L}=\|u-(x_1-x_0)\|^2$，反传。结束。

`in_dim=2c+s=36`（相对 T2V 的 16），多出的 `patch_embedding` 零初始化。

---

## 推理（`WanI2V.generate`）

**循环外只做一次**：造 $y$、CLIP、T5；$X\leftarrow\mathcal{N}(0,I)$。

**每一步**（`for t in timesteps`，常用 UniPC，约 40 步）：

1. DiT 吃 $\mathrm{cat}(X,y)$ + CLIP + T5 + 当前 $t$，出 $u$。官方再跑一遍空/负向文本做 CFG：$u=u_\emptyset+s(u_c-u_\emptyset)$；$y$ 与 CLIP 两遍相同。
2. Scheduler 用 $u$ 更新 $X$（DiT **不直接**输出下一帧视频）。

循环内变的只有 $X$ 和标量 $t$；$y$ / CLIP / T5 冻结。

**循环后**：最后的 $X$ 过 Wan-Decoder → `[3, T, H, W]`。Decoder 不看 20 维 $y$。

---

## 36 通道切分（官方实现）

| 切片 | 内容 | 训练 | 推理循环 |
| --- | --- | --- | --- |
| `x[0:16]` | $X_t$ 噪声视频 latent | $t x_1+(1-t)x_0$ | 起点 `randn`，scheduler 更新 |
| `x[16:20]` | pack 后的 mask | 固定 | 固定 |
| `x[20:36]` | $z_c$ | 固定 | 固定 |

DiT 输出始终是 16 通道速度（或推理里经 CFG 的速度），不是 36 通道。
