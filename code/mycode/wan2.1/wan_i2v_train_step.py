"""
Wan2.1 I2V 训练一步（对照论文 §4.2 Flow Matching + §5.1 I2V，以及官方推理代码）。

官方仓库没有放出 train loop。训练时 DiT 的入口和推理相同，见：
  wan/image2video.py          构造 y = cat(mask, z_c)
  wan/modules/model.py:530    x = cat(X_t, y)  → 36 通道
  wan/modules/model.py:561    CLIP 与 T5 在 cross-attn 里拼接，不进 36 通道

和推理的本质差别：训练每个样本只抽 **一个** t，X_t 由真视频 latent 与噪声线性混合，
不是上一次 DiT 的输出。

尺寸刻意缩小，只为看清形状。因果 VAE 仍是 4n+1 帧。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 缩小版：T=9 → t_lat=3，空间 16→2。通道仍用论文的 16 和 mask 的 4。
C, T, H, W = 3, 9, 16, 16
c, s = 16, 4
t_lat, h, w = 1 + (T - 1) // s, H // 8, W // 8  # 3, 2, 2
T5_LEN, CLIP_LEN, CLIP_DIM, T5_DIM = 512, 257, 1280, 4096


def causal_vae_encode(video):
    """假 Wan-VAE：空间 /8，首帧不压时间，后面每 4 帧平均成 1 格，再铺成 16 通道。
    video: [3, T, H, W] → [16, t_lat, h, w]
    """
    pix = F.avg_pool3d(video.unsqueeze(0), (1, 8, 8)).squeeze(0)  # [3, T, h, w]
    z0 = pix[:, :1]
    z_rest = pix[:, 1:].reshape(C, t_lat - 1, s, h, w).mean(2)
    z = torch.cat([z0, z_rest], 1)
    z = z.repeat((c + C - 1) // C, 1, 1, 1)[:c]
    return z


def pack_frame_mask(keep):
    """官方 image2video.py:210-217。keep: [T] 的 0/1 → mask [4, t_lat, h, w]。"""
    m = keep.view(1, T, 1, 1).expand(1, T, h, w).clone()
    m = torch.cat([m[:, :1].repeat_interleave(s, 1), m[:, 1:]], 1)  # [1, 4n+4, h, w]
    m = m.view(1, t_lat, s, h, w).transpose(1, 2)[0]
    return m


def i2v_keep():
    k = torch.zeros(T)
    k[0] = 1
    return k


def flf2v_keep():
    """first_last_frame2video.py: 首末帧为 1。"""
    k = torch.zeros(T)
    k[0] = k[-1] = 1
    return k


def continuation_keep(n_given=3):
    k = torch.zeros(T)
    k[:n_given] = 1
    return k


class TinyI2VDiT(nn.Module):
    """占位：官方是 40 层 WanModel。这里只证明**36 通道进、16 通道速度出**。"""

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv3d(c + c + s, c, kernel_size=1)  # 36 → 16，对应 patch_embedding 加宽

    def forward(self, x36, timestep, clip_fea, context):
        # clip_fea / context 在官方走 WanI2VCrossAttention，这里故意不用，避免喧宾夺主
        del timestep, clip_fea, context
        return self.patch(x36.unsqueeze(0)).squeeze(0)


def make_condition(first_frame, keep):
    """推理/训练共用：y = cat(mask, z_c)，与 image2video.py:240-249 同构。"""
    zeros = torch.zeros(C, T - 1, H, W)
    cond_video = torch.cat([first_frame[:, None], zeros], 1)
    z_c = causal_vae_encode(cond_video)
    mask = pack_frame_mask(keep)
    y = torch.cat([mask, z_c], 0)  # 官方顺序 [4|16]，不是论文示意图的 [16|4]
    return y, z_c, mask


def train_one_sample(dit, video, prompt_emb, clip_fea, keep, t=None):
    """一条训练样本、一次前向。论文 Eq.(x_t): X_t = t*x_1 + (1-t)*x_0，t∈[0,1]。"""
    x_1 = causal_vae_encode(video)  # 真视频全部帧 → 干净目标
    x_0 = torch.randn_like(x_1)  # 论文里的噪声，也叫 x_0
    if t is None:
        # 论文：logit-normal。这里均匀即可看形状
        t = torch.rand(1)
    t = t.clamp(0, 1)
    X_t = t * x_1 + (1 - t) * x_0
    v_tgt = x_1 - x_0

    y, z_c, mask = make_condition(video[:, 0], keep)
    x36 = torch.cat([X_t, y], 0)  # model.py:531  cat(X_t, y) → [36, t_lat, h, w]

    v_pred = dit(x36, t, clip_fea, prompt_emb)
    loss = F.mse_loss(v_pred, v_tgt)
    return {
        "t": float(t),
        "x_1": x_1,
        "X_t": X_t,
        "z_c": z_c,
        "mask": mask,
        "y": y,
        "x36": x36,
        "v_tgt": v_tgt,
        "v_pred": v_pred,
        "loss": loss,
    }


def infer_loop(dit, first_frame, prompt_emb, clip_fea, keep, n_steps=4):
    """对比：推理才沿 t 连环。起点是纯噪声；y 全程不变。"""
    y, _, _ = make_condition(first_frame, keep)
    X = torch.randn(c, t_lat, h, w)
    traj = []
    # 从噪声端 t=0 走到数据端 t=1（论文 RF 记号；官方 scheduler 用 1000→0 整数步，方向相同）
    ts = torch.linspace(0, 1, n_steps + 1)
    for i in range(n_steps):
        t, t_next = ts[i], ts[i + 1]
        x36 = torch.cat([X, y], 0)
        v = dit(x36, t, clip_fea, prompt_emb)
        X = X + (t_next - t) * v  # Euler：X ← X + Δt * u(X,t)
        traj.append((float(t), X.detach().clone()))
    return X, traj


def show(name, x):
    print(f"  {name:8s} {tuple(x.shape)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    video = torch.randn(C, T, H, W)  # 一条 9 帧训练视频
    prompt_emb = torch.randn(T5_LEN, T5_DIM)
    clip_fea = torch.randn(1, CLIP_LEN, CLIP_DIM)
    dit = TinyI2VDiT()

    print("=" * 60)
    print("几何")
    print("=" * 60)
    print(f"  像素  [3,{T},{H},{W}]  →  latent [16,{t_lat},{h},{w}]")
    print(f"  官方 concat: X_t[16] + mask[4] + z_c[16] = 36")

    print()
    print("=" * 60)
    print("训练一步（只抽一个 t，没有 for-loop）")
    print("=" * 60)
    out = train_one_sample(dit, video, prompt_emb, clip_fea, i2v_keep(), t=torch.tensor([0.3]))
    show("x_1", out["x_1"])
    show("X_t", out["X_t"])
    show("z_c", out["z_c"])
    show("mask", out["mask"])
    show("y", out["y"])
    show("x36", out["x36"])
    show("v_pred", out["v_pred"])
    print(f"  t={out['t']:.2f}  loss={out['loss'].item():.4f}")
    print("  通道切分 x36 = [X_t | mask | z_c]")
    print(f"    X_t   = x36[0:16]   随 t 变")
    print(f"    mask  = x36[16:20]  固定  I2V 时 t_lat=0 的 4 通道全 1")
    print(f"    z_c   = x36[20:36]  固定  [首帧|0-pad] 的 VAE")
    print(f"  mask 第 0 格均值={out['mask'][:, 0].mean():.0f}  其余均值={out['mask'][:, 1:].mean():.0f}")

    # x_1 与 z_c：第 0 格都来自首帧，后面格一个有运动、一个是 0-pad
    d0 = (out["x_1"][:, 0] - out["z_c"][:, 0]).abs().mean()
    d1 = (out["x_1"][:, 1:] - out["z_c"][:, 1:]).abs().mean()
    print(f"  |x_1-z_c|  第0格={d0:.3f}  后续格={d1:.3f}  （后续应更大：z_c 是 0-pad）")

    print()
    print("=" * 60)
    print("联合训练时只改 keep（论文 §5.1），y 的拼法不变")
    print("=" * 60)
    for name, keep in [
        ("I2V 首帧", i2v_keep()),
        ("首末帧", flf2v_keep()),
        ("续写前3帧", continuation_keep(3)),
    ]:
        m = pack_frame_mask(keep)
        packs = ["".join("1" if v > 0.5 else "0" for v in m[:, i, 0, 0].tolist()) for i in range(t_lat)]
        print(f"  {name:8s} 像素keep={''.join(str(int(x)) for x in keep.tolist())}  "
              f"每格4通道={packs}")

    print()
    print("=" * 60)
    print("推理才 for t：y 冻结，只 Euler 更新 X")
    print("=" * 60)
    dit.eval()
    with torch.no_grad():
        X_hat, traj = infer_loop(dit, video[:, 0], prompt_emb, clip_fea, i2v_keep(), n_steps=4)
    for t, X in traj:
        print(f"  t={t:.2f}  X mean={X.mean():+.3f}  std={X.std():.3f}")
    show("X_hat", X_hat)
    print("  注意：训练从不跑这段循环；上表只为对照。")
