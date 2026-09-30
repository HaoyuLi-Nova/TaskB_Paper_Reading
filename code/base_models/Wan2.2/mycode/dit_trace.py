"""把官方 WanModel 的 forward 展开，便于逐层看数据。

对照：
  wan/modules/model.py  WanModel.forward / WanAttentionBlock.forward

权重仍是 TI2V-5B 的真实参数。只在第一次 DiT forward 打印；
之后 8 步去噪静默跑。要看某一层，在 dit_block() 下断点。
"""
from types import MethodType

import torch
import torch.nn.functional as F

from wan.modules.model import sinusoidal_embedding_1d

_log_this_call = True


def show(name, x):
    if not _log_this_call:
        return
    if torch.is_tensor(x):
        v = x.detach().float()
        print(f"  {name:28s} {tuple(x.shape)}  mean={v.mean():+.4f}  std={v.std():.4f}")
    else:
        print(f"  {name:28s} {x}")


def patch_sdpa():
    """本机没有 flash_attn 时，用 PyTorch SDPA 替换官方接口。"""
    try:
        import flash_attn  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    def sdpa(q, k, v, q_lens=None, k_lens=None, **kwargs):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return out.transpose(1, 2).contiguous()

    import wan.modules.attention as attn
    import wan.modules.model as model
    attn.flash_attention = sdpa
    model.flash_attention = sdpa


def dit_block(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
    """一层 Transformer：Self-Attn → Cross-Attn(文本) → FFN。

    e 是 timestep 的 6 组 adaLN 参数 (shift/scale/gate) × 2。
    """
    i = self._layer_id
    e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

    # --- self-attn（视频 token 互看，带 3D RoPE）---
    xa = self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
    sa = self.self_attn(xa, seq_lens, grid_sizes, freqs)
    x = x + sa * e[2].squeeze(2)
    show(f"block[{i:02d}] self_attn", x)

    # --- cross-attn（Q=视频, K/V=文本）---
    x = x + self.cross_attn(self.norm3(x), context, context_lens)
    show(f"block[{i:02d}] cross_attn", x)

    # --- FFN ---
    xf = self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
    x = x + self.ffn(xf) * e[5].squeeze(2)
    show(f"block[{i:02d}] ffn", x)
    return x


def dit_forward(self, x, t, context, seq_len, y=None):
    """完整 DiT：patch → 时间/文本嵌入 → N 层 block → unpatchify。

    x: list[[C, T, H, W]]  潜空间视频（TI2V 的 C=48）
    t: [B] 或 [B, seq]     时间步
    context: list[[L, 4096]]  T5 文本特征
    """
    global _log_this_call
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if _log_this_call:
        print("\n======== DiT forward (first step) ========")
        show("latent in", x[0])
        show("t", t)
        show("t5 context", context[0])

    # I2V-A14B 才会走：噪声 16ch + 条件 20ch → 36ch。TI2V-5B 的 y 是 None。
    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        show("after cat(y)", x[0])

    # Conv3d patch (1,2,2)：空间 ÷2，通道 → dim(3072)
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
    )
    show("after patch Conv3d", x[0])
    show("grid_sizes (F,H,W)", grid_sizes[0].tolist())

    # [B,C,F,H,W] → [B, F*H*W, C]
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    x = torch.cat(
        [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
    )
    show("tokens", x)

    # timestep → 每个 token 的 adaLN 条件 e0: [B, seq, 6, dim]
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        bt = t.size(0)
        t_flat = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_flat)
            .unflatten(0, (bt, seq_len))
            .float()
        )
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
    show("time embed e", e)
    show("adaLN e0", e0)

    # T5 [L,4096] → [B, 512, dim]
    context = self.text_embedding(
        torch.stack(
            [torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]
        )
    )
    show("text embed", context)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=None,
    )
    for block in self.blocks:
        x = block(x, **kwargs)

    x = self.head(x, e)
    show("after head", x)
    x = self.unpatchify(x, grid_sizes)
    show("unpatchify out", x[0])
    if _log_this_call:
        print("======== end first DiT forward ========\n")
    _log_this_call = False
    return [u.float() for u in x]


def attach(model):
    """用上面的展开版替换已加载 WanModel 的 forward。"""
    patch_sdpa()
    model.forward = MethodType(dit_forward, model)
    for i, block in enumerate(model.blocks):
        block._layer_id = i
        block.forward = MethodType(dit_block, block)
    print(f"DiT: {model.num_layers} layers, dim={model.dim}, heads={model.num_heads}")
