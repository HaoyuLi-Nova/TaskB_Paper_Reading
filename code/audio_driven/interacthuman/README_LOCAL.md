# InterActHuman 官方开源代码

来源：`git@github.com:zhenzhiwang/InterActHuman.git`（SSH **完整克隆**，非浅克隆；`main` @ `14b2e2b`）

这是作者基于 Wan2.1 + [OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) 的 **demonstration re-implementation**，不是 ByteDance 内部原实现：无预训练权重、不保证与论文逐行等价。对照论文时以 TeX 为准，代码只当算法示意。

精读 mask predictor / 局部音频注入时优先这几处：

| 内容 | 路径 |
|------|------|
| `MaskPredictor`（Q=视频 token，K=参考图 latent） | `OmniAvatar/models/wan_video_dit.py` |
| 每层预测 mask、音频 residual 相加 | 同上 `WanModel.forward` |
| wav2vec 打包 `AudioPack` | `OmniAvatar/models/audio_pack.py` |
| 推理入口（OmniAvatar 单人流程） | `scripts/inference.py` |
| caption → 实体 JSON（Gemini） | `preprocess/caption/` |
| Grounding-SAM2 真值 mask | `preprocess/grounding-sam/` |

对照论文时注意实现差：

1. **音频未按 mask 门控。** `forward` 里是 `x = audio_cond_tmp + x` 全局相加，再另路预测 `pred_masks` 并返回；论文 Algorithm 1 是第 \(k\) 步 mask 指导第 \(k{+}1\) 步 **masked audio attention**。
2. **`MaskPredictor.k` 用 `freqs` 而不是 `freqs_ref`。** `k = rope_apply(k, freqs, ...)` 与传入的 `freqs_ref` 未对齐。
3. **无迭代调度。** 仓库没有前 10 步不用 mask或 step-to-step mask 传递；那套逻辑只在 TeX 的 Algorithm 1。

权重不在本仓库，见官方 `README.md`（下载的是 OmniAvatar / Wan2.1，不是 InterActHuman checkpoint）。
