# MultiTalk 官方推理代码

来源：`git@github.com:MeiGen-AI/MultiTalk.git`（SSH 浅克隆 `--depth 1`）

读 L-RoPE / 声–人绑定时优先这几处：

| 内容 | 路径 |
|------|------|
| 首帧 bbox → `ref_target_masks` | `wan/multitalk.py`（`construct human mask`） |
| 通道拼接 `latent ∥ y`、进 DiT | `wan/modules/multitalk_model.py`（`forward`） |
| self-attn 旁路算 \(S\) | `wan/modules/multitalk_model.py`（`WanSelfAttention`）+ `wan/utils/multitalk_utils.py`（`get_attn_map_with_target`） |
| 由 \(S\) 赋 label、1D RoPE、音频 cross-attn | `wan/modules/attention.py`（`SingleStreamMutiAttention`） |

入口：`generate_multitalk.py`。权重不在本仓库，见官方 `README.md`。
