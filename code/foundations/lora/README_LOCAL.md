# Microsoft LoRA 官方实现

来源：`git@github.com:microsoft/LoRA.git`（SSH **完整克隆**，非浅克隆；`main` @ `c4593f0`）

读论文公式 $h=W_0x+BAx$、合并、以及只打 $W_q,W_v$时优先这几处：

| 内容 | 路径 |
|------|------|
| 核心算子 `Linear` / `MergedLinear` / `Embedding` / `Conv*` | `loralib/layers.py` |
| 只训 `lora_*`、导出小 checkpoint | `loralib/utils.py`（`mark_only_lora_as_trainable`、`lora_state_dict`） |
| GPT-2：`c_attn` 用 `MergedLinear` 只启用 Q/V | `examples/NLG/src/model.py` |
| GPT-2 训练入口 | `examples/NLG/src/gpt2_ft.py` |
| RoBERTa：`query` / `value` → `lora.Linear` | `examples/NLU/src/transformers/models/roberta/modeling_roberta.py` |
| DeBERTa v2：同上，`merge_weights=False` | `examples/NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py` |

`examples/NLU/` 是整份 Hugging Face Transformers 快照，体积大；精读 **不要** 从那里逛起。

对照论文时注意两处实现差：

1. **`Linear` 初始化**：论文写 $A$ 高斯、$B=0$；代码用 Kaiming $A$、$B=0$，并注明 “different than what is described in the paper”。
2. **`Embedding` 初始化反过来了**：`lora_A` 全零、`lora_B` 正态——起步 $\Delta W$ 仍是 0，但梯度先走 $B$ 不是 $A$。

权重不在本仓库，见官方 `README.md` 的 GitHub Releases。
