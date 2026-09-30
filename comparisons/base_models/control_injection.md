# 对比：条件注入方式（Control 系）

> 在读完 ControlNet / IP-Adapter / GLIGEN / OminiControl / EasyControl 后填写。

| 维度 | ControlNet | IP-Adapter | GLIGEN | OminiControl | EasyControl |
|------|------------|------------|--------|--------------|-------------|
| 基座架构 | UNet | UNet | UNet | DiT | DiT |
| 条件类型 | 空间对齐 | 图像提示/主体 | grounding 布局 | 空间+主体统一 | 多条件灵活 |
| 参数开销 | 可训练副本 | 轻量 adapter | gated self-attn | ~0.1% | LoRA 注入 |
| 多条件组合 | 需多网络 | 可与 ControlNet 叠 | — | 统一框架 | 零样本多条件 |
| 推理效率技巧 | — | — | — | — | KV Cache / causal attn |
| 对我视频控制的启发 | | | | | |

## 结论（读完后写）

1. 
2. 
