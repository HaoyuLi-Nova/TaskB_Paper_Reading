# 对比：图像 / 视频生成基座

> 读完 LDM、SD3、SANA、Qwen-Image、Z-Image、BAGEL、Wan 后填写。

| 维度 | LDM | SD3 | SANA | Qwen-Image | Z-Image | BAGEL | Wan |
|------|-----|-----|------|------------|---------|-------|-----|
| 骨干 | UNet | MM-DiT | Linear DiT | MMDiT | Single-stream DiT | MoT 统一 | Video DiT |
| 训练目标 | DDPM/latent | Rectified Flow | — | — | — | 理解+生成 | 视频扩散 |
| 模态 | 图像 | 图像 | 图像 | 图像+编辑 | 图像 | 统一多模态 | 视频(+下游) |
| 效率主张 | latent 降算力 | 缩放 Flow | 线性注意力 | 文本渲染 | 单流参数效率 | 涌现能力 | 1.3B/14B 开源 |
| 与可控生成接口 | ControlNet 生态 | DiT 控制待建 | — | — | — | — | 运动/相机插件挂载点 |

## 结论

1. 
2. 
