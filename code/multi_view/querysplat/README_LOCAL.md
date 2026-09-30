# QuerySplat 官方推理代码

来源：https://github.com/inspatio/QuerySplat（HTTPS 浅克隆 `--depth 1`，`GIT_LFS_SKIP_SMUDGE=1`）

`main` @ `9703d6f`（2026-09-18：补 paper checkpoint、DL3DV eval 与评测协议）。本目录不含权重。

| 内容 | 路径 |
|------|------|
| 模型图：冻结 VGGT-Ω + 双分支 query decoder | `scripts/models/querysplat.py`（`QuerySplat`） |
| VGGT-Ω encoder / 相机与深度头 | `scripts/models/vggt_encoder.py`、`third_party/vggt_omega/` |
| 几何 token → 位置/尺度/旋转；外观 token → RGB/opacity | `scripts/models/querysplat.py`（`forward_decoder`） |
| 自定义图推理入口 | `scripts/infer.py`（`python -m scripts.infer`） |
| DL3DV 评测（2/4/12 视角 × 小/中/大窗） | `evaluations/evaluate_jsons.py`、`evaluations/jsons_dl3dv/` |
| 推理配置（不含权重） | `checkpoints/querysplat_vggto_1B_512_{8192,paper}.yaml` |

权重从 Hugging Face 拉到本目录 `checkpoints/`（见官方 `README.md`）：

| 组件 | Hugging Face | 本地文件 |
|------|----------------|----------|
| QuerySplat（视觉更好，SH degree 1） | [inspatio/querysplat](https://huggingface.co/inspatio/querysplat) | `querysplat_vggto_1B_512_8192.safetensors` |
| QuerySplat（论文指标，SH degree 0） | 同上 | `querysplat_vggto_1B_512_paper.safetensors` |
| 冻结 VGGT-Ω 1B/512 | [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) | `vggt_omega_1b_512.pt` |

论文原文：[`arxiv/multi_view/querysplat/`](../../../arxiv/multi_view/querysplat/)。项目页：https://inspatio.github.io/querysplat/
