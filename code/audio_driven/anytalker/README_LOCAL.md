# AnyTalker 官方推理代码

来源：`git@github.com:HKUST-C4G/AnyTalker.git`（SSH **完整克隆**，非浅克隆；`main` @ `e5490b6`，2026-04-15，含 14B infer）

读 AFCA / 静态脸框绑定 / InteractEye 时优先这几处。形状与空间门的逐步拆开见 [`notes/audio_driven/anytalker/afca.md`](../../../notes/audio_driven/anytalker/afca.md)。

| 内容 | 路径 |
|------|------|
| Audio-Face Cross Attention（论文 AFCA） | `wan/modules/model.py`（`WanAF2VCrossAttention`） |
| 多 ID 循环：每对 audio–face 做一次 cross-attn，输出乘 face mask 再求和 | 同上 `forward` |
| 首帧 InsightFace 框 → 全视频静态 `face_mask` | `utils/get_face_bbox.py`、`wan/utils/infer_utils.py`（`gen_inference_masks`） |
| 推理入口（任意人数） | `generate_a2v_batch_multiID.py`、`wan/audio2video_multiID.py` |
| InteractEye：听者眼动交互性 | `benchmark/calculate_interactivity.py` |

Gradio：`app.py`。权重不在本仓库，见官方 `README.md`（1.3B / 14B 从 Hugging Face 拉）。
