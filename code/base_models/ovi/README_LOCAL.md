# Ovi 官方推理代码

来源：https://github.com/character-ai/Ovi（sparse clone，已去掉 assets / example pngs / .git）

入口：`inference.py`、`ovi/ovi_fusion_engine.py`
视频 VAE：`ovi/modules/vae2_2.py`（Wan2.2）
音频 VAE：`ovi/modules/mmaudio/`（MMAudio 1D VAE + BigVGAN）
DiT patch→token：`ovi/modules/model.py`（`patch_embedding`）
