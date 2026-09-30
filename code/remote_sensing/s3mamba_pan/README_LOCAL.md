# S³Mamba-Pan 官方训练代码

来源：https://github.com/FreeZS-a/S3Mamba（HTTPS 浅克隆 `--depth 1`）

`main` @ `ad5f41b`。本目录不含权重和 WV3 / QB / GF2 的 h5 数据。

| 内容 | 路径 |
|------|------|
| 频率解耦双流网络（DWT / DSMM / GSA / ADR） | `models/FD-SSFN.py` |
| 训练与全分辨率 / 降分辨率评测 | `train.py` |
| WV3 / QB / GF2 配置 | `configs/config_{wv3,qb,gf2}.yaml` |

配置里的 `train_data_path` 仍指向作者机器上的绝对路径，跑之前改成本地 h5。依赖 `mamba_ssm` 的 `selective_scan_fn`；未安装时前向会在调用扫描算子处失败。

论文原文：[`arxiv/remote_sensing/s3mamba_pan/`](../../../arxiv/remote_sensing/s3mamba_pan/)。译本：[`translations/remote_sensing/s3mamba_pan/paper_zh.md`](../../../translations/remote_sensing/s3mamba_pan/paper_zh.md)。
