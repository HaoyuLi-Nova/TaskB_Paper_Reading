# Image classification

CIFAR-10 + ResNet-18. 32×32 输入把 stem 改成 3×3、stride 1，并去掉 maxpool。

```bash
cd paper_reading/code/mycode/image_cls
/data1/lihaoyu/miniconda3/envs/videodiff/bin/python train.py
```

自有数据换成 ImageFolder（`train/`、`val/` 下按类名分目录），此时用标准 224 stem：

```bash
python train.py --data /path/to/imagenet_style --lr 0.01 --epochs 30
```

权重写到 `runs/image_cls/best.pt`。
