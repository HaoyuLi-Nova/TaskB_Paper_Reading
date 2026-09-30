from __future__ import annotations

# 标准库
import argparse
from pathlib import Path

# 第三方库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def parse_args()->argparse.Namespace:
    # def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path, default=None, help ="ImageFolder root with train/ and val/")
    p.add_argument("--cifar-root",type=Path, default=Path("data/cifar10"))
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--lr",type=int, default=0.1)
    p.add_argement("--workers",type=int, default=4)
    p.add_argument("--out",type=Path,default=Path("runs/image_cls"))
    return p.parse_args()
    p = argparse.ArgumentParse

def transformers_for(image_size: int,train: bool)->transforms.Compose:
    return transforms.Compose([*aug,transforms.ToTensor(),transforms.CenterCrop(image)])
