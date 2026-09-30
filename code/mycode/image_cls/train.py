#!/usr/bin/env python3
"""Train an image classifier. CIFAR-10 by default, or an ImageFolder."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=None, help="ImageFolder root with train/ and val/")
    p.add_argument("--cifar-root", type=Path, default=Path("data/cifar10"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/image_cls"))
    return p.parse_args()


def transforms_for(image_size: int, train: bool) -> transforms.Compose:
    if image_size == 32:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        aug = (
            [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
            if train
            else [transforms.Resize(int(image_size * 256 / 224)), transforms.CenterCrop(image_size)]
        )
    return transforms.Compose([*aug, transforms.ToTensor(), transforms.Normalize(mean, std)])


def build_loaders(args: argparse.Namespace):
    if args.data is None:
        size, num_classes = 32, 10
        train_set = datasets.CIFAR10(args.cifar_root, train=True, download=True, transform=transforms_for(size, True))
        val_set = datasets.CIFAR10(args.cifar_root, train=False, download=True, transform=transforms_for(size, False))
        classes = train_set.classes
    else:
        size = 224
        train_set = datasets.ImageFolder(args.data / "train", transforms_for(size, True))
        val_set = datasets.ImageFolder(args.data / "val", transforms_for(size, False))
        classes, num_classes = train_set.classes, len(train_set.classes)
    kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    return (
        DataLoader(train_set, shuffle=True, **kw),
        DataLoader(val_set, shuffle=False, **kw),
        size,
        num_classes,
        classes,
    )


def build_model(num_classes: int, cifar: bool) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def run_epoch(model, loader, optimizer, scaler, device, train: bool) -> tuple[float, float]:
    model.train(train)
    loss_sum, correct, n = 0.0, 0, 0
    amp = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, size, num_classes, classes = build_loaders(args)
    model = build_model(num_classes, cifar=size == 32).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, scaler, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, scaler, device, train=False)
        scheduler.step()
        print(
            f"epoch {epoch:03d}  train {train_loss:.4f}/{train_acc:.3f}  "
            f"val {val_loss:.4f}/{val_acc:.3f}  lr {scheduler.get_last_lr()[0]:.4g}"
        )
        if val_acc > best:
            best = val_acc
            torch.save(
                {"model": model.state_dict(), "acc": val_acc, "classes": classes},
                args.out / "best.pt",
            )
    print(f"best val acc {best:.3f}  saved {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
