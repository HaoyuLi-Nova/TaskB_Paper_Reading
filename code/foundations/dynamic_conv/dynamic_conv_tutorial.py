"""动态卷积：用真实 nn.Conv2d 的 weight 按注意力聚合。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, K=4, temperature=30.0):
        super().__init__()
        self.K = K
        self.temperature = temperature
        self.padding = kernel_size // 2

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding)
            for _ in range(K)
        ])
        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.fc2 = nn.Linear(in_channels // 4, K)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # π
        pi = F.softmax(self.fc2(F.relu(self.fc1(x.mean([2, 3])))) / self.temperature, dim=1)

        # W̃ = Σ π_k · conv_k.weight
        w = sum(pi[:, k].view(-1, 1, 1, 1, 1) * self.convs[k].weight for k in range(self.K))
        b = sum(pi[:, k].view(-1, 1) * self.convs[k].bias for k in range(self.K))

        # 每个样本用自己的核卷积
        y = torch.cat([
            F.conv2d(x[i:i+1], w[i], b[i], padding=self.padding)
            for i in range(x.shape[0])
        ])
        return self.act(self.bn(y))


if __name__ == "__main__":
    x = torch.randn(2, 16, 32, 32)
    layer = DynamicConv2d(16, 24, K=4)

    pi = F.softmax(layer.fc2(F.relu(layer.fc1(x.mean([2, 3])))) / 30, dim=1)
    w0 = sum(pi[0, k] * layer.convs[k].weight for k in range(4))

    print("π[0]:", pi[0].detach())
    print("W̃[0] shape:", tuple(w0.shape))
    print("out:", tuple(layer(x).shape))
