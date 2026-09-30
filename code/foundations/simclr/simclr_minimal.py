"""Minimal SimCLR training loop — focus on gradient flow.

Key point: features (h, z) are NOT precomputed / frozen.
They are outputs of the CURRENT network in this forward pass,
so loss.backward() updates both projector g and encoder f.

  x --aug--> x1, x2
       |           |
       f           f     <-- same encoder, gradients flow
       |           |
       h1          h2
       |           |
       g           g     <-- projector, gradients flow
       |           |
       z1          z2
       \           /
        NT-Xent loss
              |
         loss.backward()
              |
     grads -> g.params AND f.params

When would gradients NOT flow?
  - z = z.detach() / with torch.no_grad()
  - MoCo's momentum encoder (key side uses stop-grad)
  - memory-bank features stored from previous steps (no graph)

SimCLR does none of those for the two views in the batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model: encoder f + projector g
# ---------------------------------------------------------------------------
class TinyEncoder(nn.Module):
    """Stand-in for ResNet; any CNN / ViT works the same for gradient flow."""

    def __init__(self, in_ch: int = 3, h_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, h_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # h


class Projector(nn.Module):
    """g(h) = W2 * ReLU(W1 * h)  -> z  (used only in contrastive loss)."""

    def __init__(self, h_dim: int = 128, z_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, z_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SimCLR(nn.Module):
    def __init__(self, h_dim: int = 128, z_dim: int = 64):
        super().__init__()
        self.encoder = TinyEncoder(h_dim=h_dim)  # f
        self.projector = Projector(h_dim=h_dim, z_dim=z_dim)  # g

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        # IMPORTANT: no .detach() here — keep the autograd graph
        return h, z


# ---------------------------------------------------------------------------
# NT-Xent loss (Eq. in SimCLR paper)
# ---------------------------------------------------------------------------
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """z1, z2: (N, D) — two augmented views of the same N images."""
    n = z1.shape[0]
    # L2-normalize -> cosine similarity
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate: [z1; z2] -> (2N, D)
    z = torch.cat([z1, z2], dim=0)  # indices 0..N-1 = view1, N..2N-1 = view2
    # Pairwise similarity matrix (2N, 2N)
    sim = (z @ z.T) / temperature

    # Positive pairs: (i, i+N) and (i+N, i)
    # For row i (0..N-1), positive column is i+N
    # For row i (N..2N-1), positive column is i-N
    labels = torch.arange(n, device=z.device)
    labels = torch.cat([labels + n, labels], dim=0)  # (2N,)

    # Mask self-similarity (diagonal) so it is never a "negative" or "positive" by mistake
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    # Cross-entropy: for each query, classify its positive among 2N-1 others
    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# Fake "two views" — in real SimCLR: two random augmentations of the same image
# ---------------------------------------------------------------------------
def two_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cheap stand-in for crop/color/blur augmentations."""
    noise1 = 0.05 * torch.randn_like(x)
    noise2 = 0.05 * torch.randn_like(x)
    # Flip half of view2 horizontally to look different
    x2 = torch.flip(x, dims=[-1])
    return (x + noise1).clamp(0, 1), (x2 + noise2).clamp(0, 1)


# ---------------------------------------------------------------------------
# Train one step + prove gradients reach the encoder
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    model = SimCLR().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fake batch of N images (CIFAR-like)
    n = 8
    x = torch.rand(n, 3, 32, 32, device=device)

    model.train()
    opt.zero_grad(set_to_none=True)

    x1, x2 = two_views(x)

    # ----- forward: features are computed LIVE -----
    h1, z1 = model(x1)
    h2, z2 = model(x2)

    print("--- graph check (before backward) ---")
    print(f"z1.requires_grad = {z1.requires_grad}")  # True
    print(f"z1.grad_fn       = {z1.grad_fn}")  # not None → still on the graph
    print(f"h1.grad_fn       = {h1.grad_fn}")

    loss = nt_xent(z1, z2, temperature=0.5)
    print(f"loss = {loss.item():.4f}")

    # ----- backward: grads flow z -> g -> h -> f -----
    loss.backward()

    # Encoder first-layer weight should have non-zero grad
    enc_w = model.encoder.net[0].weight
    proj_w = model.projector.net[0].weight
    print("--- gradient check (after backward) ---")
    print(f"encoder.conv1.grad  is None? {enc_w.grad is None}")
    print(f"encoder.conv1.grad  ||.||   = {enc_w.grad.norm().item():.6f}")
    print(f"projector.fc1.grad  ||.||   = {proj_w.grad.norm().item():.6f}")

    assert enc_w.grad is not None and enc_w.grad.abs().sum() > 0, (
        "BUG: encoder got no gradient — you probably detached z/h"
    )
    print("OK: contrastive loss updated BOTH projector and encoder.")

    opt.step()

    # A few more steps just to see loss move
    for step in range(1, 6):
        opt.zero_grad(set_to_none=True)
        x = torch.rand(n, 3, 32, 32, device=device)
        x1, x2 = two_views(x)
        _, z1 = model(x1)
        _, z2 = model(x2)
        loss = nt_xent(z1, z2)
        loss.backward()
        opt.step()
        print(f"step {step}: loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
