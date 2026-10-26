"""2D U-Net segmentation baseline.

A well-tuned U-Net remains the reference architecture for CAMUS (the original TMI paper's
best method was U-Net based), so it is the honest baseline to build on rather than a
strawman. Group normalisation is used instead of batch norm so results stay stable at the
small batch sizes echo volumes force.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(cout, groups=8):
    """Largest divisor of `cout` not exceeding `groups`.

    GroupNorm requires num_channels to be divisible by num_groups, so min(groups, cout)
    silently breaks for any width that is not a multiple of 8 (e.g. width=12 -> 12 channels
    with 8 groups). Picking a divisor keeps arbitrary widths usable, which matters for the
    capacity sweep.
    """
    for g in range(min(groups, cout), 0, -1):
        if cout % g == 0:
            return g
    return 1


def _block(cin, cout, groups=8):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(_num_groups(cout, groups), cout), nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(_num_groups(cout, groups), cout), nn.LeakyReLU(0.01, inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, width=32, depth=5):
        super().__init__()
        chs = [width * 2 ** i for i in range(depth)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.enc.append(_block(c, ch))
            c = ch
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.dec.append(_block(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = dec(torch.cat([x, skip], 1))
        return self.head(x)


def dice_ce_loss(logits, target, n_classes=4, eps=1e-6, ce_weight=0.5):
    """Soft Dice + cross-entropy. Dice is computed over foreground classes only, so a
    correct-but-trivial all-background prediction cannot score well."""
    ce = F.cross_entropy(logits, target)
    prob = logits.softmax(1)
    oh = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * oh).sum(dims)[1:]
    denom = prob.sum(dims)[1:] + oh.sum(dims)[1:]
    dice = (2 * inter + eps) / (denom + eps)
    return ce_weight * ce + (1 - ce_weight) * (1 - dice.mean())


@torch.no_grad()
def dice_per_class(pred, target, n_classes=4, eps=1e-6):
    """Per-sample, per-foreground-class Dice. Returns (N, n_classes-1)."""
    out = []
    for c in range(1, n_classes):
        p = (pred == c).flatten(1).float()
        t = (target == c).flatten(1).float()
        inter = (p * t).sum(1)
        denom = p.sum(1) + t.sum(1)
        out.append(torch.where(denom > 0, (2 * inter + eps) / (denom + eps),
                               torch.ones_like(denom)))
    return torch.stack(out, 1)
