"""Echo image-quality assessment: trained on CAMUS expert grades, applied to EchoNet.

This bridges the two datasets. CAMUS has expert cardiologist quality grades but only 47
Poor patients with EF, far below the ~236/arm needed to power a method claim. EchoNet has
10,030 patients but ships no quality labels. Training the grader here and applying it there
gives labels *and* power.

Design constraints that keep this honest:

* The grader must never see EF or segmentation, only the image, so a predicted grade cannot
  smuggle in outcome information and manufacture the very correlation we want to measure.
* Evaluated on held-out CAMUS patients, with the published open-source estimator as the
  reference bar (rho = 0.69 against three cardiologists, versus rho = 0.63 inter-observer;
  arXiv 2408.00591). Human-level agreement is the ceiling worth aiming at, not 1.0.
* Grades are ordinal (Good < Medium < Poor), so the loss is ordinal rather than plain
  cross-entropy: confusing Good with Poor is a worse error than confusing Good with Medium.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRADES = ["Good", "Medium", "Poor"]
GRADE_TO_IDX = {g: i for i, g in enumerate(GRADES)}


class QualityNet(nn.Module):
    """Small conv encoder with ordinal (cumulative-logit) head.

    K-1 binary outputs answering "is quality worse than grade k", which respects the
    ordering of the grades. Kept deliberately small: 500 CAMUS images is not enough data to
    train a large backbone without overfitting the quality label.
    """

    def __init__(self, width=24, n_grades=3, dropout=0.3):
        super().__init__()
        chs = [width, width * 2, width * 4, width * 8]
        layers, c = [], 1
        for ch in chs:
            layers += [
                nn.Conv2d(c, ch, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, ch), ch), nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, ch), ch), nn.LeakyReLU(0.01, inplace=True),
                nn.MaxPool2d(2),
            ]
            c = ch
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c, n_grades - 1)          # cumulative logits

    def forward(self, x):
        h = self.body(x)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.head(self.drop(h))                  # (N, K-1)


def ordinal_targets(idx, n_grades=3):
    """Grade index -> cumulative binary targets: grade 2 becomes [1, 1]."""
    t = torch.zeros(len(idx), n_grades - 1, device=idx.device)
    for k in range(n_grades - 1):
        t[:, k] = (idx > k).float()
    return t


def ordinal_loss(logits, idx, n_grades=3):
    return F.binary_cross_entropy_with_logits(logits, ordinal_targets(idx, n_grades))


def logits_to_grade(logits):
    """Expected grade in [0, K-1]: a continuous severity score, not just argmax.

    The continuous form matters downstream, because stratifying EchoNet by a threshold on a
    smooth score is more defensible than by a hard 3-way argmax whose boundaries are
    arbitrary.
    """
    p = torch.sigmoid(logits)
    return p.sum(1)


def grade_from_score(score, thresholds=(0.5, 1.5)):
    out = torch.zeros_like(score, dtype=torch.long)
    for t in thresholds:
        out += (score > t).long()
    return out
