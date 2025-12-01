"""Class balancing utilities."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


def compute_class_weights(labels: Iterable[int], num_classes: int = 2) -> Tensor:
    counts = Counter(labels)
    weights: List[float] = []
    total = sum(counts.values())
    for cls in range(num_classes):
        count = counts.get(cls, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def focal_loss(logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    return loss.mean()
