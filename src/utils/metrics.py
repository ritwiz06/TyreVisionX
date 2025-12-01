"""Metric helpers wrapping torchmetrics for binary classification (defect-positive)."""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from torchmetrics.functional import (
    accuracy,
    auroc,
    average_precision,
    f1_score,
    precision,
    recall,
)


def classification_metrics(logits: Tensor, targets: Tensor) -> Dict[str, float]:
    """Compute core metrics; class 1 (defect) is treated as positive.

    Args:
        logits: Raw model outputs of shape [N, 2].
        targets: Ground-truth labels of shape [N], values in {0, 1}.
    """
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    pos_scores = probs[:, 1]

    metrics = {
        "accuracy": accuracy(preds, targets, task="multiclass", num_classes=2).item(),
        "precision": precision(preds, targets, average="macro", task="multiclass", num_classes=2).item(),
        "recall": recall(preds, targets, average="macro", task="multiclass", num_classes=2).item(),
        "f1_macro": f1_score(preds, targets, average="macro", task="multiclass", num_classes=2).item(),
        "f1_defect": f1_score(pos_scores, targets, task="binary", pos_label=1).item(),
        "auroc": auroc(pos_scores, targets, task="binary", pos_label=1).item(),
        "auprc": average_precision(pos_scores, targets, task="binary", pos_label=1).item(),
    }
    return metrics


def reduce_metrics(metric_list: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in metric_list.items()}
