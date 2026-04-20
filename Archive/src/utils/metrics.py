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
        logits: Model outputs of shape [N, 2] (multiclass) or [N]/[N, 1] (binary score/probability).
        targets: Ground-truth labels of shape [N], values in {0, 1}.
    """
    targets = targets.int()
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        pos_scores = logits.reshape(-1).float()
        # If model emits logits, convert to probabilities for threshold/AUC metrics.
        if torch.any((pos_scores < 0.0) | (pos_scores > 1.0)):
            pos_scores = torch.sigmoid(pos_scores)
        preds = (pos_scores >= 0.5).int()
    else:
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).int()
        pos_scores = probs[:, 1]

    pos_preds = (pos_scores >= 0.5).int()

    metrics = {
        "accuracy": accuracy(preds, targets, task="multiclass", num_classes=2).item(),
        "precision": precision(preds, targets, average="macro", task="multiclass", num_classes=2).item(),
        "recall": recall(preds, targets, average="macro", task="multiclass", num_classes=2).item(),
        "f1_macro": f1_score(preds, targets, average="macro", task="multiclass", num_classes=2).item(),
        "precision_defect": precision(pos_preds, targets, task="binary").item(),
        "recall_defect": recall(pos_preds, targets, task="binary").item(),
        "f1_defect": f1_score(pos_preds, targets, task="binary").item(),
        "auroc": auroc(pos_scores, targets, task="binary").item(),
        "auprc": average_precision(pos_scores, targets, task="binary").item(),
    }
    return metrics


def reduce_metrics(metric_list: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in metric_list.items()}
