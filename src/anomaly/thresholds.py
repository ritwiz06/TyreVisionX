"""Threshold selection for anomaly scores."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdResult:
    threshold: float
    policy: str
    fallback_used: bool
    validation_metrics: dict[str, float]


def metrics_at_threshold(targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    targets = targets.astype(int)
    preds = (scores >= threshold).astype(int)

    tp = int(((targets == 1) & (preds == 1)).sum())
    tn = int(((targets == 0) & (preds == 0)).sum())
    fp = int(((targets == 0) & (preds == 1)).sum())
    fn = int(((targets == 1) & (preds == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    normal_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / len(targets) if len(targets) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "normal_fpr": float(normal_fpr),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def threshold_sweep(targets: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    unique_scores = np.unique(scores.astype(float))
    if len(unique_scores) == 0:
        raise ValueError("Cannot select threshold from empty scores.")
    thresholds = np.concatenate(
        [
            [unique_scores.min() - 1e-9],
            unique_scores,
            [unique_scores.max() + 1e-9],
        ]
    )
    return [metrics_at_threshold(targets, scores, float(t)) for t in thresholds]


def select_recall_priority_threshold(
    targets: np.ndarray,
    scores: np.ndarray,
    max_normal_fpr: float = 0.10,
) -> ThresholdResult:
    """Select validation threshold only.

    Primary rule: maximize anomaly recall subject to normal false-positive rate
    <= ``max_normal_fpr``. Ties prefer higher precision, then higher threshold.

    Fallback: maximize F1 if no threshold satisfies the FPR constraint.
    """

    rows = threshold_sweep(targets, scores)
    valid = [row for row in rows if row["normal_fpr"] <= max_normal_fpr]
    fallback_used = False
    policy = f"maximize_recall_subject_to_normal_fpr<={max_normal_fpr}"

    if valid:
        best = sorted(valid, key=lambda r: (r["recall"], r["precision"], r["threshold"]), reverse=True)[0]
    else:
        fallback_used = True
        policy = "fallback_maximize_f1"
        best = sorted(rows, key=lambda r: (r["f1"], r["recall"], r["precision"]), reverse=True)[0]

    return ThresholdResult(
        threshold=float(best["threshold"]),
        policy=policy,
        fallback_used=fallback_used,
        validation_metrics={k: float(v) for k, v in best.items()},
    )

