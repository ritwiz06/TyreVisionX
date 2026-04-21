import numpy as np

from src.anomaly.thresholds import metrics_at_threshold, select_recall_priority_threshold


def test_threshold_selection_smoke():
    targets = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    result = select_recall_priority_threshold(targets, scores, max_normal_fpr=0.34)
    metrics = metrics_at_threshold(targets, scores, result.threshold)
    assert result.threshold >= 0.0
    assert metrics["recall"] == 1.0
    assert metrics["normal_fpr"] <= 0.34

