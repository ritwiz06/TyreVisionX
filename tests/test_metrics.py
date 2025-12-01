import torch

from src.utils.metrics import classification_metrics


def test_classification_metrics_simple():
    logits = torch.tensor([[0.1, 2.0], [3.0, 0.2]])
    targets = torch.tensor([1, 0])
    metrics = classification_metrics(logits, targets)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_defect"] == 1.0
    assert metrics["f1_macro"] == 1.0
