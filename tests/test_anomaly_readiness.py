from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.anomaly.check_anomaly_readiness import inspect_readiness
from src.anomaly.features import ResNetEmbeddingExtractor


def test_readiness_checker_reports_config() -> None:
    payload = inspect_readiness("configs/anomaly/anomaly_baseline.yaml")
    assert payload["configured_backbone"] == "resnet18"
    assert "candidate_weight_sources" in payload
    assert "can_run_real_anomaly_baseline" in payload


def test_missing_local_weights_path_failure_is_clear(tmp_path: Path) -> None:
    missing = tmp_path / "missing_resnet18.pth"
    with pytest.raises(FileNotFoundError, match="weights_path does not exist"):
        ResNetEmbeddingExtractor(backbone="resnet18", pretrained=True, weights_path=missing)


def test_incompatible_local_weights_failure_is_clear(tmp_path: Path) -> None:
    bad = tmp_path / "bad_weights.pth"
    torch.save({"not_a_resnet_key": torch.ones(1)}, bad)
    with pytest.raises(RuntimeError, match="did not load any ResNet convolution"):
        ResNetEmbeddingExtractor(backbone="resnet18", pretrained=True, weights_path=bad)
