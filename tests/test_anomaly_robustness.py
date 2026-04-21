from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly.corruptions import CorruptionSpec, apply_corruption, default_corruptions
from src.anomaly.io import load_yaml
from src.anomaly.robustness import summarize_metrics, write_corruption_report


def test_corruption_generation_changes_shape_not_size() -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    for spec in default_corruptions():
        corrupted = apply_corruption(image, spec, seed=1)
        assert corrupted.shape == image.shape
        assert corrupted.dtype == np.uint8


def test_corruption_config_parses() -> None:
    config = load_yaml("configs/anomaly/anomaly_corruption_benchmark.yaml")
    assert "variants" in config
    assert "corruptions" in config
    assert any(item["name"] == "gaussian_noise_low" for item in config["corruptions"])
    assert any(item["name"] == "resnet50_knn" for item in config["variants"])


def test_noise_robust_variant_config_parses() -> None:
    config = load_yaml("configs/anomaly/anomaly_resnet50_knn_noise_robust.yaml")
    assert config["feature_extractor"]["backbone"] == "resnet50"
    assert config["scoring"]["method"] == "knn"
    assert config["train_augmentations"]


def test_corruption_summary_generation(tmp_path: Path) -> None:
    metrics = {
        "threshold_metrics": {
            "recall": 0.75,
            "precision": 0.8,
            "normal_fpr": 0.1,
            "fn": 5,
            "fp": 3,
        },
        "auroc": 0.9,
        "auprc": 0.85,
    }
    row = summarize_metrics(
        variant_name="variant",
        corruption_name="gaussian_noise_low",
        corruption_family="gaussian_noise",
        corruption_level="low",
        split="test",
        threshold=0.5,
        clean_recall=0.8,
        clean_fn=4,
        metrics=metrics,
    )
    assert row["recall_gap_vs_clean_test"] == 0.050000000000000044
    assert row["fn_increase_vs_clean_test"] == 1.0
    out = tmp_path / "report.md"
    write_corruption_report(pd.DataFrame([row]), out)
    assert "gaussian_noise_low" in out.read_text(encoding="utf-8")
