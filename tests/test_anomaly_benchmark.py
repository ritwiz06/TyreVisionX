from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.anomaly.check_backbone_benchmark_readiness import inspect_backbones
from src.anomaly.benchmark import load_benchmark_config, make_patch_grid, summarize_metadata


def test_benchmark_config_parses() -> None:
    config = load_benchmark_config("configs/anomaly/anomaly_benchmark.yaml")
    names = {variant["name"] for variant in config["variants"]}
    assert "resnet18_knn" in names
    assert "resnet50_knn" in names
    assert "resnet18_patch_grid_knn" in names
    assert config["base_run_config"]["data"]["normal_train_manifest"].endswith("D1_anomaly_train_normal.csv")


def test_backbone_readiness_payload_shape() -> None:
    payload = inspect_backbones()
    rows = {row["backbone"]: row for row in payload["backbones"]}
    assert "resnet18" in rows
    assert "resnet50" in rows
    assert "efficientnet_b0" in rows
    assert all("runnable_now" in row for row in rows.values())
    assert all("code_support_exists" in row for row in rows.values())


def test_patch_grid_shape() -> None:
    images = torch.zeros(2, 3, 224, 224)
    patches = make_patch_grid(images)
    assert patches.shape == (2, 5, 3, 224, 224)


def test_variant_metadata_summary_export_fields() -> None:
    metadata = {
        "feature_extractor": {"backbone": "resnet18", "weight_source": "test_weights", "embedding_dim": 512},
        "scoring": {"method": "knn"},
        "threshold": {"value": 0.5, "policy": "validation_only"},
        "validation": {"threshold_metrics": {"recall": 0.8, "normal_fpr": 0.1}},
        "test": {
            "auroc": 0.9,
            "auprc": 0.8,
            "threshold_metrics": {"recall": 0.7, "precision": 0.6, "normal_fpr": 0.2, "fn": 3, "fp": 4},
        },
        "outputs": {"run_dir": "artifacts/anomaly/test"},
    }
    row = summarize_metadata("variant_a", {"scorer": "knn", "embedding_type": "pooled_penultimate"}, metadata, "executed")
    assert row["variant"] == "variant_a"
    assert row["test_false_negatives"] == 3
    assert row["threshold"] == 0.5


def test_model_comparison_csv_has_required_columns_if_present() -> None:
    path = Path("reports/anomaly/model_comparison.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    required = {"variant", "run_status", "backbone", "scorer", "test_recall", "test_false_negatives"}
    assert required <= set(df.columns)
    assert np.isfinite(df["test_false_negatives"].dropna()).all()
