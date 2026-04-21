from pathlib import Path

import numpy as np
import torch

from src.anomaly.local_benchmark import load_local_benchmark_config, summarize_metadata
from src.anomaly.local_features import aggregate_local_scores, flatten_local_embeddings
from src.anomaly.multicrop import make_fine_patch_grid_batch, make_multicrop_batch


def test_local_feature_config_parses():
    cfg = load_local_benchmark_config("configs/anomaly/anomaly_local_feature_benchmark.yaml")
    assert cfg["benchmark"]["name"] == "d1_resnet50_local_feature_benchmark_v1"
    assert {variant["name"] for variant in cfg["variants"]} >= {
        "resnet50_knn_reference",
        "resnet50_multicrop_knn",
        "resnet50_patch_grid_knn_fine",
    }


def test_multicrop_shapes():
    images = torch.zeros(2, 3, 224, 224)
    crops = make_multicrop_batch(images, crop_fraction=0.82)
    assert crops.shape == (2, 6, 3, 224, 224)


def test_fine_patch_grid_shapes():
    images = torch.zeros(2, 3, 224, 224)
    crops = make_fine_patch_grid_batch(images, grid_size=3, include_full=True)
    assert crops.shape == (2, 10, 3, 224, 224)


def test_local_score_aggregation_and_flattening():
    embeddings = np.zeros((3, 4, 8), dtype=np.float32)
    flat = flatten_local_embeddings(embeddings)
    assert flat.shape == (12, 8)

    scores = np.array([0.1, 0.4, 0.2, 0.3, 0.9, 0.1, 0.2, 0.3], dtype=np.float32)
    max_scores = aggregate_local_scores(scores, n_images=2, n_crops=4, mode="max")
    top2_scores = aggregate_local_scores(scores, n_images=2, n_crops=4, mode="top2_mean")
    assert np.allclose(max_scores, [0.4, 0.9])
    assert np.allclose(top2_scores, [0.35, 0.6])


def test_variant_metadata_summary_exports_expected_fields(tmp_path: Path):
    metadata = {
        "feature_extractor": {"backbone": "resnet50"},
        "local_features": {"mode": "multicrop", "aggregation": "max", "crops_per_image": 6},
        "scoring": {"method": "knn"},
        "threshold": {"value": 1.5, "policy": "validation"},
        "validation": {"threshold_metrics": {"recall": 0.8, "normal_fpr": 0.1}},
        "test": {
            "threshold_metrics": {
                "recall": 0.75,
                "precision": 0.7,
                "normal_fpr": 0.2,
                "fn": 5,
                "fp": 3,
            },
            "auroc": 0.9,
            "auprc": 0.88,
        },
        "outputs": {"run_dir": str(tmp_path)},
    }
    row = summarize_metadata({"name": "demo"}, metadata, "executed")
    assert row["variant"] == "demo"
    assert row["local_aggregation_used"] is True
    assert row["test_false_negatives"] == 5
