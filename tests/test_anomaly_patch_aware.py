import numpy as np
import torch

from src.anomaly.feature_map_patches import ResNetFeatureMapExtractor
from src.anomaly.patch_benchmark import load_patch_benchmark_config, summarize_patch_metadata
from src.anomaly.patch_memory import PatchMemoryScorer, RobustScoreNormalizer, aggregate_patch_scores, build_memory_bank


def test_patch_aware_config_parses():
    cfg = load_patch_benchmark_config("configs/anomaly/anomaly_patch_aware_benchmark.yaml")
    assert cfg["benchmark"]["name"] == "d1_resnet50_patch_aware_benchmark_v1"
    assert {variant["name"] for variant in cfg["variants"]} >= {
        "resnet50_knn_threshold_sweep_reference",
        "resnet50_featuremap_patch_knn",
        "resnet50_featuremap_patch_knn_threshold_sweep",
    }


def test_feature_map_patch_extractor_shape_without_pretrained():
    model = ResNetFeatureMapExtractor(backbone="resnet50", layer="layer4", pretrained=False)
    images = torch.zeros(2, 3, 224, 224)
    features = model(images)
    assert features.shape[0] == 2
    assert features.shape[1] == 2048
    assert features.shape[2:] == (7, 7)


def test_layer3_and_layer23_feature_map_shapes_without_pretrained():
    images = torch.zeros(1, 3, 224, 224)
    layer3 = ResNetFeatureMapExtractor(backbone="resnet50", layer="layer3", pretrained=False)(images)
    layer23 = ResNetFeatureMapExtractor(backbone="resnet50", layer="layer2_layer3", pretrained=False)(images)
    assert layer3.shape == (1, 1024, 14, 14)
    assert layer23.shape == (1, 1536, 28, 28)


def test_layer_specific_patch_configs_parse():
    layer3 = load_patch_benchmark_config("configs/anomaly/anomaly_patch_aware_layer3.yaml")
    layer23 = load_patch_benchmark_config("configs/anomaly/anomaly_patch_aware_layer23.yaml")
    assert any(v["feature_layer"] == "layer3" for v in layer3["variants"] if "feature_layer" in v)
    assert any(v["feature_layer"] == "layer2_layer3" for v in layer23["variants"] if "feature_layer" in v)


def test_patch_memory_bank_and_scoring():
    rng = np.random.default_rng(42)
    patches = rng.normal(size=(20, 8)).astype(np.float32)
    memory = build_memory_bank(patches, max_memory_patches=5, seed=7)
    assert memory.shape == (5, 8)
    scorer = PatchMemoryScorer(memory_bank=memory, k=1)
    scores = scorer.score_patches(patches[:3])
    assert scores.shape == (3,)
    assert np.all(scores >= 0)


def test_patch_score_aggregation():
    scores = np.array([0.1, 0.4, 0.2, 0.9, 0.3, 0.5], dtype=np.float32)
    image_indices = np.array([0, 0, 0, 1, 1, 1])
    assert np.allclose(aggregate_patch_scores(scores, image_indices, 2, "max"), [0.4, 0.9])
    assert np.allclose(aggregate_patch_scores(scores, image_indices, 2, "top3_mean"), [0.23333333, 0.56666666])


def test_robust_score_normalizer():
    normalizer = RobustScoreNormalizer.fit(np.array([1.0, 1.1, 1.2, 5.0], dtype=np.float32))
    transformed = normalizer.transform(np.array([1.1], dtype=np.float32))
    assert transformed.shape == (1,)
    assert abs(float(transformed[0])) < 1.0


def test_patch_metadata_summary():
    metadata = {
        "feature_extractor": {"backbone": "resnet50", "feature_layer": "layer4"},
        "patch_aware": {
            "method": "featuremap_patch_knn",
            "aggregation": "max",
            "memory_patches": 100,
            "patches_per_image": 49,
        },
        "threshold": {"value": 0.5, "policy": "validation"},
        "validation": {"threshold_metrics": {"recall": 0.9, "normal_fpr": 0.1}},
        "test": {
            "threshold_metrics": {
                "recall": 0.8,
                "precision": 0.75,
                "normal_fpr": 0.2,
                "fn": 4,
                "fp": 6,
            },
            "auroc": 0.91,
            "auprc": 0.92,
        },
        "outputs": {"run_dir": "artifacts/demo"},
    }
    row = summarize_patch_metadata({"name": "demo"}, metadata, "executed")
    assert row["variant"] == "demo"
    assert row["memory_patches"] == 100
    assert row["test_false_negatives"] == 4
