"""End-to-end anomaly baseline pipeline."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.anomaly.datasets import AnomalyManifestDataset
from src.anomaly.evaluate import (
    build_predictions,
    evaluate_scores,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_score_distributions,
)
from src.anomaly.features import ResNetEmbeddingExtractor, extract_embeddings, save_embeddings
from src.anomaly.io import write_json
from src.anomaly.scorers import build_scorer
from src.anomaly.thresholds import select_recall_priority_threshold, threshold_sweep
from src.data.transforms import get_eval_transforms


def _device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but CUDA is not available.")
    return device


def _transform_config(image_size: list[int]) -> dict[str, Any]:
    return {
        "size": image_size,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def _loader(manifest: str | Path, image_size: list[int], batch_size: int, num_workers: int) -> DataLoader:
    ds = AnomalyManifestDataset(
        manifest_path=manifest,
        transforms=get_eval_transforms(_transform_config(image_size)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _ensure_normal_train(targets: np.ndarray) -> None:
    bad = int((targets != 0).sum())
    if bad:
        raise ValueError(f"normal_train manifest contains {bad} non-normal samples. It must contain only target=0.")


def _write_predictions_and_plots(
    out_dir: Path,
    split_name: str,
    records: list[dict[str, Any]],
    targets: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = build_predictions(records, targets, scores, threshold)
    predictions_path = out_dir / f"predictions_{split_name}.csv"
    predictions.to_csv(predictions_path, index=False)

    false_negatives_path = out_dir / f"false_negatives_{split_name}.csv"
    false_positives_path = out_dir / f"false_positives_{split_name}.csv"
    predictions[predictions["is_false_negative"] == 1].to_csv(false_negatives_path, index=False)
    predictions[predictions["is_false_positive"] == 1].to_csv(false_positives_path, index=False)

    dist_plot = plot_score_distributions(
        predictions,
        out_dir / f"score_distribution_{split_name}.png",
        title=f"Anomaly score distribution ({split_name})",
    )
    pr_plot = plot_pr_curve(targets, scores, out_dir / f"pr_curve_{split_name}.png")
    cm_plot = plot_confusion_matrix(
        evaluate_scores(targets, scores, threshold)["confusion_matrix"],
        out_dir / f"confusion_matrix_{split_name}.png",
        title=f"Anomaly confusion matrix ({split_name})",
    )

    return {
        "predictions_csv": str(predictions_path),
        "false_negatives_csv": str(false_negatives_path),
        "false_positives_csv": str(false_positives_path),
        "score_distribution_plot": dist_plot,
        "pr_curve_plot": pr_plot,
        "confusion_matrix_plot": cm_plot,
    }


def run_anomaly_baseline(config: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Any]:
    """Run fit, validation calibration, and test evaluation."""

    run_name = config["experiment"]["name"]
    output_dir = Path(config["outputs"]["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    image_size = list(data_cfg.get("image_size", [224, 224]))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    device = _device_from_config(str(data_cfg.get("device", "auto")))

    for role in ["normal_train_manifest", "validation_manifest", "test_manifest"]:
        path = Path(data_cfg[role])
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {role}: {path}. "
                "Run scripts/data/create_anomaly_manifests.py before running the anomaly baseline."
            )

    train_loader = _loader(data_cfg["normal_train_manifest"], image_size, batch_size, num_workers)
    val_loader = _loader(data_cfg["validation_manifest"], image_size, batch_size, num_workers)
    test_loader = _loader(data_cfg["test_manifest"], image_size, batch_size, num_workers)

    feature_cfg = config["feature_extractor"]
    try:
        extractor = ResNetEmbeddingExtractor(
            backbone=str(feature_cfg.get("backbone", "resnet18")),
            pretrained=bool(feature_cfg.get("pretrained", True)),
            normalize=bool(config.get("embedding", {}).get("normalize", True)),
            weights_path=feature_cfg.get("weights_path"),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize the frozen CNN feature extractor. "
            "If pretrained weights are missing in an offline environment, pre-download the torchvision "
            "weights or set feature_extractor.pretrained=false for smoke testing only."
        ) from exc

    train_features = extract_embeddings(extractor, train_loader, device)
    _ensure_normal_train(train_features.targets)
    val_features = extract_embeddings(extractor, val_loader, device)
    test_features = extract_embeddings(extractor, test_loader, device)

    save_embeddings(output_dir / "train_embeddings.npy", train_features.embeddings)
    save_embeddings(output_dir / "validation_embeddings.npy", val_features.embeddings)
    save_embeddings(output_dir / "test_embeddings.npy", test_features.embeddings)

    scoring_cfg = config["scoring"]
    scorer = build_scorer(
        method=str(scoring_cfg.get("method", "mahalanobis")),
        embeddings=train_features.embeddings,
        regularization=float(scoring_cfg.get("regularization", 1e-3)),
        k=int(scoring_cfg.get("knn_k", 5)),
    )
    scorer_path = output_dir / f"{scoring_cfg.get('method', 'mahalanobis')}_scorer.npz"
    scorer.save(scorer_path)

    val_scores = scorer.score(val_features.embeddings)
    test_scores = scorer.score(test_features.embeddings)

    threshold_cfg = config["threshold_policy"]
    threshold_result = select_recall_priority_threshold(
        targets=val_features.targets,
        scores=val_scores,
        max_normal_fpr=float(threshold_cfg.get("max_normal_fpr", 0.10)),
    )

    val_sweep = pd.DataFrame(threshold_sweep(val_features.targets, val_scores))
    val_sweep_path = output_dir / "threshold_sweep_validation.csv"
    val_sweep.to_csv(val_sweep_path, index=False)

    val_eval = evaluate_scores(val_features.targets, val_scores, threshold_result.threshold)
    test_eval = evaluate_scores(test_features.targets, test_scores, threshold_result.threshold)

    val_outputs = _write_predictions_and_plots(
        output_dir, "validation", val_features.records, val_features.targets, val_scores, threshold_result.threshold
    )
    test_outputs = _write_predictions_and_plots(
        output_dir, "test", test_features.records, test_features.targets, test_scores, threshold_result.threshold
    )

    metadata = {
        "run_name": run_name,
        "config_path": str(config_path) if config_path else None,
        "manifests": {
            "normal_train": data_cfg["normal_train_manifest"],
            "validation": data_cfg["validation_manifest"],
            "test": data_cfg["test_manifest"],
        },
        "feature_extractor": {
            "backbone": feature_cfg.get("backbone", "resnet18"),
            "pretrained": bool(feature_cfg.get("pretrained", True)),
            "weights_path": feature_cfg.get("weights_path"),
            "weight_source": extractor.weight_source,
            "embedding_dim": extractor.embedding_dim,
        },
        "scoring": scoring_cfg,
        "threshold": {
            "value": threshold_result.threshold,
            "policy": threshold_result.policy,
            "fallback_used": threshold_result.fallback_used,
            "validation_metrics_at_threshold": threshold_result.validation_metrics,
        },
        "validation": val_eval,
        "test": test_eval,
        "outputs": {
            "run_dir": str(output_dir),
            "scorer": str(scorer_path),
            "threshold_sweep_validation": str(val_sweep_path),
            "validation": val_outputs,
            "test": test_outputs,
        },
    }

    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "threshold.json", metadata["threshold"])
    write_json(output_dir / "metrics_validation.json", val_eval)
    write_json(output_dir / "metrics_test.json", test_eval)

    if config_path:
        shutil.copy(config_path, output_dir / "config.yaml")

    return metadata
