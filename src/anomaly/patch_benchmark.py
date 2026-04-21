"""Controlled patch-aware anomaly benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.anomaly.datasets import AnomalyManifestDataset
from src.anomaly.evaluate import evaluate_scores
from src.anomaly.feature_map_patches import PatchFeatureBatch, ResNetFeatureMapExtractor, extract_patch_features
from src.anomaly.io import load_yaml, write_json
from src.anomaly.patch_memory import PatchMemoryScorer, RobustScoreNormalizer, build_memory_bank
from src.anomaly.pipeline import _device_from_config, _write_predictions_and_plots
from src.anomaly.thresholds import metrics_at_threshold, threshold_sweep
from src.data.transforms import get_eval_transforms


def load_patch_benchmark_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    if "variants" not in config:
        raise ValueError("Patch-aware benchmark config must include variants.")
    return config


def run_patch_aware_benchmark(config: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in config["variants"]:
        if variant.get("reuse_from"):
            row = summarize_existing_variant(variant)
        else:
            metadata = run_patch_variant(config, variant, config_path)
            row = summarize_patch_metadata(variant, metadata, "executed")
        rows.append(row)
        out_csv = Path(config["outputs"]["comparison_csv"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(config["outputs"]["comparison_csv"], index=False)
    write_patch_comparison_report(df, config["outputs"]["comparison_report"])
    return {"variants": rows, "comparison_csv": config["outputs"]["comparison_csv"]}


def run_patch_variant(config: dict[str, Any], variant: dict[str, Any], config_path: str | Path | None) -> dict[str, Any]:
    data_cfg = config["data"]
    run_dir = Path(variant["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_config(str(data_cfg.get("device", "auto")))
    image_size = list(data_cfg.get("image_size", [224, 224]))
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))
    transform_cfg = {"size": image_size, "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
    loaders = {
        split: DataLoader(
            AnomalyManifestDataset(path, transforms=get_eval_transforms(transform_cfg)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        for split, path in {
            "train": data_cfg["normal_train_manifest"],
            "validation": data_cfg["validation_manifest"],
            "test": data_cfg["test_manifest"],
        }.items()
    }

    extractor = ResNetFeatureMapExtractor(
        backbone=variant.get("backbone", "resnet50"),
        layer=variant.get("feature_layer", "layer4"),
        weights_path=variant.get("weights_path"),
        pretrained=True,
        normalize=True,
    )
    max_patches_per_image = variant.get("max_patches_per_image")
    train = extract_patch_features(extractor, loaders["train"], device, max_patches_per_image=max_patches_per_image)
    if int((train.targets != 0).sum()):
        raise ValueError("Patch-aware normal train manifest must contain only target=0.")
    np.save(run_dir / "train_patch_embeddings.npy", train.patch_embeddings)
    memory = build_memory_bank(
        train.patch_embeddings,
        max_memory_patches=int(variant.get("max_memory_patches", 10000)),
        seed=int(variant.get("seed", 42)),
    )
    scorer = PatchMemoryScorer(memory_bank=memory, k=int(variant.get("knn_k", 1)))
    scorer.save(run_dir / "patch_memory_scorer.npz")

    split_metrics: dict[str, Any] = {}
    split_outputs: dict[str, Any] = {}
    split_batches: dict[str, PatchFeatureBatch] = {}
    raw_scores_by_split: dict[str, np.ndarray] = {}
    scores_by_split: dict[str, np.ndarray] = {}
    normalizer = None
    for split in ["validation", "test"]:
        batch = extract_patch_features(extractor, loaders[split], device, max_patches_per_image=max_patches_per_image)
        np.save(run_dir / f"{split}_patch_embeddings.npy", batch.patch_embeddings)
        raw_scores = scorer.score_images(
            batch.patch_embeddings,
            batch.image_indices,
            n_images=len(batch.records),
            aggregation=str(variant.get("aggregation", "max")),
        )
        split_batches[split] = batch
        raw_scores_by_split[split] = raw_scores
        if split == "validation":
            normalizer = fit_score_normalizer(raw_scores, batch.targets, variant)
        assert normalizer is not None
        scores_by_split[split] = normalizer.transform(raw_scores) if normalizer else raw_scores

    threshold = None
    for split in ["validation", "test"]:
        batch = split_batches[split]
        scores = scores_by_split[split]
        if split == "validation":
            pd.DataFrame(threshold_sweep(batch.targets, scores)).to_csv(run_dir / "threshold_sweep_validation.csv", index=False)
            threshold = select_patch_threshold(scores, batch.targets, variant)
        assert threshold is not None
        split_metrics[split] = evaluate_scores(batch.targets, scores, threshold)
        split_outputs[split] = _write_predictions_and_plots(run_dir, split, batch.records, batch.targets, scores, threshold)

    metadata = {
        "run_name": variant["name"],
        "config_path": str(config_path) if config_path else None,
        "feature_extractor": {
            "backbone": extractor.backbone,
            "feature_layer": extractor.layer,
            "weights_path": variant.get("weights_path"),
            "weight_source": extractor.weight_source,
            "embedding_dim": extractor.embedding_dim,
        },
        "patch_aware": {
            "method": variant.get("method", "featuremap_patch_knn"),
            "patches_per_image": train.patches_per_image,
            "max_patches_per_image": max_patches_per_image,
            "memory_patches": int(len(memory)),
            "aggregation": variant.get("aggregation", "max"),
            "max_memory_patches": int(variant.get("max_memory_patches", 10000)),
            "score_normalization": {
                "enabled": bool(variant.get("score_normalization", True)),
                "fit_split": "validation_normal" if variant.get("score_normalization", True) else "none",
                "normalizer": normalizer.to_dict() if normalizer else None,
            },
        },
        "scoring": {"method": "patch_knn", "knn_k": int(variant.get("knn_k", 1))},
        "threshold": {
            "value": threshold,
            "policy": threshold_policy_name(variant),
            "validation_metrics_at_threshold": split_metrics["validation"]["threshold_metrics"],
        },
        "validation": split_metrics["validation"],
        "test": split_metrics["test"],
        "outputs": {"run_dir": str(run_dir), "validation": split_outputs["validation"], "test": split_outputs["test"]},
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "metrics_validation.json", split_metrics["validation"])
    write_json(run_dir / "metrics_test.json", split_metrics["test"])
    return metadata


def select_patch_threshold(scores: np.ndarray, targets: np.ndarray, variant: dict[str, Any]) -> float:
    rows = pd.DataFrame(threshold_sweep(targets, scores))
    max_fpr = float(variant.get("max_normal_fpr", 0.10))
    valid = rows[rows["normal_fpr"] <= max_fpr]
    if len(valid):
        best = valid.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        best = rows.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"])


def fit_score_normalizer(scores: np.ndarray, targets: np.ndarray, variant: dict[str, Any]) -> RobustScoreNormalizer:
    """Fit score normalizer on validation-normal scores only."""

    if not variant.get("score_normalization", True):
        return RobustScoreNormalizer(median=0.0, scale=1.0)
    normal_scores = scores[np.asarray(targets).astype(int) == 0]
    if len(normal_scores) == 0:
        normal_scores = scores
    return RobustScoreNormalizer.fit(normal_scores)


def threshold_policy_name(variant: dict[str, Any]) -> str:
    return f"validation_patch_threshold_max_recall_subject_to_fpr<={float(variant.get('max_normal_fpr', 0.10))}"


def summarize_existing_variant(variant: dict[str, Any]) -> dict[str, Any]:
    metadata = json.loads((Path(variant["reuse_from"]) / "metadata.json").read_text(encoding="utf-8"))
    row = summarize_patch_metadata(variant, metadata, "reused_existing_result")
    row["patch_method"] = "pooled_embedding_reference"
    row["feature_layer"] = "pooled_penultimate"
    row["memory_patches"] = None
    row["patches_per_image"] = 1
    return row


def summarize_patch_metadata(variant: dict[str, Any], metadata: dict[str, Any], status: str) -> dict[str, Any]:
    test = metadata["test"]["threshold_metrics"]
    val = metadata["validation"]["threshold_metrics"]
    patch = metadata.get("patch_aware", {})
    feature = metadata.get("feature_extractor", {})
    return {
        "variant": variant["name"],
        "run_status": status,
        "backbone": feature.get("backbone", variant.get("backbone")),
        "feature_layer": feature.get("feature_layer", variant.get("feature_layer", "unknown")),
        "patch_method": patch.get("method", variant.get("method", "reference")),
        "aggregation": patch.get("aggregation", variant.get("aggregation", "none")),
        "score_normalization": patch.get("score_normalization", {}).get("enabled", False),
        "memory_patches": patch.get("memory_patches"),
        "patches_per_image": patch.get("patches_per_image", 1),
        "threshold": metadata.get("threshold", {}).get("value"),
        "threshold_policy": metadata.get("threshold", {}).get("policy"),
        "artifact_dir": metadata.get("outputs", {}).get("run_dir", variant.get("run_dir", variant.get("reuse_from"))),
        "validation_recall": val.get("recall"),
        "validation_normal_fpr": val.get("normal_fpr"),
        "test_auroc": metadata["test"].get("auroc"),
        "test_auprc": metadata["test"].get("auprc"),
        "test_recall": test.get("recall"),
        "test_precision": test.get("precision"),
        "test_normal_fpr": test.get("normal_fpr"),
        "test_false_negatives": test.get("fn"),
        "test_false_positives": test.get("fp"),
    }


def write_patch_comparison_report(df: pd.DataFrame, path: str | Path) -> None:
    lines = [
        "# Patch-Aware Model Comparison",
        "",
        "Thresholds are selected on validation only and applied once to test.",
        "",
        "| Variant | Status | Layer | Method | Normalized | Memory Patches | Recall | Precision | FN | FP | AUROC | AUPRC |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        memory = "" if pd.isna(row.get("memory_patches")) else f"{float(row['memory_patches']):.0f}"
        lines.append(
            f"| `{row['variant']}` | {row['run_status']} | {row['feature_layer']} | {row['patch_method']} | "
            f"{bool(row.get('score_normalization', False))} | {memory} | "
            f"{float(row['test_recall']):.4f} | {float(row['test_precision']):.4f} | "
            f"{float(row['test_false_negatives']):.0f} | {float(row['test_false_positives']):.0f} | "
            f"{float(row['test_auroc']):.4f} | {float(row['test_auprc']):.4f} |"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
