"""Controlled local-feature anomaly benchmark."""
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
from src.anomaly.features import ResNetEmbeddingExtractor
from src.anomaly.io import load_yaml, write_json
from src.anomaly.local_features import (
    aggregate_local_scores,
    extract_local_embeddings,
    flatten_local_embeddings,
    save_local_embeddings,
)
from src.anomaly.multicrop import make_fine_patch_grid_batch, make_multicrop_batch
from src.anomaly.pipeline import _device_from_config, _write_predictions_and_plots
from src.anomaly.scorers import build_scorer
from src.anomaly.thresholds import metrics_at_threshold, threshold_sweep
from src.data.transforms import get_eval_transforms


def load_local_benchmark_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    if "variants" not in config:
        raise ValueError("Local-feature benchmark config must include variants.")
    return config


def run_local_feature_benchmark(config: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Any]:
    rows = []
    for variant in config["variants"]:
        if variant.get("reuse_from"):
            row = summarize_existing_variant(variant)
        elif variant.get("mode") == "threshold_sweep":
            row = run_threshold_sweep(config, variant)
        else:
            metadata = run_local_variant(config, variant, config_path)
            row = summarize_metadata(variant, metadata, "executed")
        rows.append(row)
        out_csv = Path(config["outputs"]["comparison_csv"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(config["outputs"]["comparison_csv"], index=False)
    write_comparison_report(df, config["outputs"]["comparison_report"])
    return {"variants": rows, "comparison_csv": config["outputs"]["comparison_csv"]}


def run_local_variant(config: dict[str, Any], variant: dict[str, Any], config_path: str | Path | None) -> dict[str, Any]:
    data_cfg = config["data"]
    run_dir = Path(variant["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_config(str(data_cfg.get("device", "auto")))
    image_size = list(data_cfg.get("image_size", [224, 224]))
    transform_cfg = {"size": image_size, "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))
    loaders = {
        key: DataLoader(
            AnomalyManifestDataset(path, transforms=get_eval_transforms(transform_cfg)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        for key, path in {
            "train": data_cfg["normal_train_manifest"],
            "validation": data_cfg["validation_manifest"],
            "test": data_cfg["test_manifest"],
        }.items()
    }
    extractor = ResNetEmbeddingExtractor(
        backbone=variant["backbone"],
        pretrained=True,
        normalize=True,
        weights_path=variant.get("weights_path"),
    )
    crop_fn = build_crop_fn(variant)
    train = extract_local_embeddings(extractor, loaders["train"], device, crop_fn)
    if int((train["targets"] != 0).sum()):
        raise ValueError("Local-feature normal train manifest must contain only target=0.")
    train_flat = flatten_local_embeddings(train["local_embeddings"])
    save_local_embeddings(run_dir / "train_local_embeddings.npy", train["local_embeddings"])
    scorer = build_scorer(
        method=variant.get("scorer", "knn"),
        embeddings=train_flat,
        k=int(variant.get("knn_k", 5)),
        regularization=float(variant.get("regularization", 1e-3)),
    )
    scorer.save(run_dir / f"{variant.get('scorer', 'knn')}_local_scorer.npz")

    split_metrics = {}
    split_outputs = {}
    threshold = None
    for split in ["validation", "test"]:
        batch = extract_local_embeddings(extractor, loaders[split], device, crop_fn)
        save_local_embeddings(run_dir / f"{split}_local_embeddings.npy", batch["local_embeddings"])
        local_scores = scorer.score(flatten_local_embeddings(batch["local_embeddings"]))
        scores = aggregate_local_scores(
            local_scores,
            n_images=batch["local_embeddings"].shape[0],
            n_crops=batch["local_embeddings"].shape[1],
            mode=str(variant.get("aggregation", "max")),
        )
        if split == "validation":
            threshold = select_threshold(scores, batch["targets"], float(variant.get("max_normal_fpr", 0.10)), variant)
            pd.DataFrame(threshold_sweep(batch["targets"], scores)).to_csv(run_dir / "threshold_sweep_validation.csv", index=False)
        assert threshold is not None
        split_metrics[split] = evaluate_scores(batch["targets"], scores, threshold)
        split_outputs[split] = _write_predictions_and_plots(run_dir, split, batch["records"], batch["targets"], scores, threshold)

    metadata = {
        "run_name": variant["name"],
        "config_path": str(config_path) if config_path else None,
        "feature_extractor": {
            "backbone": variant["backbone"],
            "weights_path": variant.get("weights_path"),
            "weight_source": extractor.weight_source,
            "embedding_dim": extractor.embedding_dim,
        },
        "local_features": {
            "mode": variant.get("local_mode"),
            "aggregation": variant.get("aggregation", "max"),
            "crops_per_image": int(train["local_embeddings"].shape[1]),
        },
        "scoring": {"method": variant.get("scorer", "knn"), "knn_k": int(variant.get("knn_k", 5))},
        "threshold": {
            "value": threshold,
            "policy": str(variant.get("threshold_policy", "validation_recall_priority")),
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


def build_crop_fn(variant: dict[str, Any]):
    mode = variant.get("local_mode")
    if mode == "multicrop":
        return lambda images: make_multicrop_batch(images, crop_fraction=float(variant.get("crop_fraction", 0.82)))
    if mode == "patch_grid_fine":
        return lambda images: make_fine_patch_grid_batch(images, grid_size=int(variant.get("grid_size", 3)), include_full=True)
    raise ValueError(f"Unsupported local feature mode: {mode}")


def select_threshold(scores: np.ndarray, targets: np.ndarray, max_fpr: float, variant: dict[str, Any]) -> float:
    rows = pd.DataFrame(threshold_sweep(targets, scores))
    if variant.get("threshold_policy") == "threshold_sweep_fpr_0.30":
        max_fpr = 0.30
    valid = rows[rows["normal_fpr"] <= max_fpr]
    if len(valid):
        best = valid.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        best = rows.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"])


def run_threshold_sweep(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    source_dir = Path(variant["source_run"])
    run_dir = Path(variant["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(source_dir / "predictions_validation.csv")
    test = pd.read_csv(source_dir / "predictions_test.csv")
    max_fpr = float(variant.get("max_normal_fpr", 0.30))
    rows = pd.DataFrame(threshold_sweep(val["target"].to_numpy(), val["anomaly_score"].to_numpy()))
    rows.to_csv(run_dir / "threshold_sweep_validation.csv", index=False)
    valid = rows[rows["normal_fpr"] <= max_fpr]
    best = valid.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    threshold = float(best["threshold"])
    val_eval = evaluate_scores(val["target"].to_numpy(), val["anomaly_score"].to_numpy(), threshold)
    test_eval = evaluate_scores(test["target"].to_numpy(), test["anomaly_score"].to_numpy(), threshold)
    val_outputs = _write_predictions_and_plots(run_dir, "validation", val.to_dict(orient="records"), val["target"].to_numpy(), val["anomaly_score"].to_numpy(), threshold)
    test_outputs = _write_predictions_and_plots(run_dir, "test", test.to_dict(orient="records"), test["target"].to_numpy(), test["anomaly_score"].to_numpy(), threshold)
    metadata = {
        "run_name": variant["name"],
        "feature_extractor": {"backbone": "resnet50", "weight_source": "reused_resnet50_knn_scores"},
        "local_features": {"mode": "none", "aggregation": "none", "crops_per_image": 1},
        "scoring": {"method": "knn"},
        "threshold": {
            "value": threshold,
            "policy": f"validation_threshold_sweep_max_recall_subject_to_fpr<={max_fpr}",
            "validation_metrics_at_threshold": metrics_at_threshold(val["target"].to_numpy(), val["anomaly_score"].to_numpy(), threshold),
        },
        "validation": val_eval,
        "test": test_eval,
        "outputs": {"run_dir": str(run_dir), "validation": val_outputs, "test": test_outputs},
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "metrics_validation.json", val_eval)
    write_json(run_dir / "metrics_test.json", test_eval)
    return summarize_metadata(variant, metadata, "executed")


def summarize_existing_variant(variant: dict[str, Any]) -> dict[str, Any]:
    metadata = json.loads((Path(variant["reuse_from"]) / "metadata.json").read_text(encoding="utf-8"))
    return summarize_metadata(variant, metadata, "reused_existing_result")


def summarize_metadata(variant: dict[str, Any], metadata: dict[str, Any], status: str) -> dict[str, Any]:
    test = metadata["test"]["threshold_metrics"]
    val = metadata["validation"]["threshold_metrics"]
    local = metadata.get("local_features", {})
    return {
        "variant": variant["name"],
        "run_status": status,
        "backbone": metadata.get("feature_extractor", {}).get("backbone", variant.get("backbone")),
        "scorer": metadata.get("scoring", {}).get("method", variant.get("scorer", "knn")),
        "threshold": metadata.get("threshold", {}).get("value"),
        "threshold_policy": metadata.get("threshold", {}).get("policy"),
        "local_mode": local.get("mode", variant.get("local_mode", "none")),
        "aggregation": local.get("aggregation", variant.get("aggregation", "none")),
        "crops_per_image": local.get("crops_per_image", 1),
        "local_aggregation_used": local.get("mode", "none") not in {"none", None},
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


def write_comparison_report(df: pd.DataFrame, path: str | Path) -> None:
    lines = [
        "# Local Feature Model Comparison",
        "",
        "Thresholds are selected on validation only and applied once to test.",
        "",
        "| Variant | Status | Local Mode | Crops | Recall | Precision | FN | FP | AUROC | AUPRC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| `{row['variant']}` | {row['run_status']} | {row['local_mode']} | {int(row['crops_per_image'])} | "
            f"{float(row['test_recall']):.4f} | {float(row['test_precision']):.4f} | "
            f"{float(row['test_false_negatives']):.0f} | {float(row['test_false_positives']):.0f} | "
            f"{float(row['test_auroc']):.4f} | {float(row['test_auprc']):.4f} |"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
