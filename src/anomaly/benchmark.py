"""Controlled anomaly benchmark harness for TyreVisionX."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.anomaly.datasets import AnomalyManifestDataset
from src.anomaly.evaluate import evaluate_scores
from src.anomaly.features import ResNetEmbeddingExtractor, extract_embeddings, save_embeddings
from src.anomaly.io import load_yaml, write_json
from src.anomaly.pipeline import _device_from_config, _loader, _write_predictions_and_plots, run_anomaly_baseline
from src.anomaly.scorers import build_scorer
from src.anomaly.thresholds import metrics_at_threshold, select_recall_priority_threshold, threshold_sweep
from src.data.transforms import get_eval_transforms


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    if "variants" not in config or not isinstance(config["variants"], list):
        raise ValueError("Benchmark config must contain a list under 'variants'.")
    return config


def run_benchmark(config: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Any]:
    output_dir = Path(config["outputs"]["root_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for variant in config["variants"]:
        row = run_variant(config, variant, config_path=config_path)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison_path = Path(config["outputs"]["comparison_csv"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    write_model_comparison_markdown(comparison, config["outputs"]["comparison_report"])
    write_json(output_dir / "benchmark_summary.json", {"variants": rows, "comparison_csv": str(comparison_path)})
    return {"variants": rows, "comparison_csv": str(comparison_path)}


def run_variant(config: dict[str, Any], variant: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Any]:
    name = variant["name"]
    if variant.get("reuse_from"):
        return summarize_existing_variant(name, variant)
    if variant.get("mode") == "threshold_sweep":
        return run_threshold_sweep_variant(config, variant)
    if variant.get("embedding_type") == "patch_grid":
        metadata = run_patch_grid_variant(config, variant, config_path=config_path)
    else:
        single_config = build_single_run_config(config, variant)
        metadata = run_anomaly_baseline(single_config, config_path=config_path)
    return summarize_metadata(name, variant, metadata, "executed")


def build_single_run_config(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    base = copy.deepcopy(config["base_run_config"])
    base["experiment"]["name"] = variant["name"]
    base["feature_extractor"]["backbone"] = variant["backbone"]
    base["feature_extractor"]["weights_path"] = variant.get("weights_path")
    base["scoring"]["method"] = variant["scorer"]
    if "knn_k" in variant:
        base["scoring"]["knn_k"] = variant["knn_k"]
    if "max_normal_fpr" in variant:
        base["threshold_policy"]["max_normal_fpr"] = variant["max_normal_fpr"]
    base["outputs"]["run_dir"] = variant["run_dir"]
    return base


def summarize_existing_variant(name: str, variant: dict[str, Any]) -> dict[str, Any]:
    metadata_path = Path(variant["reuse_from"]) / "metadata.json"
    if not metadata_path.exists():
        return {"variant": name, "run_status": "pending_missing_reuse_artifact", "artifact_dir": variant["reuse_from"]}
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return summarize_metadata(name, variant, metadata, "reused_existing_result")


def summarize_metadata(name: str, variant: dict[str, Any], metadata: dict[str, Any], status: str) -> dict[str, Any]:
    test_metrics = metadata.get("test", {}).get("threshold_metrics", {})
    val_metrics = metadata.get("validation", {}).get("threshold_metrics", {})
    return {
        "variant": name,
        "run_status": status,
        "backbone": variant.get("backbone", metadata.get("feature_extractor", {}).get("backbone")),
        "weight_source": metadata.get("feature_extractor", {}).get("weight_source", variant.get("weights_path")),
        "embedding_type": variant.get("embedding_type", "pooled_penultimate"),
        "scorer": variant.get("scorer", metadata.get("scoring", {}).get("method")),
        "threshold_policy": metadata.get("threshold", {}).get("policy", variant.get("threshold_policy", "")),
        "threshold": metadata.get("threshold", {}).get("value"),
        "multi_crop_or_patch": variant.get("embedding_type") in {"patch_grid", "multi_crop"},
        "artifact_dir": metadata.get("outputs", {}).get("run_dir", variant.get("run_dir")),
        "test_auroc": metadata.get("test", {}).get("auroc"),
        "test_auprc": metadata.get("test", {}).get("auprc"),
        "test_recall": test_metrics.get("recall"),
        "test_precision": test_metrics.get("precision"),
        "test_normal_fpr": test_metrics.get("normal_fpr"),
        "test_false_negatives": test_metrics.get("fn"),
        "test_false_positives": test_metrics.get("fp"),
        "validation_recall": val_metrics.get("recall"),
        "validation_normal_fpr": val_metrics.get("normal_fpr"),
    }


def run_threshold_sweep_variant(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(variant["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(variant["source_run"])
    val = pd.read_csv(source_dir / "predictions_validation.csv")
    test = pd.read_csv(source_dir / "predictions_test.csv")
    max_fpr = float(variant.get("max_normal_fpr", 0.30))
    rows = threshold_sweep(val["target"].to_numpy(), val["anomaly_score"].to_numpy())
    sweep = pd.DataFrame(rows)
    sweep.to_csv(run_dir / "threshold_sweep_validation.csv", index=False)
    valid = sweep[sweep["normal_fpr"] <= max_fpr]
    if len(valid):
        best = valid.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
        fallback = False
        policy = f"threshold_sweep_max_recall_subject_to_normal_fpr<={max_fpr}"
    else:
        best = sweep.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
        fallback = True
        policy = "threshold_sweep_fallback_max_f1"
    threshold = float(best["threshold"])
    val_eval = evaluate_scores(val["target"].to_numpy(), val["anomaly_score"].to_numpy(), threshold)
    test_eval = evaluate_scores(test["target"].to_numpy(), test["anomaly_score"].to_numpy(), threshold)
    val_outputs = _write_predictions_and_plots(
        run_dir,
        "validation",
        val.to_dict(orient="records"),
        val["target"].to_numpy(),
        val["anomaly_score"].to_numpy(),
        threshold,
    )
    test_outputs = _write_predictions_and_plots(
        run_dir,
        "test",
        test.to_dict(orient="records"),
        test["target"].to_numpy(),
        test["anomaly_score"].to_numpy(),
        threshold,
    )
    metadata = {
        "run_name": variant["name"],
        "feature_extractor": {"backbone": variant["backbone"], "weight_source": "reused_resnet18_scores"},
        "scoring": {"method": variant["scorer"]},
        "threshold": {
            "value": threshold,
            "policy": policy,
            "fallback_used": fallback,
            "validation_metrics_at_threshold": metrics_at_threshold(
                val["target"].to_numpy(), val["anomaly_score"].to_numpy(), threshold
            ),
        },
        "validation": val_eval,
        "test": test_eval,
        "outputs": {"run_dir": str(run_dir), "validation": val_outputs, "test": test_outputs},
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "metrics_validation.json", val_eval)
    write_json(run_dir / "metrics_test.json", test_eval)
    return summarize_metadata(variant["name"], variant, metadata, "executed")


def run_patch_grid_variant(
    config: dict[str, Any], variant: dict[str, Any], config_path: str | Path | None = None
) -> dict[str, Any]:
    run_cfg = build_single_run_config(config, variant)
    run_dir = Path(run_cfg["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = run_cfg["data"]
    image_size = list(data_cfg.get("image_size", [224, 224]))
    device = _device_from_config(str(data_cfg.get("device", "auto")))
    transform_cfg = {"size": image_size, "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
    loaders = {
        "train": DataLoader(
            AnomalyManifestDataset(data_cfg["normal_train_manifest"], transforms=get_eval_transforms(transform_cfg)),
            batch_size=int(data_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
        ),
        "validation": DataLoader(
            AnomalyManifestDataset(data_cfg["validation_manifest"], transforms=get_eval_transforms(transform_cfg)),
            batch_size=int(data_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
        ),
        "test": DataLoader(
            AnomalyManifestDataset(data_cfg["test_manifest"], transforms=get_eval_transforms(transform_cfg)),
            batch_size=int(data_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
        ),
    }
    feature_cfg = run_cfg["feature_extractor"]
    extractor = ResNetEmbeddingExtractor(
        backbone=feature_cfg["backbone"],
        pretrained=bool(feature_cfg.get("pretrained", True)),
        normalize=bool(run_cfg.get("embedding", {}).get("normalize", True)),
        weights_path=feature_cfg.get("weights_path"),
    ).to(device)
    extractor.eval()

    train_batch = extract_patch_grid_embeddings(extractor, loaders["train"], device)
    if int((train_batch["targets"] != 0).sum()):
        raise ValueError("Patch-grid normal train manifest must contain only target=0.")
    train_patch_embeddings = train_batch["patch_embeddings"].reshape(-1, train_batch["patch_embeddings"].shape[-1])
    save_embeddings(run_dir / "train_patch_embeddings.npy", train_patch_embeddings)

    scorer = build_scorer(
        method=run_cfg["scoring"]["method"],
        embeddings=train_patch_embeddings,
        regularization=float(run_cfg["scoring"].get("regularization", 1e-3)),
        k=int(run_cfg["scoring"].get("knn_k", 5)),
    )
    scorer.save(run_dir / f"{run_cfg['scoring']['method']}_patch_grid_scorer.npz")

    split_outputs = {}
    split_metrics = {}
    threshold = None
    for split in ["validation", "test"]:
        batch = extract_patch_grid_embeddings(extractor, loaders[split], device)
        patch_scores = scorer.score(batch["patch_embeddings"].reshape(-1, batch["patch_embeddings"].shape[-1]))
        image_scores = patch_scores.reshape(batch["patch_embeddings"].shape[0], batch["patch_embeddings"].shape[1]).max(axis=1)
        save_embeddings(run_dir / f"{split}_patch_embeddings.npy", batch["patch_embeddings"].reshape(-1, batch["patch_embeddings"].shape[-1]))
        if split == "validation":
            threshold_result = select_recall_priority_threshold(
                batch["targets"], image_scores, max_normal_fpr=float(run_cfg["threshold_policy"].get("max_normal_fpr", 0.10))
            )
            threshold = threshold_result.threshold
            pd.DataFrame(threshold_sweep(batch["targets"], image_scores)).to_csv(
                run_dir / "threshold_sweep_validation.csv", index=False
            )
        assert threshold is not None
        split_metrics[split] = evaluate_scores(batch["targets"], image_scores, threshold)
        split_outputs[split] = _write_predictions_and_plots(
            run_dir, split, batch["records"], batch["targets"], image_scores, threshold
        )

    metadata = {
        "run_name": variant["name"],
        "config_path": str(config_path) if config_path else None,
        "feature_extractor": {
            "backbone": feature_cfg["backbone"],
            "pretrained": bool(feature_cfg.get("pretrained", True)),
            "weights_path": feature_cfg.get("weights_path"),
            "weight_source": extractor.weight_source,
            "embedding_dim": extractor.embedding_dim,
        },
        "embedding": {"method": "patch_grid_max_score", "patches_per_image": 5},
        "scoring": run_cfg["scoring"],
        "threshold": {
            "value": threshold,
            "policy": f"patch_grid_{run_cfg['threshold_policy'].get('primary', 'recall_priority')}",
            "fallback_used": False,
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


def extract_patch_grid_embeddings(
    extractor: ResNetEmbeddingExtractor, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    patch_batches: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, batch_targets, meta in loader:
            images = images.to(device)
            patches = make_patch_grid(images)
            b, p, c, h, w = patches.shape
            embeddings = extractor(patches.reshape(b * p, c, h, w)).reshape(b, p, -1).detach().cpu().numpy()
            patch_batches.append(embeddings)
            targets.append(batch_targets.detach().cpu().numpy().astype(int))
            for i in range(len(batch_targets)):
                records.append(
                    {
                        "image_path": str(_meta_value(meta, "image_path", i)),
                        "target": int(batch_targets[i]),
                        "label_str": str(_meta_value(meta, "label_str", i)),
                        "split": str(_meta_value(meta, "split", i)),
                        "source_dataset": str(_meta_value(meta, "source_dataset", i)),
                        "product_type": str(_meta_value(meta, "product_type", i)),
                    }
                )
    return {
        "patch_embeddings": np.concatenate(patch_batches, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "records": records,
    }


def make_patch_grid(images: torch.Tensor) -> torch.Tensor:
    """Return full image plus four quadrant crops resized back to input size."""

    _, _, h, w = images.shape
    h2, w2 = h // 2, w // 2
    crops = [
        images,
        images[:, :, :h2, :w2],
        images[:, :, :h2, w2:],
        images[:, :, h2:, :w2],
        images[:, :, h2:, w2:],
    ]
    resized = [crops[0]] + [F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False) for crop in crops[1:]]
    return torch.stack(resized, dim=1)


def _meta_value(meta: Any, key: str, index: int) -> Any:
    if isinstance(meta, dict):
        value = meta.get(key, "")
        if isinstance(value, (list, tuple)):
            return value[index]
        if torch.is_tensor(value):
            return value[index].item()
        return value
    return ""


def write_model_comparison_markdown(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Anomaly Model Comparison",
        "",
        "This table compares variants using validation-selected thresholds. Test metrics are reported once per executed variant.",
        "",
        "| Variant | Status | Backbone | Scorer | Embedding | Recall | Precision | Normal FPR | FN | FP | AUROC | AUPRC |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| `{row['variant']}` | {row['run_status']} | {row.get('backbone', '')} | {row.get('scorer', '')} | "
            f"{row.get('embedding_type', '')} | {_fmt(row.get('test_recall'))} | {_fmt(row.get('test_precision'))} | "
            f"{_fmt(row.get('test_normal_fpr'))} | {_fmt(row.get('test_false_negatives'), 0)} | "
            f"{_fmt(row.get('test_false_positives'), 0)} | {_fmt(row.get('test_auroc'))} | {_fmt(row.get('test_auprc'))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"
