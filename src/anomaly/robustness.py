"""Corruption robustness and mild noise-robust anomaly variants."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.anomaly.corruptions import CorruptionSpec, apply_corruption, spec_from_dict
from src.anomaly.datasets import AnomalyManifestDataset
from src.anomaly.evaluate import evaluate_scores
from src.anomaly.features import ResNetEmbeddingExtractor, extract_embeddings, save_embeddings
from src.anomaly.io import load_yaml, write_json
from src.anomaly.pipeline import _device_from_config, _write_predictions_and_plots
from src.anomaly.scorers import KNNScorer, MahalanobisScorer, build_scorer
from src.anomaly.thresholds import select_recall_priority_threshold
from src.data.transforms import get_eval_transforms


class CorruptedAnomalyManifestDataset(AnomalyManifestDataset):
    """Anomaly dataset that applies one deterministic corruption before tensor transforms."""

    def __init__(self, manifest_path: str | Path, transforms, corruption: CorruptionSpec | None = None) -> None:
        super().__init__(manifest_path=manifest_path, transforms=transforms)
        self.corruption = corruption

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self._resolve_path(str(row["image_path"]))
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = apply_corruption(image, self.corruption, seed=index)
        image_tensor = self.transforms(image=image)["image"] if self.transforms else torch.from_numpy(image).permute(2, 0, 1)
        target = int(row["target"])
        meta = {
            "image_path": str(image_path),
            "target": target,
            "label": int(row.get("label", target)),
            "label_str": str(row.get("label_str", "")),
            "split": str(row.get("split", "")),
            "is_normal": bool(row["is_normal"]),
            "source_dataset": str(row.get("source_dataset", row.get("dataset_id", ""))),
            "product_type": str(row.get("product_type", "")),
        }
        return image_tensor, target, meta


def transform_config(image_size: list[int]) -> dict[str, Any]:
    return {"size": image_size, "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}


def corrupted_loader(
    manifest: str | Path,
    image_size: list[int],
    batch_size: int,
    num_workers: int,
    corruption: CorruptionSpec | None,
) -> DataLoader:
    dataset = CorruptedAnomalyManifestDataset(
        manifest_path=manifest,
        transforms=get_eval_transforms(transform_config(image_size)),
        corruption=corruption,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_scorer(method: str, artifact_dir: str | Path):
    artifact_dir = Path(artifact_dir)
    if method == "knn":
        return KNNScorer.load(artifact_dir / "knn_scorer.npz")
    if method == "mahalanobis":
        return MahalanobisScorer.load(artifact_dir / "mahalanobis_scorer.npz")
    raise ValueError(f"Unsupported scorer method: {method}")


def summarize_metrics(
    *,
    variant_name: str,
    corruption_name: str,
    corruption_family: str,
    corruption_level: str,
    split: str,
    threshold: float,
    clean_recall: float,
    clean_fn: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    threshold_metrics = metrics["threshold_metrics"]
    recall = float(threshold_metrics["recall"])
    fn = float(threshold_metrics["fn"])
    return {
        "variant": variant_name,
        "corruption": corruption_name,
        "family": corruption_family,
        "level": corruption_level,
        "split": split,
        "threshold": float(threshold),
        "recall": recall,
        "precision": float(threshold_metrics["precision"]),
        "normal_fpr": float(threshold_metrics["normal_fpr"]),
        "false_negatives": fn,
        "false_positives": float(threshold_metrics["fp"]),
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "recall_gap_vs_clean_test": float(clean_recall - recall) if split == "test" else np.nan,
        "fn_increase_vs_clean_test": float(fn - clean_fn) if split == "test" else np.nan,
    }


def run_corruption_benchmark(config: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Any]:
    data_cfg = config["data"]
    image_size = list(data_cfg.get("image_size", [224, 224]))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    device = _device_from_config(str(data_cfg.get("device", "auto")))
    corruptions = [spec_from_dict(item) for item in config["corruptions"]]
    rows: list[dict[str, Any]] = []
    out_csv = Path(config["outputs"]["corruption_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for variant in config["variants"]:
        metadata = load_yaml_or_json(Path(variant["artifact_dir"]) / "metadata.json")
        threshold = float(metadata["threshold"]["value"])
        clean_test = metadata["test"]["threshold_metrics"]
        clean_recall = float(clean_test["recall"])
        clean_fn = float(clean_test["fn"])
        extractor = ResNetEmbeddingExtractor(
            backbone=variant["backbone"],
            pretrained=True,
            normalize=True,
            weights_path=variant.get("weights_path"),
        )
        scorer = load_scorer(variant["scorer"], variant["artifact_dir"])
        for corruption in corruptions:
            for split, manifest_key in [("validation", "validation_manifest"), ("test", "test_manifest")]:
                loader = corrupted_loader(
                    data_cfg[manifest_key], image_size, batch_size, num_workers, corruption if corruption.name != "clean" else None
                )
                features = extract_embeddings(extractor, loader, device)
                scores = scorer.score(features.embeddings)
                metrics = evaluate_scores(features.targets, scores, threshold)
                rows.append(
                    summarize_metrics(
                        variant_name=variant["name"],
                        corruption_name=corruption.name,
                        corruption_family=corruption.family,
                        corruption_level=corruption.level,
                        split=split,
                        threshold=threshold,
                        clean_recall=clean_recall,
                        clean_fn=clean_fn,
                        metrics=metrics,
                    )
                )
                # Write incrementally so long CPU runs leave auditable partial output.
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    write_corruption_report(df, config["outputs"]["corruption_report"])
    return {"rows": rows, "csv": str(out_csv)}


def run_noise_robust_variant(config: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Any]:
    data_cfg = config["data"]
    image_size = list(data_cfg.get("image_size", [224, 224]))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    device = _device_from_config(str(data_cfg.get("device", "auto")))
    run_dir = Path(config["outputs"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    extractor = ResNetEmbeddingExtractor(
        backbone=config["feature_extractor"]["backbone"],
        pretrained=True,
        normalize=True,
        weights_path=config["feature_extractor"].get("weights_path"),
    )
    train_embeddings = []
    train_targets = []
    train_corruptions = [None] + [spec_from_dict(item) for item in config["train_augmentations"]]
    for corruption in train_corruptions:
        loader = corrupted_loader(
            data_cfg["normal_train_manifest"], image_size, batch_size, num_workers, corruption
        )
        features = extract_embeddings(extractor, loader, device)
        train_embeddings.append(features.embeddings)
        train_targets.append(features.targets)
    embeddings = np.concatenate(train_embeddings, axis=0)
    targets = np.concatenate(train_targets, axis=0)
    if int((targets != 0).sum()):
        raise ValueError("Noise-robust anomaly training still requires normal-only train data.")
    save_embeddings(run_dir / "train_embeddings_augmented.npy", embeddings)

    scorer_cfg = config["scoring"]
    scorer = build_scorer(
        method=scorer_cfg["method"],
        embeddings=embeddings,
        regularization=float(scorer_cfg.get("regularization", 1e-3)),
        k=int(scorer_cfg.get("knn_k", 5)),
    )
    scorer.save(run_dir / f"{scorer_cfg['method']}_scorer.npz")

    split_metrics: dict[str, Any] = {}
    split_outputs: dict[str, Any] = {}
    clean_val = extract_embeddings(
        extractor,
        corrupted_loader(data_cfg["validation_manifest"], image_size, batch_size, num_workers, None),
        device,
    )
    clean_val_scores = scorer.score(clean_val.embeddings)
    threshold_result = select_recall_priority_threshold(
        clean_val.targets,
        clean_val_scores,
        max_normal_fpr=float(config["threshold_policy"].get("max_normal_fpr", 0.10)),
    )
    for split, manifest_key in [("validation", "validation_manifest"), ("test", "test_manifest")]:
        features = clean_val if split == "validation" else extract_embeddings(
            extractor,
            corrupted_loader(data_cfg[manifest_key], image_size, batch_size, num_workers, None),
            device,
        )
        scores = clean_val_scores if split == "validation" else scorer.score(features.embeddings)
        split_metrics[split] = evaluate_scores(features.targets, scores, threshold_result.threshold)
        split_outputs[split] = _write_predictions_and_plots(
            run_dir, split, features.records, features.targets, scores, threshold_result.threshold
        )

    metadata = {
        "run_name": config["experiment"]["name"],
        "config_path": str(config_path) if config_path else None,
        "feature_extractor": {
            "backbone": config["feature_extractor"]["backbone"],
            "weights_path": config["feature_extractor"].get("weights_path"),
            "weight_source": extractor.weight_source,
            "embedding_dim": extractor.embedding_dim,
        },
        "training": {
            "normal_only": True,
            "augmentation_count": len(train_corruptions) - 1,
            "augmented_embedding_count": int(len(embeddings)),
        },
        "scoring": scorer_cfg,
        "threshold": {
            "value": threshold_result.threshold,
            "policy": threshold_result.policy,
            "fallback_used": threshold_result.fallback_used,
            "validation_metrics_at_threshold": threshold_result.validation_metrics,
        },
        "validation": split_metrics["validation"],
        "test": split_metrics["test"],
        "outputs": {"run_dir": str(run_dir), "validation": split_outputs["validation"], "test": split_outputs["test"]},
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "metrics_validation.json", split_metrics["validation"])
    write_json(run_dir / "metrics_test.json", split_metrics["test"])
    return metadata


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def write_corruption_report(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    test_df = df[df["split"] == "test"].copy()
    lines = [
        "# Corruption Benchmark",
        "",
        "Primary rule: use the clean validation-selected threshold for corrupted validation/test evaluation. No corrupted test retuning.",
        "",
        "| Variant | Corruption | Recall | Precision | FN | FP | AUROC | AUPRC | Recall Gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test_df.iterrows():
        lines.append(
            f"| `{row['variant']}` | {row['corruption']} | {row['recall']:.4f} | {row['precision']:.4f} | "
            f"{row['false_negatives']:.0f} | {row['false_positives']:.0f} | {row['auroc']:.4f} | "
            f"{row['auprc']:.4f} | {row['recall_gap_vs_clean_test']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
