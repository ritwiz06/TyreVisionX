"""Local-feature extraction and aggregation for anomaly scoring."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.anomaly.features import ResNetEmbeddingExtractor, save_embeddings


def extract_local_embeddings(
    extractor: ResNetEmbeddingExtractor,
    loader: DataLoader,
    device: torch.device,
    crop_fn: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    """Extract local crop embeddings with shape (n_images, n_crops, dim)."""

    extractor.to(device)
    extractor.eval()
    embedding_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets, meta in loader:
            images = images.to(device)
            crops = crop_fn(images)
            b, cnum, channels, h, w = crops.shape
            embeddings = extractor(crops.reshape(b * cnum, channels, h, w)).reshape(b, cnum, -1)
            embedding_batches.append(embeddings.detach().cpu().numpy())
            targets_np = targets.detach().cpu().numpy().astype(int)
            target_batches.append(targets_np)
            for i in range(len(targets_np)):
                records.append(
                    {
                        "image_path": str(_meta_value(meta, "image_path", i)),
                        "target": int(targets_np[i]),
                        "label_str": str(_meta_value(meta, "label_str", i)),
                        "split": str(_meta_value(meta, "split", i)),
                        "source_dataset": str(_meta_value(meta, "source_dataset", i)),
                        "product_type": str(_meta_value(meta, "product_type", i)),
                    }
                )
    return {
        "local_embeddings": np.concatenate(embedding_batches, axis=0),
        "targets": np.concatenate(target_batches, axis=0),
        "records": records,
    }


def flatten_local_embeddings(local_embeddings: np.ndarray) -> np.ndarray:
    return local_embeddings.reshape(-1, local_embeddings.shape[-1])


def aggregate_local_scores(local_scores: np.ndarray, n_images: int, n_crops: int, mode: str = "max") -> np.ndarray:
    scores = local_scores.reshape(n_images, n_crops)
    if mode == "max":
        return scores.max(axis=1)
    if mode == "top2_mean":
        sorted_scores = np.sort(scores, axis=1)
        return sorted_scores[:, -2:].mean(axis=1)
    if mode == "mean":
        return scores.mean(axis=1)
    raise ValueError(f"Unsupported local score aggregation mode: {mode}")


def save_local_embeddings(path: str | Path, local_embeddings: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, local_embeddings)


def _meta_value(meta: Any, key: str, index: int) -> Any:
    if isinstance(meta, dict):
        value = meta.get(key, "")
        if isinstance(value, (list, tuple)):
            return value[index]
        if torch.is_tensor(value):
            return value[index].item()
        return value
    return ""
