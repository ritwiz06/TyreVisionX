"""Patch-memory nearest-neighbor scoring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class PatchMemoryScorer:
    """Nearest-neighbor anomaly score for local patch descriptors."""

    memory_bank: np.ndarray
    k: int = 1

    def __post_init__(self) -> None:
        if self.memory_bank.ndim != 2 or len(self.memory_bank) < 1:
            raise ValueError("memory_bank must have shape (n_patches, n_features)")
        self.k = min(int(self.k), len(self.memory_bank))
        self._nn = NearestNeighbors(n_neighbors=self.k)
        self._nn.fit(self.memory_bank)

    def score_patches(self, patch_embeddings: np.ndarray) -> np.ndarray:
        distances, _ = self._nn.kneighbors(patch_embeddings)
        return distances.mean(axis=1)

    def score_images(
        self,
        patch_embeddings: np.ndarray,
        image_indices: np.ndarray,
        n_images: int,
        aggregation: str = "max",
    ) -> np.ndarray:
        patch_scores = self.score_patches(patch_embeddings)
        return aggregate_patch_scores(patch_scores, image_indices, n_images, aggregation=aggregation)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, memory_bank=self.memory_bank, k=self.k)

    @classmethod
    def load(cls, path: str | Path) -> "PatchMemoryScorer":
        data = np.load(path)
        return cls(memory_bank=data["memory_bank"], k=int(data["k"]))


def build_memory_bank(
    patch_embeddings: np.ndarray,
    max_memory_patches: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Build a deterministic reduced memory bank from normal training patches."""

    if patch_embeddings.ndim != 2:
        raise ValueError("patch_embeddings must have shape (n_patches, n_features)")
    max_memory_patches = int(max_memory_patches)
    if max_memory_patches <= 0:
        raise ValueError("max_memory_patches must be positive")
    if len(patch_embeddings) <= max_memory_patches:
        return patch_embeddings.astype(np.float32, copy=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(patch_embeddings), size=max_memory_patches, replace=False)
    indices.sort()
    return patch_embeddings[indices].astype(np.float32, copy=False)


def aggregate_patch_scores(
    patch_scores: np.ndarray,
    image_indices: np.ndarray,
    n_images: int,
    aggregation: str = "max",
) -> np.ndarray:
    """Aggregate patch anomaly scores into image anomaly scores."""

    output = np.zeros(n_images, dtype=np.float32)
    for idx in range(n_images):
        scores = patch_scores[image_indices == idx]
        if len(scores) == 0:
            output[idx] = 0.0
        elif aggregation == "max":
            output[idx] = float(scores.max())
        elif aggregation == "top3_mean":
            output[idx] = float(np.sort(scores)[-min(3, len(scores)) :].mean())
        elif aggregation == "mean":
            output[idx] = float(scores.mean())
        else:
            raise ValueError(f"Unsupported patch score aggregation: {aggregation}")
    return output


@dataclass
class RobustScoreNormalizer:
    """Median/MAD normalizer for image-level anomaly scores."""

    median: float
    scale: float

    @classmethod
    def fit(cls, scores: np.ndarray, eps: float = 1e-6) -> "RobustScoreNormalizer":
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            raise ValueError("Cannot fit score normalizer from empty scores.")
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        scale = max(1.4826 * mad, eps)
        return cls(median=median, scale=scale)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return ((np.asarray(scores, dtype=np.float32) - self.median) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"median": float(self.median), "scale": float(self.scale)}
