"""Anomaly scoring methods."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class MahalanobisScorer:
    """Mahalanobis distance to the normal training embedding distribution."""

    mean: np.ndarray
    precision: np.ndarray
    regularization: float = 1e-3

    @classmethod
    def fit(cls, embeddings: np.ndarray, regularization: float = 1e-3) -> "MahalanobisScorer":
        if embeddings.ndim != 2:
            raise ValueError("Expected embeddings with shape (n_samples, n_features)")
        if len(embeddings) < 2:
            raise ValueError("At least two normal embeddings are required for Mahalanobis fitting.")

        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        cov = np.cov(centered, rowvar=False)
        cov = np.atleast_2d(cov)
        cov += np.eye(cov.shape[0]) * regularization
        precision = np.linalg.pinv(cov)
        return cls(mean=mean, precision=precision, regularization=regularization)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        centered = embeddings - self.mean
        squared = np.einsum("ij,jk,ik->i", centered, self.precision, centered)
        return np.sqrt(np.maximum(squared, 0.0))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, precision=self.precision, regularization=self.regularization)

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisScorer":
        data = np.load(path)
        return cls(
            mean=data["mean"],
            precision=data["precision"],
            regularization=float(data["regularization"]),
        )


@dataclass
class KNNScorer:
    """k-nearest-neighbor distance anomaly score."""

    train_embeddings: np.ndarray
    k: int = 5

    def __post_init__(self) -> None:
        if len(self.train_embeddings) < 1:
            raise ValueError("At least one normal embedding is required for kNN scoring.")
        self.k = min(self.k, len(self.train_embeddings))
        self._nn = NearestNeighbors(n_neighbors=self.k)
        self._nn.fit(self.train_embeddings)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        distances, _ = self._nn.kneighbors(embeddings)
        return distances.mean(axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, train_embeddings=self.train_embeddings, k=self.k)

    @classmethod
    def load(cls, path: str | Path) -> "KNNScorer":
        data = np.load(path)
        return cls(train_embeddings=data["train_embeddings"], k=int(data["k"]))


def build_scorer(method: str, embeddings: np.ndarray, regularization: float = 1e-3, k: int = 5):
    if method == "mahalanobis":
        return MahalanobisScorer.fit(embeddings, regularization=regularization)
    if method == "knn":
        return KNNScorer(train_embeddings=embeddings, k=k)
    raise ValueError(f"Unsupported anomaly scoring method: {method}")

