"""Small PatchCore-style helpers for TyreVisionX.

This is intentionally lightweight: it provides deterministic memory-bank
reduction and patch nearest-neighbor scoring without depending on external
anomaly-detection frameworks.
"""
from __future__ import annotations

import numpy as np

from src.anomaly.patch_memory import PatchMemoryScorer, build_memory_bank


def fit_patchcore_lite(
    normal_patch_embeddings: np.ndarray,
    max_memory_patches: int = 10000,
    seed: int = 42,
    k: int = 1,
) -> PatchMemoryScorer:
    """Fit a reduced patch-memory scorer."""

    memory = build_memory_bank(normal_patch_embeddings, max_memory_patches=max_memory_patches, seed=seed)
    return PatchMemoryScorer(memory_bank=memory, k=k)
