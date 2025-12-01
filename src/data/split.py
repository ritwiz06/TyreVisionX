"""Helpers for stratified splits and folds."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def stratified_split(
    manifest: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    df = manifest.copy()
    y = df["label"]

    if test_ratio > 0:
        train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=y, random_state=seed)
    else:
        train_df, test_df = df, pd.DataFrame()

    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df["label"], random_state=seed)

    train_df["split"] = "train"
    val_df["split"] = "val"
    if not test_df.empty:
        test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def create_folds(manifest: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = manifest.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = []
    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        fold_indices.extend([(int(i), fold) for i in val_idx])
    fold_df = pd.DataFrame(fold_indices, columns=["index", "fold"])
    df = df.reset_index().merge(fold_df, on="index").drop(columns=["index"])
    return df


def save_split(manifest_with_split: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_with_split.to_csv(path, index=False)
    return path
