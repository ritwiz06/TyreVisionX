"""Unified dataset wrapper for TyreVisionX manifests."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import cv2
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset


class TyreManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        split: Optional[str] = None,
        transforms: Optional[Callable] = None,
        root: Optional[Path] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.df = pd.read_csv(self.manifest_path)
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split]
        self.transforms = transforms
        self.root = Path(root) if root else None

        required_cols = {"image_path", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in manifest {self.manifest_path}: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute() and self.root is not None:
            path = self.root / path
        return path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row["image_path"])
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = int(row["label"])
        meta = {
            "image_path": str(img_path),
            "dataset_id": row.get("dataset_id", ""),
            "label_str": row.get("label_str", ""),
        }
        return image_tensor, label, meta


def load_combined_datasets(
    manifest_paths: Iterable[Path | str],
    split: Optional[str],
    transforms: Optional[Callable],
    roots: Optional[Dict[str, Path]] = None,
) -> Dataset:
    datasets: List[Dataset] = []
    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path)
        df = pd.read_csv(manifest_path)
        dataset_id = None
        if "dataset_id" in df.columns and len(df["dataset_id"].unique()) == 1:
            dataset_id = str(df["dataset_id"].iloc[0])
        root = None
        if roots and dataset_id and dataset_id in roots:
            root = roots[dataset_id]
        ds = TyreManifestDataset(manifest_path=manifest_path, split=split, transforms=transforms, root=root)
        datasets.append(ds)
    if not datasets:
        raise ValueError("No datasets loaded")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
