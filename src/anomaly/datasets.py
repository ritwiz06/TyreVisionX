"""Manifest-driven datasets for anomaly detection."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"image_path", "target", "is_normal"}


class AnomalyManifestDataset(Dataset):
    """Dataset for anomaly manifests.

    ``target`` is generic: ``0`` means normal and ``1`` means anomaly.
    """

    def __init__(self, manifest_path: str | Path, transforms: Callable | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Anomaly manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in anomaly manifest {self.manifest_path}: {sorted(missing)}")

        self.transforms = transforms
        self.repo_root = self._find_repo_root(self.manifest_path)

    @staticmethod
    def _find_repo_root(path: Path) -> Path:
        for parent in path.resolve().parents:
            if (parent / "data").exists() and (parent / "src").exists():
                return parent
        return Path.cwd()

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self._resolve_path(str(row["image_path"]))
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image_tensor = self.transforms(image=image)["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

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

