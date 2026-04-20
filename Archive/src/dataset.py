"""Legacy baseline dataset utilities kept as a compatibility import path."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_manifest_df(path: str) -> pd.DataFrame:
    """Load manifest CSV into a DataFrame."""
    return pd.read_csv(path)


def class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Return class distribution based on label_str or label."""
    if "label_str" in df.columns:
        return df["label_str"].value_counts().to_dict()
    if "label" in df.columns:
        return df["label"].value_counts().to_dict()
    raise ValueError("Manifest missing label_str or label columns")


class TyreManifestDataset(Dataset):
    """Dataset backed by a manifest CSV with split filtering."""

    def __init__(self, manifest_csv: str, split: str, transforms: Callable | None = None) -> None:
        self.manifest_csv = manifest_csv
        self.split = split
        self.transforms = transforms
        self._manifest_path = Path(manifest_csv).resolve()
        self._repo_root = self._find_repo_root(self._manifest_path)

        self.df = load_manifest_df(manifest_csv)
        if "split" not in self.df.columns:
            raise ValueError("Manifest missing 'split' column")
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        required_cols = {"image_path", "label", "dataset_id"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    @staticmethod
    def _find_repo_root(manifest_path: Path) -> Path:
        for parent in manifest_path.parents:
            if (parent / "data").exists():
                return parent
        return Path.cwd()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            candidate = self._repo_root / img_path
            if candidate.exists():
                img_path = candidate
            else:
                img_path = Path.cwd() / img_path

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = int(row["label"])
        meta = {
            "image_path": str(img_path),
            "dataset_id": row["dataset_id"],
            "split": row["split"],
        }
        return image_tensor, label, meta
