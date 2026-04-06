"""Canonical dataset loading utilities for TyreVisionX."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import pandas as pd
import torch
import yaml
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
        self.repo_root = self._find_repo_root(self.manifest_path)
        self.default_roots = _load_default_dataset_roots()

        required_cols = {"image_path", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in manifest {self.manifest_path}: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _find_repo_root(manifest_path: Path) -> Path:
        for parent in manifest_path.parents:
            if (parent / "data").exists():
                return parent
        return Path.cwd()

    def _resolve_path(self, path_str: str, dataset_id: str = "") -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path

        if self.root is not None:
            return self.root / path

        repo_candidate = self.repo_root / path
        if repo_candidate.exists():
            return repo_candidate

        dataset_root = self.default_roots.get(dataset_id)
        if dataset_root is not None:
            candidate = dataset_root / path
            if candidate.exists():
                return candidate
            return candidate

        return path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        dataset_id = str(row.get("dataset_id", ""))
        img_path = self._resolve_path(row["image_path"], dataset_id=dataset_id)
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
            "dataset_id": dataset_id,
            "label_str": row.get("label_str", ""),
            "split": row.get("split", ""),
        }
        return image_tensor, label, meta


def load_data_config(config_path: Path | str) -> Dict:
    """Load a dataset configuration YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_default_dataset_roots() -> Dict[str, Path]:
    for config_path in (Path("configs/data/datasets.yaml"), Path("configs/data.yaml")):
        if not config_path.exists():
            continue
        cfg = load_data_config(config_path)
        roots = {}
        for dataset_id, dataset_cfg in cfg.get("paths", {}).items():
            roots[str(dataset_id)] = Path(dataset_cfg["root"])
        return roots
    return {}


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


def load_dataset_from_runtime_config(
    data_cfg: Dict,
    split: Optional[str],
    transforms: Optional[Callable],
) -> Tuple[Dataset, List[str]]:
    """Build a dataset from either a direct manifest or a dataset config file.

    The config-driven path is canonical. ``manifest_csv`` remains supported as a
    compatibility fallback for older experiments and legacy scripts.
    """

    manifest_csv = data_cfg.get("manifest_csv")
    if manifest_csv:
        dataset = TyreManifestDataset(manifest_path=manifest_csv, split=split, transforms=transforms)
        return dataset, [manifest_csv]

    config_file = data_cfg.get("config_file")
    if not config_file:
        raise ValueError("Expected either data.manifest_csv or data.config_file")

    datasets_cfg = load_data_config(Path(config_file))
    selected = data_cfg.get("use_datasets", datasets_cfg.get("use_datasets", []))
    manifests: List[str] = []
    roots: Dict[str, Path] = {}
    for dataset_id in selected:
        dataset_cfg = datasets_cfg["paths"].get(dataset_id)
        if not dataset_cfg:
            raise ValueError(f"Dataset {dataset_id} not found in data config {config_file}")
        manifests.append(dataset_cfg["manifest"])
        roots[str(dataset_id)] = Path(dataset_cfg["root"])

    dataset = load_combined_datasets(manifests, split=split, transforms=transforms, roots=roots)
    return dataset, manifests
