"""Create stratified train/val/test splits for manifests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.split import stratified_split  # noqa: E402


def load_data_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_manifest(manifest_path: Path, ratios: Dict, seed: int) -> None:
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        print(f"Skipping {manifest_path} (missing or empty)")
        return
    try:
        df = pd.read_csv(manifest_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping {manifest_path} (no data)")
        return
    if "label" not in df.columns:
        raise ValueError(f"Manifest {manifest_path} missing 'label' column")
    split_df = stratified_split(df, train_ratio=ratios.get("train", 0.7), val_ratio=ratios.get("val", 0.15), test_ratio=ratios.get("test", 0.15), seed=seed)
    split_df.to_csv(manifest_path, index=False)
    print(f"Updated splits in {manifest_path} (n={len(split_df)})")


def main(config_path: str, seed: int):
    cfg = load_data_config(Path(config_path))
    ratios = cfg.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    for _, info in cfg.get("paths", {}).items():
        manifest_path = Path(info["manifest"])
        process_manifest(manifest_path, ratios, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/datasets.yaml", help="Data config path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args.config, args.seed)
