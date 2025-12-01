"""Scan datasets and create manifests for TyreVisionX."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_data_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_images(root: Path, dataset_id: str, label_map: Dict[str, int]) -> List[Dict]:
    rows = []
    for label_str, label_int in label_map.items():
        label_dir = root / label_str
        if not label_dir.exists():
            continue
        for img_path in label_dir.rglob("*"):
            if img_path.suffix.lower() not in EXTS:
                continue
            rows.append(
                {
                    "image_path": str(img_path.relative_to(root)),
                    "label_str": label_str,
                    "label": label_int,
                    "dataset_id": dataset_id,
                    "split": "",
                }
            )
    return rows


def save_manifest(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


def main(config_path: str):
    cfg = load_data_config(Path(config_path))
    paths = cfg.get("paths", {})
    label_map = cfg.get("label_mapping", {"good": 0, "defect": 1})
    for dataset_id, info in paths.items():
        root = Path(info["root"])
        manifest_out = Path(info["manifest"])
        rows = collect_images(root, dataset_id, label_map)
        save_manifest(rows, manifest_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml", help="Data config path")
    args = parser.parse_args()
    main(args.config)
