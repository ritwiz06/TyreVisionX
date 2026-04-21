"""Prepare a local Roboflow export for TyreVisionX manifest review.

This script does not download data. It scans an already-exported Roboflow
folder and writes a normalized review manifest that can be audited before any
merge into canonical TyreVisionX manifests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_registry(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"External dataset registry not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_dataset_entry(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    for entry in registry.get("datasets", []):
        if entry.get("id") == dataset_id:
            return entry
    raise KeyError(f"Dataset id not found in registry: {dataset_id}")


def normalize_label(label: str, mapping: dict[str, str]) -> str:
    if label in mapping:
        return mapping[label]
    lowered = label.lower()
    for source, target in mapping.items():
        if str(source).lower() == lowered:
            return str(target)
    return "unmapped"


def scan_classification_export(export_dir: str | Path, entry: dict[str, Any]) -> pd.DataFrame:
    export_dir = Path(export_dir)
    if not export_dir.exists():
        raise FileNotFoundError(f"Roboflow export directory not found: {export_dir}")
    mapping = entry.get("label_mapping", {})
    rows = []
    for split_dir in [p for p in export_dir.iterdir() if p.is_dir()]:
        split = split_dir.name
        for class_dir in [p for p in split_dir.iterdir() if p.is_dir()]:
            label = class_dir.name
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                normalized = normalize_label(label, mapping)
                rows.append(
                    {
                        "image_path": str(image_path),
                        "source_dataset_id": entry["id"],
                        "source_url": entry.get("source_url", ""),
                        "source_task_type": entry.get("task_type", ""),
                        "source_label": label,
                        "normalized_label": normalized,
                        "target": 0 if normalized == "normal" else 1 if normalized == "anomaly" else "",
                        "split": split,
                        "license": entry.get("visible_license", ""),
                        "license_status": entry.get("license_status", ""),
                        "import_status": "review_required",
                    }
                )
    return pd.DataFrame(rows)


def prepare_import_manifest(
    export_dir: str | Path,
    dataset_id: str,
    registry_path: str | Path,
    out_csv: str | Path,
) -> Path:
    registry = load_registry(registry_path)
    entry = get_dataset_entry(registry, dataset_id)
    if entry.get("task_type") != "classification":
        raise ValueError(
            f"Only classification Roboflow exports are supported for near-term manifest preparation. "
            f"{dataset_id} is task_type={entry.get('task_type')!r}."
        )
    df = scan_classification_export(export_dir, entry)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a review manifest from a local Roboflow export.")
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--registry", default="configs/data/external_dataset_registry.yaml")
    parser.add_argument("--out_csv", default="data/interim/external_roboflow_import_review.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = prepare_import_manifest(args.export_dir, args.dataset_id, args.registry, args.out_csv)
    print(f"Wrote review manifest: {out}")


if __name__ == "__main__":
    main()
