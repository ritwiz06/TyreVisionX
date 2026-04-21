"""Collect candidate image URLs into metadata records.

This script does discovery only. It does not download files and does not label
images as good/defect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.io import load_yaml, write_candidates_csv
from src.web_collection.providers import build_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect web-image candidate metadata.")
    parser.add_argument("--config", default="configs/web_collection/web_collection.yaml")
    parser.add_argument("--out", default=None, help="Override metadata output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    provider = build_provider(config)
    records = provider.collect()
    out_path = args.out or config["storage"]["metadata_csv_path"]
    write_candidates_csv(out_path, records)
    print(f"Collected {len(records)} candidate URL records.")
    print(f"Wrote metadata: {out_path}")


if __name__ == "__main__":
    main()
