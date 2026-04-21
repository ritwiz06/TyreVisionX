"""Build a human-review queue from filtered web candidates."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.io import load_yaml, read_candidates_csv
from src.web_collection.review import build_review_queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a human review queue CSV.")
    parser.add_argument("--config", default="configs/web_collection/web_collection.yaml")
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = Path(config["storage"]["curated_output_dir"])
    inputs = args.inputs or [
        str(output_dir / "candidates_kept.csv"),
        str(output_dir / "candidates_review_needed.csv"),
    ]
    frames = [read_candidates_csv(path) for path in inputs if Path(path).exists()]
    if not frames:
        raise FileNotFoundError("No kept/review-needed candidate CSVs found. Run filter_candidates.py first.")
    queue = build_review_queue(pd.concat(frames, ignore_index=True))
    out_path = Path(args.out or config["review"]["queue_csv_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(out_path, index=False)
    print(f"Wrote review queue with {len(queue)} rows: {out_path}")


if __name__ == "__main__":
    main()
