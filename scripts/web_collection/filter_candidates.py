"""Apply deduplication and quality filters to candidate images."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.filters import apply_quality_and_dedupe_filters
from src.web_collection.io import load_yaml, read_candidates_csv, write_candidates_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter downloaded web-image candidates.")
    parser.add_argument("--config", default="configs/web_collection/web_collection.yaml")
    parser.add_argument("--metadata", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    metadata_path = args.metadata or config["storage"]["metadata_csv_path"]
    output_dir = Path(config["storage"]["curated_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    result = apply_quality_and_dedupe_filters(read_candidates_csv(metadata_path), config)
    write_candidates_csv(output_dir / "candidates_kept.csv", result.kept)
    write_candidates_csv(output_dir / "candidates_rejected.csv", result.rejected)
    write_candidates_csv(output_dir / "candidates_review_needed.csv", result.review_needed)
    (output_dir / "filter_summary.json").write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    print(result.summary)
    print(f"Wrote filtered outputs under: {output_dir}")


if __name__ == "__main__":
    main()
