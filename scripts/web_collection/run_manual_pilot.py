"""Run the first controlled manual tyre-candidate pilot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.io import load_yaml
from src.web_collection.pilot import run_manual_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual TyreVisionX web-candidate pilot.")
    parser.add_argument("--config", default="configs/web_collection/web_collection.yaml")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--output_dir", default="data/interim/web_candidates/pilot_01")
    parser.add_argument("--dry-run", action="store_true", help="Parse inputs but do not copy/download images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    result = run_manual_pilot(args.input_csv, args.output_dir, config=config, dry_run=args.dry_run)
    print(result.to_dict())


if __name__ == "__main__":
    main()
