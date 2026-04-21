"""Import a manually approved Google-discovery CSV.

This script does not scrape Google. The input CSV must be prepared by a
researcher from browser-visible results that are approved for research review.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.io import write_candidates_csv
from src.web_collection.providers_manual_google import ManualGoogleDiscoveryProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import manual Google-discovery candidate rows.")
    parser.add_argument(
        "--input_csv",
        default="data/external/manual_candidate_urls/manual_google_discovery_template.csv",
        help="Researcher-prepared CSV. Do not point this at scraped Google HTML output.",
    )
    parser.add_argument(
        "--out_csv",
        default="data/interim/web_candidates/manual_google_candidates.csv",
        help="Output candidate metadata CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = ManualGoogleDiscoveryProvider(args.input_csv)
    records = provider.collect()
    write_candidates_csv(args.out_csv, records)
    print(f"Imported {len(records)} manually approved Google-discovery candidates.")
    print(f"Wrote candidate metadata: {args.out_csv}")
    print("Reminder: imported candidates are not labels and still require filtering and human review.")


if __name__ == "__main__":
    main()
