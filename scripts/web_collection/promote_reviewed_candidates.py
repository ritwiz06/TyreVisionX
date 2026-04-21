"""Promote human-approved likely-normal candidates into a curated manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.pilot import promote_reviewed_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote reviewed candidates to curated likely-normal manifest.")
    parser.add_argument("--decisions_csv", default="reports/web_collection/pilot_review_decisions_template.csv")
    parser.add_argument("--out", default="data/manifests/web_curated_tyres_likely_normal_v1.csv")
    parser.add_argument("--status", default="reports/web_collection/pilot_promotion_status.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status_path = Path(args.status)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = promote_reviewed_candidates(args.decisions_csv, args.out)
        status = "promoted" if result["approved_count"] else "no_approved_candidates"
        status_path.write_text(
            "# Pilot Promotion Status\n\n"
            f"Status: `{status}`\n\n"
            "```json\n"
            f"{json.dumps(result, indent=2)}\n"
            "```\n",
            encoding="utf-8",
        )
        print(result)
    except Exception as exc:
        payload = {"status": "blocked", "error": str(exc), "error_type": type(exc).__name__}
        status_path.write_text(
            "# Pilot Promotion Status\n\n"
            "Status: `blocked`\n\n"
            "```json\n"
            f"{json.dumps(payload, indent=2)}\n"
            "```\n",
            encoding="utf-8",
        )
        print(payload)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
