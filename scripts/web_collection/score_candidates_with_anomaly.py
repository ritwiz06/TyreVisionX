"""Optional anomaly-triage hook for web candidates.

This script does not label images. It only records pending/ranking status for
human review. Full candidate embedding scoring can be added after the anomaly
baseline artifacts are available locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.io import load_yaml, read_candidates_csv, write_candidates_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach anomaly triage status to web candidates.")
    parser.add_argument("--config", default="configs/web_collection/web_collection.yaml")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    triage_cfg = config.get("anomaly_triage", {})
    metadata_path = args.metadata or config["storage"]["metadata_csv_path"]
    out_path = args.out or metadata_path
    status_path = Path("reports/web_collection/anomaly_triage_status.md")

    df = read_candidates_csv(metadata_path)
    artifact_dir = Path(str(triage_cfg.get("artifact_dir") or ""))
    metadata_json = artifact_dir / "metadata.json" if artifact_dir else Path("")
    threshold_json = artifact_dir / "threshold.json" if artifact_dir else Path("")

    if not triage_cfg.get("enabled", False):
        df["anomaly_triage_bucket"] = "not_scored"
        reason = "Anomaly triage is disabled in config."
    elif not artifact_dir or not metadata_json.exists() or not threshold_json.exists():
        df["anomaly_triage_bucket"] = "pending_artifact"
        reason = (
            "Anomaly triage pending: fitted anomaly artifacts are missing. "
            "Expected metadata.json and threshold.json under the configured artifact_dir."
        )
    else:
        # Future extension point: load scorer + feature extractor and score candidates.
        df["anomaly_triage_bucket"] = "pending_scoring_implementation"
        reason = (
            "Anomaly artifacts exist, but web-candidate scoring is intentionally left as a small future extension. "
            "Scores must be used only for review prioritization, not labels."
        )

    write_candidates_csv(out_path, df)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_payload = {
        "metadata_path": str(metadata_path),
        "output_path": str(out_path),
        "artifact_dir": str(artifact_dir),
        "reason": reason,
        "rows": int(len(df)),
    }
    status_path.write_text(
        "# Anomaly Triage Status\n\n"
        "Status: pending unless fitted anomaly artifacts and candidate scoring are available.\n\n"
        "Anomaly scores are triage signals for human review, not ground-truth labels.\n\n"
        "```json\n"
        f"{json.dumps(status_payload, indent=2)}\n"
        "```\n",
        encoding="utf-8",
    )
    print(reason)
    print(f"Wrote metadata: {out_path}")
    print(f"Wrote status: {status_path}")


if __name__ == "__main__":
    main()
