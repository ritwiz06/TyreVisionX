"""Manual pilot orchestration for tyre web-candidate ingestion."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.web_collection.filters import apply_quality_and_dedupe_filters
from src.web_collection.io import read_candidates_csv, read_manual_url_file, write_candidates_csv
from src.web_collection.providers import ManualURLProvider
from src.web_collection.review import build_review_queue
from src.web_collection.schemas import CANDIDATE_COLUMNS, REVIEW_STATUSES, CandidateRecord


PILOT_DIR = Path("data/interim/web_candidates/pilot_01")
PILOT_REPORT = Path("reports/web_collection/pilot_run_report.md")
REVIEW_QUEUE_REPORT = Path("reports/web_collection/pilot_review_queue.csv")
REVIEW_DECISIONS_TEMPLATE = Path("reports/web_collection/pilot_review_decisions_template.csv")
CURATED_MANIFEST = Path("data/manifests/web_curated_tyres_likely_normal_v1.csv")


@dataclass
class PilotRunResult:
    status: str
    input_csv: str | None
    output_dir: str
    counts: dict[str, int]
    pending: list[str]
    outputs: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "input_csv": self.input_csv,
            "output_dir": self.output_dir,
            "counts": self.counts,
            "pending": self.pending,
            "outputs": self.outputs,
        }


def find_approved_input(input_dir: str | Path = "data/external/manual_candidate_urls") -> Path | None:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return None
    candidates = sorted(
        path
        for path in input_dir.glob("*.csv")
        if "template" not in path.name.lower() and "decision" not in path.name.lower()
    )
    return candidates[0] if candidates else None


def validate_manual_input(path: str | Path) -> pd.DataFrame:
    rows = read_manual_url_file(path)
    df = pd.DataFrame(rows)
    required_any = {"source_url", "url", "local_source_path"}
    if not any(column in df.columns for column in required_any):
        raise ValueError("Manual pilot CSV must include source_url/url or local_source_path.")
    if len(df) > 50:
        raise ValueError(f"Manual pilot is capped at 50 candidates for this first run; found {len(df)} rows.")
    if len(df) < 1:
        raise ValueError("Manual pilot CSV contains no rows.")
    return df


def collect_manual_candidates(input_csv: str | Path, product_type: str = "tyre") -> pd.DataFrame:
    validate_manual_input(input_csv)
    records = ManualURLProvider(input_csv, product_type=product_type).collect()
    return pd.DataFrame([record.to_dict() for record in records], columns=CANDIDATE_COLUMNS)


def copy_or_download_candidates(
    metadata: pd.DataFrame,
    raw_dir: str | Path,
    timeout_seconds: int = 20,
) -> pd.DataFrame:
    from scripts.web_collection.download_candidates import _copy_or_download, _extension_from_url

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = metadata.copy()
    for idx, row in df.iterrows():
        source_url = str(row.get("source_url") or "")
        local_source_path = str(row.get("local_source_path") or "")
        if not source_url and local_source_path:
            source_url = Path(local_source_path).expanduser().resolve().as_uri()
        out_path = raw_dir / f"{row['candidate_id']}{_extension_from_url(source_url)}"
        try:
            _copy_or_download(source_url, out_path, timeout=timeout_seconds)
            df.at[idx, "local_raw_path"] = str(out_path)
            df.at[idx, "download_status"] = "downloaded"
        except Exception as exc:
            df.at[idx, "download_status"] = "failed"
            note = str(row.get("review_notes") or "")
            df.at[idx, "review_notes"] = ";".join(
                part for part in [note, f"download_failed:{type(exc).__name__}:{exc}"] if part
            )
    return df


def write_pilot_report(result: PilotRunResult, path: str | Path = PILOT_REPORT) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Pilot Run Report",
        "",
        f"Status: `{result.status}`",
        "",
        f"Input CSV: `{result.input_csv or 'not_found'}`",
        f"Output directory: `{result.output_dir}`",
        "",
        "## Counts",
        "",
        "| Stage | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in result.counts.items())
    lines.extend(["", "## Outputs", "", "| Output | Path |", "|---|---|"])
    lines.extend(f"| {key} | `{value}` |" for key, value in result.outputs.items())
    lines.extend(["", "## Pending", ""])
    lines.extend(f"- {item}" for item in result.pending)
    lines.extend(
        [
            "",
            "## Integrity Notes",
            "- No candidate is auto-labeled as good.",
            "- Human review is required before promotion to a likely-normal manifest.",
            "- Anomaly triage is advisory only when available.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_manual_pilot(
    input_csv: str | Path | None = None,
    output_dir: str | Path = PILOT_DIR,
    config: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> PilotRunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or {}
    storage_cfg = config.get("storage", {})
    download_cfg = config.get("download", {})
    raw_dir = output_dir / "raw"
    input_path = Path(input_csv) if input_csv else find_approved_input()
    is_default_pilot_dir = output_dir == PILOT_DIR
    report_path = PILOT_REPORT if is_default_pilot_dir else output_dir / "pilot_run_report.md"
    review_queue_report = REVIEW_QUEUE_REPORT if is_default_pilot_dir else output_dir / "pilot_review_queue.csv"
    review_decisions_template = (
        REVIEW_DECISIONS_TEMPLATE if is_default_pilot_dir else output_dir / "pilot_review_decisions_template.csv"
    )

    outputs = {
        "candidate_metadata": str(output_dir / "candidate_metadata.csv"),
        "downloaded_metadata": str(output_dir / "candidate_metadata_downloaded.csv"),
        "kept": str(output_dir / "candidates_kept.csv"),
        "rejected": str(output_dir / "candidates_rejected.csv"),
        "review_needed": str(output_dir / "candidates_review_needed.csv"),
        "review_queue": str(output_dir / "review_queue.csv"),
        "filter_summary": str(output_dir / "filter_summary.json"),
        "pilot_run_report": str(report_path),
    }

    if input_path is None or not input_path.exists():
        result = PilotRunResult(
            status="blocked_missing_approved_input",
            input_csv=None,
            output_dir=str(output_dir),
            counts={"candidates": 0, "downloaded": 0, "kept": 0, "review_needed": 0, "rejected": 0},
            pending=[
                "Provide a reviewed/approved manual pilot CSV under data/external/manual_candidate_urls/.",
                "Use approved_tyres_pilot_urls_template.csv as the schema.",
            ],
            outputs=outputs,
        )
        write_pilot_report(result, report_path)
        (output_dir / "pilot_status.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    metadata = collect_manual_candidates(input_path, product_type=config.get("product", {}).get("type", "tyre"))
    write_candidates_csv(outputs["candidate_metadata"], metadata)
    if dry_run:
        downloaded = metadata.copy()
        downloaded["download_status"] = "dry_run_not_downloaded"
    else:
        downloaded = copy_or_download_candidates(
            metadata,
            raw_dir=config.get("pilot", {}).get("raw_image_dir", raw_dir),
            timeout_seconds=int(download_cfg.get("timeout_seconds", 20)),
        )
    write_candidates_csv(outputs["downloaded_metadata"], downloaded)

    filter_result = apply_quality_and_dedupe_filters(downloaded, config)
    write_candidates_csv(outputs["kept"], filter_result.kept)
    write_candidates_csv(outputs["rejected"], filter_result.rejected)
    write_candidates_csv(outputs["review_needed"], filter_result.review_needed)
    Path(outputs["filter_summary"]).write_text(json.dumps(filter_result.summary, indent=2), encoding="utf-8")

    queue = build_review_queue(pd.concat([filter_result.kept, filter_result.review_needed], ignore_index=True))
    queue.to_csv(outputs["review_queue"], index=False)
    review_queue_report.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(review_queue_report, index=False)
    review_decisions_template.parent.mkdir(parents=True, exist_ok=True)
    queue.assign(human_review_status="pending_review", reviewer="", reviewed_at="", decision_notes="").to_csv(
        review_decisions_template,
        index=False,
    )

    result = PilotRunResult(
        status="pilot_executed_pending_human_review" if len(queue) else "pilot_executed_no_reviewable_candidates",
        input_csv=str(input_path),
        output_dir=str(output_dir),
        counts={
            "candidates": int(len(metadata)),
            "downloaded": int((downloaded["download_status"] == "downloaded").sum()),
            "kept": int(len(filter_result.kept)),
            "review_needed": int(len(filter_result.review_needed)),
            "rejected": int(len(filter_result.rejected)),
            "review_queue": int(len(queue)),
        },
        pending=[
            "Human review decisions are pending.",
            "Anomaly triage remains optional and advisory.",
        ],
        outputs=outputs,
    )
    write_pilot_report(result, report_path)
    (output_dir / "pilot_status.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    write_candidate_summary(result, output_dir / "candidate_summary.md")
    return result


def write_candidate_summary(result: PilotRunResult, path: str | Path) -> None:
    path = Path(path)
    path.write_text(
        "# Pilot Candidate Summary\n\n"
        f"Status: `{result.status}`\n\n"
        "| Stage | Count |\n"
        "|---|---:|\n"
        + "\n".join(f"| {key} | {value} |" for key, value in result.counts.items())
        + "\n",
        encoding="utf-8",
    )


def promote_reviewed_candidates(
    decisions_csv: str | Path,
    output_manifest: str | Path = CURATED_MANIFEST,
) -> dict[str, Any]:
    decisions_csv = Path(decisions_csv)
    if not decisions_csv.exists():
        raise FileNotFoundError(f"Missing review decisions CSV: {decisions_csv}")
    df = pd.read_csv(decisions_csv).fillna("")
    if "human_review_status" not in df.columns:
        raise ValueError("Review decisions CSV must contain human_review_status.")
    invalid = sorted(set(df["human_review_status"]) - set(REVIEW_STATUSES))
    if invalid:
        raise ValueError(f"Invalid review statuses: {invalid}")
    approved = df[df["human_review_status"] == "approved_likely_normal"].copy()
    manifest = pd.DataFrame(
        {
            "image_path": approved["local_raw_path"],
            "target": 0,
            "label": 0,
            "label_str": "likely_normal_web_reviewed",
            "split": "curated_pool",
            "is_normal": True,
            "source_dataset": "web_curated_pilot_01",
            "product_type": approved.get("product_type", "tyre"),
            "candidate_id": approved["candidate_id"],
            "source_url": approved["source_url"],
            "page_url": approved.get("page_url", ""),
            "human_review_status": approved["human_review_status"],
        }
    )
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    return {
        "decisions_csv": str(decisions_csv),
        "output_manifest": str(output_manifest),
        "approved_count": int(len(manifest)),
        "total_decisions": int(len(df)),
    }


def copy_review_pack_images(queue: pd.DataFrame, pack_dir: str | Path) -> pd.DataFrame:
    pack_dir = Path(pack_dir)
    images_dir = pack_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in queue.iterrows():
        local_path = Path(str(row.get("local_raw_path") or ""))
        copied_path = ""
        if local_path.exists():
            suffix = local_path.suffix or ".jpg"
            dest = images_dir / f"{row['candidate_id']}{suffix}"
            shutil.copy2(local_path, dest)
            copied_path = str(dest)
        copied = dict(row)
        copied["review_pack_image"] = copied_path
        rows.append(copied)
    return pd.DataFrame(rows)
