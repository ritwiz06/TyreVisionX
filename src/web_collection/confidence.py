"""Review-priority confidence buckets for web candidates.

These utilities rank candidates for human review. They do not assign labels.
"""
from __future__ import annotations

from typing import Any


def assign_review_priority(candidate: dict[str, Any]) -> str:
    """Assign a review-priority bucket without creating a ground-truth label."""

    quality_status = str(candidate.get("quality_status", "pending"))
    review_notes = str(candidate.get("review_notes", "")).lower()
    anomaly_bucket = str(candidate.get("anomaly_triage_bucket", "not_scored"))
    query_priority = _safe_int(candidate.get("query_priority", candidate.get("priority", 3)), default=3)

    if quality_status == "rejected" or "duplicate_exact_hash" in review_notes or "not_downloaded" in review_notes:
        return "reject_before_review"
    if anomaly_bucket == "likely_anomalous":
        return "highest_priority_review"
    if quality_status == "review_needed" or anomaly_bucket in {"uncertain", "pending_artifact", "not_scored"}:
        return "uncertain_review"
    if quality_status == "kept" and anomaly_bucket == "likely_normal" and query_priority <= 2:
        return "normal_candidate_review"
    if quality_status == "kept":
        return "uncertain_review"
    return "uncertain_review"


def add_review_priority(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        row = dict(record)
        row["review_priority_bucket"] = assign_review_priority(row)
        output.append(row)
    return output


def _safe_int(value: Any, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
