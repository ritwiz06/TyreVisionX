"""Human-review queue helpers for web-candidate images."""
from __future__ import annotations

import pandas as pd

from src.web_collection.schemas import REVIEW_STATUSES


REVIEW_COLUMNS = [
    "candidate_id",
    "local_raw_path",
    "source_url",
    "page_url",
    "local_source_path",
    "query_text",
    "quality_status",
    "anomaly_score",
    "anomaly_triage_bucket",
    "human_review_status",
    "review_notes",
    "product_type",
]


def build_review_queue(df: pd.DataFrame, include_quality_statuses: set[str] | None = None) -> pd.DataFrame:
    """Create a human-review CSV skeleton from candidate metadata."""

    include_quality_statuses = include_quality_statuses or {"kept", "review_needed"}
    queue = df[df["quality_status"].isin(include_quality_statuses)].copy()
    if "human_review_status" not in queue.columns:
        queue["human_review_status"] = "pending_review"
    queue.loc[~queue["human_review_status"].isin(REVIEW_STATUSES), "human_review_status"] = "pending_review"
    for column in REVIEW_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    return queue[REVIEW_COLUMNS].sort_values(["anomaly_triage_bucket", "quality_status", "candidate_id"])
