"""Schemas for web-image candidate metadata and query definitions."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


CANDIDATE_COLUMNS = [
    "candidate_id",
    "provider",
    "source_provider",
    "query_id",
    "source_query_id",
    "query_family",
    "query_text",
    "source_url",
    "page_url",
    "license_name",
    "license_url",
    "attribution_text",
    "local_source_path",
    "retrieval_timestamp",
    "local_raw_path",
    "download_status",
    "content_hash",
    "perceptual_hash",
    "width",
    "height",
    "blur_score",
    "file_size_bytes",
    "mime_type",
    "dedupe_group",
    "quality_status",
    "anomaly_score",
    "anomaly_triage_bucket",
    "human_review_status",
    "review_notes",
    "product_type",
    "view_type",
    "source_dataset",
    "notes",
]

REVIEW_STATUSES = [
    "pending_review",
    "approved_likely_normal",
    "rejected_irrelevant",
    "rejected_low_quality",
    "rejected_likely_defect",
    "uncertain_holdout",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_candidate_id(source_provider: str, source_url: str, source_query_id: str = "") -> str:
    raw = f"{source_provider}|{source_query_id}|{source_url}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class QuerySpec:
    """One editable query specification for candidate image collection."""

    query_id: str
    query_text: str
    product_family: str = "tyre"
    domain: str = "manufacturing_inspection"
    view_type: str = ""
    condition_adjective: str = "undamaged"
    environment_context: str = ""
    positive_keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    language: str = "en"
    priority: int = 3
    intended_use: str = "candidate_likely_normal_review"
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuerySpec":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateRecord:
    """Metadata for a web-image candidate.

    A candidate is not a label. The record tracks provenance, quality, dedupe,
    anomaly-triage, and human-review fields so the image can be evaluated later.
    """

    candidate_id: str
    provider: str
    source_provider: str
    query_id: str
    source_query_id: str
    query_family: str
    query_text: str
    source_url: str
    page_url: str = ""
    license_name: str = ""
    license_url: str = ""
    attribution_text: str = ""
    local_source_path: str = ""
    retrieval_timestamp: str = field(default_factory=utc_now_iso)
    local_raw_path: str = ""
    download_status: str = "not_downloaded"
    content_hash: str = ""
    perceptual_hash: str = ""
    width: int | None = None
    height: int | None = None
    blur_score: float | None = None
    file_size_bytes: int | None = None
    mime_type: str = ""
    dedupe_group: str = ""
    quality_status: str = "pending"
    anomaly_score: float | None = None
    anomaly_triage_bucket: str = "not_scored"
    human_review_status: str = "pending_review"
    review_notes: str = ""
    product_type: str = "tyre"
    view_type: str = ""
    source_dataset: str = "web_candidate"
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateRecord":
        values = {column: payload.get(column, "") for column in CANDIDATE_COLUMNS}
        if not values["source_provider"] and values["provider"]:
            values["source_provider"] = values["provider"]
        if not values["provider"] and values["source_provider"]:
            values["provider"] = values["source_provider"]
        if not values["source_query_id"] and values["query_id"]:
            values["source_query_id"] = values["query_id"]
        if not values["query_id"] and values["source_query_id"]:
            values["query_id"] = values["source_query_id"]
        if not values["candidate_id"]:
            stable_source = str(values["source_url"] or values.get("local_source_path") or "")
            values["candidate_id"] = stable_candidate_id(
                str(values["source_provider"]),
                stable_source,
                str(values["source_query_id"]),
            )
        for key in ["width", "height", "file_size_bytes"]:
            values[key] = int(values[key]) if str(values[key]).strip() not in {"", "nan", "None"} else None
        values["blur_score"] = (
            float(values["blur_score"]) if str(values["blur_score"]).strip() not in {"", "nan", "None"} else None
        )
        values["anomaly_score"] = (
            float(values["anomaly_score"])
            if str(values["anomaly_score"]).strip() not in {"", "nan", "None"}
            else None
        )
        return cls(**values)

    @classmethod
    def from_url(
        cls,
        source_url: str,
        source_provider: str = "manual",
        provider: str = "",
        source_query_id: str = "",
        query_id: str = "",
        query_family: str = "",
        query_text: str = "",
        page_url: str = "",
        license_name: str = "",
        license_url: str = "",
        attribution_text: str = "",
        local_source_path: str = "",
        product_type: str = "tyre",
        view_type: str = "",
        notes: str = "",
    ) -> "CandidateRecord":
        provider = provider or source_provider
        source_provider = source_provider or provider
        query_id = query_id or source_query_id
        source_query_id = source_query_id or query_id
        stable_source = source_url or local_source_path
        return cls(
            candidate_id=stable_candidate_id(source_provider, stable_source, source_query_id),
            provider=provider,
            source_provider=source_provider,
            query_id=query_id,
            source_query_id=source_query_id,
            query_family=query_family,
            query_text=query_text,
            source_url=source_url,
            page_url=page_url,
            license_name=license_name,
            license_url=license_url,
            attribution_text=attribution_text,
            local_source_path=local_source_path,
            product_type=product_type,
            view_type=view_type,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {column: payload.get(column, "") for column in CANDIDATE_COLUMNS}
