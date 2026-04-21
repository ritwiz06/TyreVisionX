from __future__ import annotations

from src.web_collection.confidence import assign_review_priority


def test_confidence_bucket_reject_before_review() -> None:
    bucket = assign_review_priority({"quality_status": "rejected", "review_notes": "duplicate_exact_hash"})
    assert bucket == "reject_before_review"


def test_confidence_bucket_highest_priority_for_likely_anomalous() -> None:
    bucket = assign_review_priority({"quality_status": "kept", "anomaly_triage_bucket": "likely_anomalous"})
    assert bucket == "highest_priority_review"


def test_confidence_bucket_normal_candidate_is_not_label() -> None:
    bucket = assign_review_priority(
        {"quality_status": "kept", "anomaly_triage_bucket": "likely_normal", "query_priority": 1}
    )
    assert bucket == "normal_candidate_review"


def test_confidence_bucket_uncertain_default() -> None:
    bucket = assign_review_priority({"quality_status": "kept", "anomaly_triage_bucket": "not_scored"})
    assert bucket == "uncertain_review"
