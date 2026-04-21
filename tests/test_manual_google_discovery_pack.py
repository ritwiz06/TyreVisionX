from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.web_collection.generate_manual_google_discovery_pack import (
    NEGATIVE_KEYWORDS,
    build_manual_google_queries,
    generate_pack,
)


REQUIRED_PREFILL_COLUMNS = {
    "candidate_id",
    "query_id",
    "query_text",
    "source_url",
    "page_url",
    "local_source_path",
    "product_type",
    "notes",
}


def test_query_batch_generation_counts_and_groups() -> None:
    queries = build_manual_google_queries()
    assert len(queries) == 40
    groups = {query.group for query in queries}
    assert groups == {"tread", "sidewall", "mounted_tyre", "inspection_like", "industrial_off_highway"}
    assert all("-\"damage\"" in query.browser_query for query in queries)
    assert all(keyword in NEGATIVE_KEYWORDS for keyword in ["damage", "defect", "cracked"])


def test_discovery_pack_csv_creation(tmp_path: Path) -> None:
    outputs = generate_pack(tmp_path)
    csv_path = outputs["prefilled_csv"]
    assert isinstance(csv_path, Path)
    df = pd.read_csv(csv_path).fillna("")
    assert len(df) == 40
    assert set(df.columns) >= REQUIRED_PREFILL_COLUMNS
    assert set(df["product_type"]) == {"tyre"}
    assert (df["source_url"] == "").all()
    assert (df["page_url"] == "").all()
    assert (df["local_source_path"] == "").all()
    assert df["candidate_id"].is_unique


def test_discovery_pack_markdown_outputs(tmp_path: Path) -> None:
    outputs = generate_pack(tmp_path)
    query_batches = outputs["query_batches"]
    checklist = outputs["checklist"]
    assert isinstance(query_batches, Path)
    assert isinstance(checklist, Path)
    query_text = query_batches.read_text(encoding="utf-8")
    checklist_text = checklist.read_text(encoding="utf-8")
    assert "Total queries: `40`" in query_text
    assert "No scraping" in query_text
    assert "Do not auto-label" in checklist_text or "not a label" in checklist_text
