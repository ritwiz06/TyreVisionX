from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.web_collection.generate_query_catalog import build_default_queries
from scripts.web_collection.export_review_pack import export_review_pack
from src.web_collection.filters import apply_quality_and_dedupe_filters
from src.web_collection.io import read_candidates_csv, write_candidates_csv
from src.web_collection.pilot import collect_manual_candidates, promote_reviewed_candidates, run_manual_pilot
from src.web_collection.providers import ManualURLProvider
from src.web_collection.review import build_review_queue
from src.web_collection.schemas import CandidateRecord


def test_query_catalog_generation_smoke() -> None:
    queries = build_default_queries()
    assert len(queries) >= 20
    assert any("sidewall" in query.query_text for query in queries)
    assert all(query.negative_keywords for query in queries)


def test_candidate_record_serialization() -> None:
    record = CandidateRecord.from_url(
        source_url="https://example.com/tyre.jpg",
        source_provider="manual",
        source_query_id="q1",
        query_text="clean tyre sidewall",
    )
    restored = CandidateRecord.from_dict(record.to_dict())
    assert restored.candidate_id == record.candidate_id
    assert restored.source_url == "https://example.com/tyre.jpg"
    assert restored.human_review_status == "pending_review"


def test_manual_provider_loading(tmp_path: Path) -> None:
    manual_csv = tmp_path / "manual_urls.csv"
    manual_csv.write_text(
        "source_url,source_query_id,query_text,page_url\n"
        "https://example.com/a.jpg,q1,clean tyre,https://example.com/page\n",
        encoding="utf-8",
    )
    records = ManualURLProvider(manual_csv).collect()
    assert len(records) == 1
    assert records[0].source_provider == "manual"
    assert records[0].source_query_id == "q1"


def test_local_file_ingestion_path(tmp_path: Path) -> None:
    image_path = tmp_path / "local_tyre.png"
    Image.fromarray(np.full((256, 256, 3), 180, dtype=np.uint8)).save(image_path)
    manual_csv = tmp_path / "manual_local.csv"
    manual_csv.write_text(
        "candidate_id,source_provider,query_text,source_url,page_url,local_source_path,product_type,notes\n"
        f"pilot_local_001,local_file,clean tyre,,,{image_path},tyre,test local file\n",
        encoding="utf-8",
    )
    df = collect_manual_candidates(manual_csv)
    assert len(df) == 1
    assert df.iloc[0]["candidate_id"] == "pilot_local_001"
    assert df.iloc[0]["source_url"].startswith("file://")
    assert df.iloc[0]["local_source_path"] == str(image_path)


def test_dedupe_filter_behavior(tmp_path: Path) -> None:
    image_path = tmp_path / "tyre_a.png"
    duplicate_path = tmp_path / "tyre_b.png"
    arr = np.full((256, 256, 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(image_path)
    Image.fromarray(arr).save(duplicate_path)

    records = [
        CandidateRecord.from_url(str(image_path), source_query_id="q1"),
        CandidateRecord.from_url(str(duplicate_path), source_query_id="q1"),
    ]
    df = pd.DataFrame([record.to_dict() for record in records])
    df["local_raw_path"] = [str(image_path), str(duplicate_path)]
    df["download_status"] = "downloaded"

    result = apply_quality_and_dedupe_filters(
        df,
        {
            "quality_filtering": {"min_width": 224, "min_height": 224, "blur_laplacian_threshold": 0.0},
            "deduplication": {"exact_file_hash": True, "perceptual_hash": True, "hash_distance_threshold": 6},
        },
    )
    assert result.summary["total"] == 2
    assert result.summary["kept"] == 1
    assert result.summary["rejected"] == 1


def test_review_queue_generation(tmp_path: Path) -> None:
    record = CandidateRecord.from_url("https://example.com/a.jpg")
    df = pd.DataFrame([record.to_dict()])
    df["quality_status"] = "kept"
    queue = build_review_queue(df)
    assert len(queue) == 1
    assert queue.iloc[0]["human_review_status"] == "pending_review"

    out = tmp_path / "candidates.csv"
    write_candidates_csv(out, df)
    assert len(read_candidates_csv(out)) == 1


def test_pilot_orchestration_local_file(tmp_path: Path) -> None:
    image_path = tmp_path / "local_tyre.png"
    Image.fromarray(np.full((256, 256, 3), 200, dtype=np.uint8)).save(image_path)
    manual_csv = tmp_path / "approved_tyres_pilot_urls_001.csv"
    manual_csv.write_text(
        "candidate_id,source_provider,query_text,source_url,page_url,local_source_path,product_type,notes\n"
        f"pilot_local_001,local_file,clean tyre,,,{image_path},tyre,test local file\n",
        encoding="utf-8",
    )
    result = run_manual_pilot(
        input_csv=manual_csv,
        output_dir=tmp_path / "pilot_out",
        config={
            "product": {"type": "tyre"},
            "download": {"timeout_seconds": 5},
            "quality_filtering": {"min_width": 224, "min_height": 224, "blur_laplacian_threshold": 0.0},
            "deduplication": {"exact_file_hash": True, "perceptual_hash": True, "hash_distance_threshold": 6},
        },
    )
    assert result.counts["candidates"] == 1
    assert result.counts["downloaded"] == 1
    assert result.counts["review_queue"] == 1


def test_review_pack_generation_when_local_images_exist(tmp_path: Path) -> None:
    image_path = tmp_path / "candidate.png"
    Image.fromarray(np.full((256, 256, 3), 150, dtype=np.uint8)).save(image_path)
    queue = pd.DataFrame(
        [
            {
                "candidate_id": "candidate_1",
                "local_raw_path": str(image_path),
                "source_url": image_path.resolve().as_uri(),
                "page_url": "",
                "local_source_path": str(image_path),
                "query_text": "clean tyre",
                "quality_status": "kept",
                "anomaly_score": "",
                "anomaly_triage_bucket": "not_scored",
                "human_review_status": "pending_review",
                "review_notes": "",
                "product_type": "tyre",
            }
        ]
    )
    queue_csv = tmp_path / "review_queue.csv"
    queue.to_csv(queue_csv, index=False)
    result = export_review_pack(queue_csv, tmp_path / "review_pack")
    assert result["rows"] == 1
    assert Path(str(result["gallery"])).exists()
    assert Path(str(result["contact_sheet"])).exists()


def test_reviewed_candidate_promotion(tmp_path: Path) -> None:
    image_path = tmp_path / "approved.png"
    Image.fromarray(np.full((256, 256, 3), 150, dtype=np.uint8)).save(image_path)
    decisions = pd.DataFrame(
        [
            {
                "candidate_id": "approved_1",
                "local_raw_path": str(image_path),
                "source_url": image_path.resolve().as_uri(),
                "page_url": "",
                "human_review_status": "approved_likely_normal",
                "product_type": "tyre",
            },
            {
                "candidate_id": "rejected_1",
                "local_raw_path": str(image_path),
                "source_url": image_path.resolve().as_uri(),
                "page_url": "",
                "human_review_status": "rejected_likely_defect",
                "product_type": "tyre",
            },
        ]
    )
    decisions_csv = tmp_path / "decisions.csv"
    out_manifest = tmp_path / "curated.csv"
    decisions.to_csv(decisions_csv, index=False)
    result = promote_reviewed_candidates(decisions_csv, out_manifest)
    manifest = pd.read_csv(out_manifest)
    assert result["approved_count"] == 1
    assert len(manifest) == 1
    assert manifest.iloc[0]["target"] == 0
