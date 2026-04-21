from __future__ import annotations

from pathlib import Path

import pytest

from scripts.web_collection.provider_smoke_check import provider_statuses
from src.web_collection.io import load_yaml
from src.web_collection.providers import build_provider
from src.web_collection.providers_flickr import FlickrProvider, MissingProviderCredentialError as FlickrCredentialError
from src.web_collection.providers_manual_google import ManualGoogleDiscoveryProvider
from src.web_collection.providers_pexels import MissingProviderCredentialError as PexelsCredentialError
from src.web_collection.providers_pexels import PexelsProvider
from src.web_collection.providers_unsplash import MissingProviderCredentialError as UnsplashCredentialError
from src.web_collection.providers_unsplash import UnsplashProvider
from src.web_collection.schemas import CANDIDATE_COLUMNS


REQUIRED_SOURCE_FIELDS = {
    "provider",
    "query_id",
    "query_text",
    "source_url",
    "page_url",
    "license_name",
    "license_url",
    "attribution_text",
    "retrieval_timestamp",
    "product_type",
    "view_type",
    "notes",
}


def test_provider_source_config_parses() -> None:
    config = load_yaml("configs/web_collection/provider_sources.yaml")
    assert config["policy"]["google_mode"] == "manual_discovery_only"
    assert "wikimedia_commons" in config["sources"]
    assert "manual_google_discovery" in config["sources"]
    assert set(config["metadata_required"]) >= REQUIRED_SOURCE_FIELDS


def test_manual_google_discovery_import_preserves_source_metadata(tmp_path: Path) -> None:
    manual_csv = tmp_path / "manual_google.csv"
    manual_csv.write_text(
        "provider,query_id,query_text,source_url,page_url,license_name,license_url,"
        "attribution_text,retrieval_timestamp,product_type,view_type,notes\n"
        "manual_google_discovery,q1,clean tyre,https://example.com/tyre.jpg,"
        "https://example.com/page,example-license,https://example.com/license,"
        "Example Source,2026-04-19T00:00:00+00:00,tyre,tread,manual approval row\n",
        encoding="utf-8",
    )
    records = ManualGoogleDiscoveryProvider(manual_csv).collect()
    assert len(records) == 1
    record = records[0]
    assert record.provider == "manual_google_discovery"
    assert record.source_provider == "manual_google_discovery"
    assert record.query_id == "q1"
    assert record.source_query_id == "q1"
    assert record.license_name == "example-license"
    assert record.attribution_text == "Example Source"
    assert record.view_type == "tread"


def test_missing_api_key_failure_clarity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)

    with pytest.raises(PexelsCredentialError, match="PEXELS_API_KEY"):
        PexelsProvider().check_credentials()
    with pytest.raises(UnsplashCredentialError, match="UNSPLASH_ACCESS_KEY"):
        UnsplashProvider().check_credentials()
    with pytest.raises(FlickrCredentialError, match="FLICKR_API_KEY"):
        FlickrProvider().check_credentials()


def test_metadata_schema_contains_source_and_license_fields() -> None:
    assert set(CANDIDATE_COLUMNS) >= REQUIRED_SOURCE_FIELDS
    assert {"source_provider", "source_query_id", "query_family"} <= set(CANDIDATE_COLUMNS)


def test_provider_smoke_check_reports_manual_and_pending_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    config = load_yaml("configs/web_collection/provider_sources.yaml")
    statuses = {item["provider"]: item["status"] for item in provider_statuses(config)}
    assert statuses["wikimedia_commons"] == "ready_no_api_key"
    assert statuses["manual_google_discovery"] == "ready_manual_csv"
    assert statuses["pexels"] == "pending_api_key"
    assert statuses["unsplash"] == "pending_api_key"
    assert statuses["flickr"] == "pending_api_key"


def test_build_provider_recognizes_official_provider_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    provider = build_provider({"provider": {"type": "pexels"}, "product": {"type": "tyre"}})
    with pytest.raises(PexelsCredentialError, match="PEXELS_API_KEY"):
        provider.collect()
