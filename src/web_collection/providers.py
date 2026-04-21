"""Provider abstraction for web-image candidate discovery."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.web_collection.io import read_manual_url_file
from src.web_collection.schemas import CandidateRecord


class CandidateProvider(ABC):
    """Base interface for candidate URL providers."""

    @abstractmethod
    def collect(self) -> list[CandidateRecord]:
        """Return candidate records without downloading image files."""


class ManualURLProvider(CandidateProvider):
    """Load candidate URLs from a user-provided CSV or JSON file."""

    def __init__(self, input_path: str | Path, product_type: str = "tyre") -> None:
        self.input_path = Path(input_path)
        self.product_type = product_type

    def collect(self) -> list[CandidateRecord]:
        rows = read_manual_url_file(self.input_path)
        records: list[CandidateRecord] = []
        for row in rows:
            source_url = str(row.get("source_url") or row.get("url") or "").strip()
            local_source_path = str(row.get("local_source_path") or "").strip()
            if not source_url and local_source_path:
                source_url = _source_url_from_local_path(local_source_path)
            if not source_url and not local_source_path:
                continue
            candidate_id = str(row.get("candidate_id") or "").strip()
            source_provider = str(row.get("source_provider") or row.get("provider") or "manual")
            source_query_id = str(row.get("source_query_id") or row.get("query_id") or "")
            record = CandidateRecord.from_url(
                source_url=source_url,
                source_provider=source_provider,
                provider=str(row.get("provider") or source_provider),
                source_query_id=source_query_id,
                query_id=str(row.get("query_id") or source_query_id),
                query_family=str(row.get("query_family") or ""),
                query_text=str(row.get("query_text") or ""),
                page_url=str(row.get("page_url") or ""),
                license_name=str(row.get("license_name") or ""),
                license_url=str(row.get("license_url") or ""),
                attribution_text=str(row.get("attribution_text") or ""),
                local_source_path=local_source_path,
                product_type=str(row.get("product_type") or self.product_type),
                view_type=str(row.get("view_type") or ""),
                notes=str(row.get("notes") or ""),
            )
            if candidate_id:
                record.candidate_id = candidate_id
            records.append(
                record
            )
        if not records:
            raise ValueError(
                f"No usable source_url/url/local_source_path values found in manual input: {self.input_path}"
            )
        return records


class SearchProviderStub(CandidateProvider):
    """Interface placeholder for future provider-backed search adapters."""

    def __init__(self, provider_name: str, config: dict[str, Any]) -> None:
        self.provider_name = provider_name
        self.config = config

    def collect(self) -> list[CandidateRecord]:
        raise NotImplementedError(
            f"{self.provider_name} search collection is not implemented in this research scaffold. "
            "Use provider.type=manual_csv_json, or implement an approved API adapter without scraping HTML."
        )


class GoogleCustomSearchProviderStub(SearchProviderStub):
    """Placeholder for a future Google Custom Search JSON API adapter."""

    def collect(self) -> list[CandidateRecord]:
        api_key_env = self.config.get("api_key_env", "GOOGLE_CUSTOM_SEARCH_API_KEY")
        engine_id_env = self.config.get("engine_id_env", "GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        raise NotImplementedError(
            "Google Custom Search is configured only as a future adapter placeholder. "
            f"Provide {api_key_env} and {engine_id_env}, then implement the official JSON API adapter. "
            "Do not scrape Google HTML."
        )


class OfficialAPIProviderStub(SearchProviderStub):
    """Safe placeholder for configured official API providers."""

    def collect(self) -> list[CandidateRecord]:
        if self.provider_name == "pexels":
            from src.web_collection.providers_pexels import PexelsProvider

            PexelsProvider(api_key_env=self.config.get("api_key_env", "PEXELS_API_KEY")).check_credentials()
        elif self.provider_name == "unsplash":
            from src.web_collection.providers_unsplash import UnsplashProvider

            UnsplashProvider(access_key_env=self.config.get("access_key_env", "UNSPLASH_ACCESS_KEY")).check_credentials()
        elif self.provider_name == "flickr":
            from src.web_collection.providers_flickr import FlickrProvider

            FlickrProvider(api_key_env=self.config.get("api_key_env", "FLICKR_API_KEY")).check_credentials()
        raise NotImplementedError(
            f"{self.provider_name} official API collection is configured but not implemented for bulk ingestion yet. "
            "Use the source-acquisition pilot plan first, or implement a provider-specific collector that preserves "
            "license, attribution, page URL, and query metadata."
        )


def build_provider(config: dict[str, Any]) -> CandidateProvider:
    provider_cfg = config.get("provider", {})
    provider_type = str(provider_cfg.get("type", "manual_csv_json"))
    product_type = str(config.get("product", {}).get("type", "tyre"))

    if provider_type == "manual_csv_json":
        input_path = provider_cfg.get("manual_input_path")
        if not input_path:
            raise ValueError("provider.manual_input_path is required for manual_csv_json mode.")
        return ManualURLProvider(input_path=input_path, product_type=product_type)
    if provider_type == "manual_google_discovery":
        from src.web_collection.providers_manual_google import ManualGoogleDiscoveryProvider

        input_path = provider_cfg.get("manual_input_path") or provider_cfg.get("manual_google_discovery", {}).get(
            "template_path"
        )
        if not input_path:
            raise ValueError("provider.manual_input_path is required for manual_google_discovery mode.")
        return ManualGoogleDiscoveryProvider(input_path=input_path, product_type=product_type)
    if provider_type == "google_custom_search":
        return GoogleCustomSearchProviderStub(provider_type, provider_cfg)
    if provider_type in {"wikimedia_commons", "pexels", "unsplash", "flickr"}:
        return OfficialAPIProviderStub(provider_type, provider_cfg)
    if provider_type in {"search_stub", "provider_stub"}:
        return SearchProviderStub(provider_type, provider_cfg)
    raise ValueError(
        f"Unsupported provider.type={provider_type!r}. Supported modes: manual_csv_json, "
        "google_custom_search placeholder, search_stub."
    )


def _source_url_from_local_path(path_text: str) -> str:
    parsed = urlparse(path_text)
    if parsed.scheme == "file":
        return path_text
    return Path(path_text).expanduser().resolve().as_uri()
