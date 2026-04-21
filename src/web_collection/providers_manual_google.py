"""Manual Google discovery import.

This module deliberately does not scrape Google. It reads a researcher-prepared
CSV of browser-discovered, manually approved candidate rows.
"""
from __future__ import annotations

from pathlib import Path

from src.web_collection.providers import ManualURLProvider
from src.web_collection.schemas import CandidateRecord


class ManualGoogleDiscoveryProvider(ManualURLProvider):
    """Load Google-discovered rows from an approved manual CSV."""

    def __init__(self, input_path: str | Path, product_type: str = "tyre") -> None:
        super().__init__(input_path=input_path, product_type=product_type)

    def collect(self) -> list[CandidateRecord]:
        records = super().collect()
        for record in records:
            record.provider = record.provider or "manual_google_discovery"
            record.source_provider = record.source_provider or record.provider
            if record.source_provider in {"manual", ""}:
                record.source_provider = "manual_google_discovery"
                record.provider = "manual_google_discovery"
            record.notes = (record.notes or "").strip()
        return records
