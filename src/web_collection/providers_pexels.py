"""Pexels API provider adapter."""
from __future__ import annotations

import os
from dataclasses import dataclass


class MissingProviderCredentialError(RuntimeError):
    """Raised when an API provider is configured without required credentials."""


@dataclass(frozen=True)
class PexelsProvider:
    """Configuration helper for the official Pexels API."""

    api_key_env: str = "PEXELS_API_KEY"
    endpoint: str = "https://api.pexels.com/v1/search"
    provider_name: str = "pexels"

    def check_credentials(self) -> str:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise MissingProviderCredentialError(
                f"Pexels provider requires environment variable {self.api_key_env}. "
                "Use manual CSV mode or set the key before running provider collection."
            )
        return api_key

    def source_status(self) -> dict[str, str]:
        try:
            self.check_credentials()
            status = "ready_with_api_key"
        except MissingProviderCredentialError:
            status = "pending_api_key"
        return {
            "provider": self.provider_name,
            "status": status,
            "api_key_env": self.api_key_env,
            "notes": "Use the official Pexels API and preserve photographer/source attribution.",
        }
