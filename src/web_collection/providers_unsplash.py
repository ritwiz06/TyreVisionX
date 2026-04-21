"""Unsplash API provider adapter."""
from __future__ import annotations

import os
from dataclasses import dataclass


class MissingProviderCredentialError(RuntimeError):
    """Raised when an API provider is configured without required credentials."""


@dataclass(frozen=True)
class UnsplashProvider:
    """Configuration helper for the official Unsplash API."""

    access_key_env: str = "UNSPLASH_ACCESS_KEY"
    endpoint: str = "https://api.unsplash.com/search/photos"
    provider_name: str = "unsplash"

    def check_credentials(self) -> str:
        access_key = os.environ.get(self.access_key_env, "").strip()
        if not access_key:
            raise MissingProviderCredentialError(
                f"Unsplash provider requires environment variable {self.access_key_env}. "
                "Use manual CSV mode or set the key before running provider collection."
            )
        return access_key

    def source_status(self) -> dict[str, str]:
        try:
            self.check_credentials()
            status = "ready_with_api_key"
        except MissingProviderCredentialError:
            status = "pending_api_key"
        return {
            "provider": self.provider_name,
            "status": status,
            "access_key_env": self.access_key_env,
            "notes": "Use the official Unsplash API and retain source/attribution metadata.",
        }
