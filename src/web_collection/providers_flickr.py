"""Flickr API provider adapter."""
from __future__ import annotations

import os
from dataclasses import dataclass


class MissingProviderCredentialError(RuntimeError):
    """Raised when an API provider is configured without required credentials."""


@dataclass(frozen=True)
class FlickrProvider:
    """Configuration helper for the official Flickr API."""

    api_key_env: str = "FLICKR_API_KEY"
    endpoint: str = "https://www.flickr.com/services/rest/"
    provider_name: str = "flickr"

    def check_credentials(self) -> str:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise MissingProviderCredentialError(
                f"Flickr provider requires environment variable {self.api_key_env}. "
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
            "notes": "Use the official Flickr API and preserve license/owner/page metadata.",
        }
