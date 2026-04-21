"""Wikimedia Commons provider adapter.

This adapter targets the official Wikimedia Commons API. It is intentionally
lightweight: it builds/validates provider metadata and request parameters, but
does not run network collection unless a future script explicitly adds that.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode


@dataclass(frozen=True)
class WikimediaCommonsProvider:
    """Configuration helper for Wikimedia Commons candidate discovery."""

    endpoint: str = "https://commons.wikimedia.org/w/api.php"
    provider_name: str = "wikimedia_commons"

    def check_credentials(self) -> None:
        """Wikimedia Commons image search does not require an API key."""

    def build_search_url(self, query_text: str, limit: int = 25) -> str:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query_text,
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "format": "json",
        }
        return f"{self.endpoint}?{urlencode(params)}"

    def source_status(self) -> dict[str, str]:
        return {
            "provider": self.provider_name,
            "status": "ready_no_api_key",
            "notes": "Use the official Wikimedia Commons API; retain license and attribution metadata.",
        }
