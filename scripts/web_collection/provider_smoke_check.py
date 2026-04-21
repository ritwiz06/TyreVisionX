"""Check web-source provider configuration without downloading images."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.io import load_yaml
from src.web_collection.providers_flickr import FlickrProvider
from src.web_collection.providers_pexels import PexelsProvider
from src.web_collection.providers_unsplash import UnsplashProvider
from src.web_collection.providers_wikimedia import WikimediaCommonsProvider


def provider_statuses(config: dict) -> list[dict[str, str]]:
    sources = config.get("sources", {})
    statuses: list[dict[str, str]] = []

    if "wikimedia_commons" in sources:
        statuses.append(WikimediaCommonsProvider().source_status())
    if "pexels" in sources:
        pexels_cfg = sources["pexels"]
        statuses.append(PexelsProvider(api_key_env=pexels_cfg.get("api_key_env", "PEXELS_API_KEY")).source_status())
    if "unsplash" in sources:
        unsplash_cfg = sources["unsplash"]
        statuses.append(
            UnsplashProvider(access_key_env=unsplash_cfg.get("access_key_env", "UNSPLASH_ACCESS_KEY")).source_status()
        )
    if "flickr" in sources:
        flickr_cfg = sources["flickr"]
        statuses.append(FlickrProvider(api_key_env=flickr_cfg.get("api_key_env", "FLICKR_API_KEY")).source_status())

    if "manual_google_discovery" in sources:
        statuses.append(
            {
                "provider": "manual_google_discovery",
                "status": "ready_manual_csv",
                "notes": "Reads researcher-prepared CSV only; no Google HTML scraping.",
            }
        )
    if "manufacturer_internal" in sources:
        statuses.append(
            {
                "provider": "manufacturer_internal",
                "status": "ready_manual_or_internal",
                "notes": "Preferred source tier when approved internal data exists.",
            }
        )
    return statuses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check configured web-image providers.")
    parser.add_argument("--config", default="configs/web_collection/provider_sources.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    for status in provider_statuses(config):
        print(f"{status['provider']}: {status['status']} - {status.get('notes', '')}")


if __name__ == "__main__":
    main()
