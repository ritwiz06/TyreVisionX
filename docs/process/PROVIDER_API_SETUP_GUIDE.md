# Provider API Setup Guide

Updated: 2026-04-19

## Purpose

TyreVisionX supports official-provider source acquisition in a controlled way. Provider adapters must use documented APIs and must fail clearly when credentials are missing.

## Current Provider Readiness

| Provider | Mode | Credential Required | Current Status |
|---|---|---:|---|
| Manufacturer/internal export | manual/internal | no | ready when user has data |
| Wikimedia Commons | official API | no | config/check ready |
| Pexels | official API | yes: `PEXELS_API_KEY` | pending key |
| Unsplash | official API | yes: `UNSPLASH_ACCESS_KEY` | pending key |
| Flickr | official API | yes: `FLICKR_API_KEY` | pending key |
| Google | manual CSV only | no | manual discovery only |

## Environment Variables

Set credentials only when you plan to use that provider:

```bash
export PEXELS_API_KEY="..."
export UNSPLASH_ACCESS_KEY="..."
export FLICKR_API_KEY="..."
```

The repository does not store credentials.

## Smoke Check

Run:

```bash
python scripts/web_collection/provider_smoke_check.py \
  --config configs/web_collection/provider_sources.yaml
```

Expected behavior:
- Wikimedia reports `ready_no_api_key`.
- Manual Google reports `ready_manual_csv`.
- Pexels/Unsplash/Flickr report `pending_api_key` unless credentials are set.

## Why Not Implement Bulk Provider Downloads Yet

The current priority is a small tyre-specific pilot and review workflow. Bulk provider downloads should wait until the first 30-50 candidate pilot proves that metadata, filtering, and review outputs are useful.
