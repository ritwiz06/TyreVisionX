# Web Source Acquisition Guide

Updated: 2026-04-19

## Purpose

TyreVisionX can use web images only as research candidates for expanding the pool of likely-normal tyre images. Source acquisition means finding candidate image URLs with enough provenance to review, filter, and possibly promote later.

Web acquisition is not labeling. A collected image is not `good` just because the search query looked normal.

## Safe Source Tiers

| Tier | Source | Status | Why It Is Used |
|---:|---|---|---|
| 1 | Manufacturer/internal data | preferred when available | Strongest provenance and most inspection-relevant imagery. |
| 2 | Wikimedia Commons | supported with official API metadata | Often includes license and attribution metadata. |
| 3 | Pexels | adapter scaffold, needs API key | Useful public image API; rights and relevance still need review. |
| 3 | Unsplash | adapter scaffold, needs access key | Useful public image API; rights and relevance still need review. |
| 3 | Flickr | adapter scaffold, needs API key | Can expose license metadata through official API. |
| 4 | Manual Google discovery | manual CSV only | Useful for researcher discovery, but no Google HTML scraping is allowed. |

## Google Rule

Google is manual discovery only in this repository.

Allowed:
- A researcher uses a browser.
- The researcher manually selects candidate image/page URLs.
- The researcher enters approved rows into `data/external/manual_candidate_urls/manual_google_discovery_template.csv`.
- TyreVisionX imports that CSV as candidate metadata.

Not allowed:
- Scraping Google HTML.
- Automating browser result extraction.
- Treating Google results as labeled good tyres.

## Required Metadata

Every candidate should track:

- provider
- query_id
- query_text
- source_url
- page_url
- license_name
- license_url
- attribution_text
- retrieval_timestamp
- product_type
- view_type
- notes

Additional pipeline fields such as hashes, quality status, anomaly score, and human review status are added later.

## Rights Caution

Public images may be useful for research experiments, but rights and terms matter. Keep URLs, page URLs, license names, and attribution text so future productization decisions can be reviewed. Do not assume a public image can be used in production.

## Why Candidates Are Not Labels

Search queries are weak signals. A result for "new tyre tread close up" can still be used, damaged, edited, irrelevant, or mislabeled. TyreVisionX requires filtering and human review before a candidate can enter a curated likely-normal manifest.
