# Source Strategy Report

Updated: 2026-04-19

## Status

TyreVisionX now has a safe source-acquisition layer for tyre candidate discovery.

Implemented:
- source-tier policy
- provider source config
- Wikimedia/Pexels/Unsplash/Flickr adapter scaffolds
- manual Google-discovery CSV workflow
- source/license metadata schema extension
- provider smoke-check script

Not implemented:
- Google HTML scraping
- automatic labeling
- bulk provider downloads
- rights validation for production use

## Source Strategy

1. Prefer manufacturer/internal tyre images when available.
2. Use Wikimedia Commons through official API metadata where useful.
3. Use Pexels, Unsplash, and Flickr only after API credentials and terms are reviewed.
4. Use Google only as manual discovery: browser search plus researcher-approved CSV.

## Candidate Status

All source-acquired images are research candidates. They must pass quality/deduplication checks and human review before any row can be promoted to a curated likely-normal manifest.

## First Pilot Recommendation

Create 30-50 manually approved candidate rows across:
- tread close-ups
- sidewall close-ups
- mounted tyre product photos
- inspection-like views
- industrial/off-highway tyre exterior views

Use `data/external/manual_candidate_urls/manual_google_discovery_template.csv` or a copied CSV with the same schema.
