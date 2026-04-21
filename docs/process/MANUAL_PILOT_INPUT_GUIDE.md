# Manual Pilot Input Guide

Updated: 2026-04-19

This guide explains how to prepare the first TyreVisionX manual candidate-ingestion pilot.

## Goal

Create a small, approved list of 20-50 likely-normal tyre image candidates. The list can reference approved image URLs or local files for offline testing.

## Input Location

Place the real pilot CSV under:

```text
data/external/manual_candidate_urls/
```

Do not include `template` in the filename for the real run. Example:

```text
data/external/manual_candidate_urls/approved_tyres_pilot_urls_001.csv
```

## Recommended Columns

| Column | Required | Meaning |
|---|---|---|
| `candidate_id` | optional | Stable ID chosen by the researcher. If blank, the pipeline generates one. |
| `source_provider` | recommended | `manual`, `local_file`, provider name, or source collection method. |
| `query_text` | recommended | Search/query phrase or reason the image was selected. |
| `source_url` | required if no local path | Image URL or `file://` URI. |
| `page_url` | recommended | Source page URL for provenance. |
| `local_source_path` | required if no URL | Local file path for offline/controlled pilots. |
| `product_type` | recommended | Use `tyre` for this project phase. |
| `notes` | optional | Researcher notes about why this candidate was selected. |

## Rules

- Do not use scraped Google HTML as a source.
- Do not auto-label images as good.
- Keep provenance fields.
- Human review is required before any candidate can be promoted.
- Local files are acceptable for a controlled dry run or offline pilot.

## First Pilot Command

```bash
python3 scripts/web_collection/run_manual_pilot.py \
  --input_csv data/external/manual_candidate_urls/approved_tyres_pilot_urls_001.csv
```

If no real input CSV exists, the pipeline reports `blocked_missing_approved_input`.
