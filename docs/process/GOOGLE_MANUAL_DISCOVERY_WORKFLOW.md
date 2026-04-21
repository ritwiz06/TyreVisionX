# Google Manual Discovery Workflow

Updated: 2026-04-19

## Purpose

This workflow allows a researcher to use Google as a manual discovery aid without building a scraper.

## Steps

1. Open a query from `configs/web_collection/query_catalog.yaml` in a normal browser.
2. Manually inspect result pages and image previews.
3. Choose only candidates that look relevant to normal tyre appearance.
4. Record each approved candidate in:
   `data/external/manual_candidate_urls/manual_google_discovery_template.csv`
5. Include both `source_url` and `page_url` when possible.
6. Fill license/attribution fields if the source page provides them.
7. Import the CSV:

```bash
python scripts/web_collection/import_manual_google_discovery.py \
  --input_csv data/external/manual_candidate_urls/manual_google_discovery_template.csv \
  --out_csv data/interim/web_candidates/manual_google_candidates.csv
```

8. Run the existing download/copy, filter, review queue, and review-pack steps.

## Required CSV Fields

| Field | Meaning |
|---|---|
| `provider` | Use `manual_google_discovery`. |
| `query_id` | Query ID from the TyreVisionX query catalog. |
| `query_text` | Search text used by the researcher. |
| `source_url` | Direct image URL if known. |
| `page_url` | Page where the image was found. |
| `license_name` | License shown by the source, if known. |
| `license_url` | License URL, if known. |
| `attribution_text` | Photographer/source attribution, if known. |
| `retrieval_timestamp` | When the row was recorded. |
| `product_type` | `tyre` for this project. |
| `view_type` | Example: `tread`, `sidewall`, `mounted`, `inspection`. |
| `notes` | Researcher notes and uncertainty. |

## Safety Rules

- Do not scrape Google HTML.
- Do not batch-extract search result URLs with browser automation.
- Do not auto-label candidates as good.
- Human review is mandatory before promotion.
