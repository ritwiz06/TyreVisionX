# Web Curation Workflow

Updated: 2026-04-19

This workflow describes how TyreVisionX expands likely-normal tyre data while keeping labels honest.

## Step 1: Generate Query Catalog

Run:

```bash
python3 scripts/web_collection/generate_query_catalog.py
```

Output:
- `configs/web_collection/query_catalog.yaml`
- `reports/web_collection/query_catalog_report.md`

Queries are grouped by intent, view type, context, and priority. They are editable by hand.

## Step 2: Collect Candidate URLs

Use manual CSV/JSON import first:

```bash
python3 scripts/web_collection/collect_candidates.py --config configs/web_collection/web_collection.yaml
```

The manual input file should include `source_url` or `url`. Optional columns include `query_id`, `query_text`, `page_url`, and `product_type`.

Output:
- `data/interim/web_collection_metadata.csv`

## Step 3: Download Or Copy Candidate Images

Run only with approved URLs or local test files:

```bash
python3 scripts/web_collection/download_candidates.py --config configs/web_collection/web_collection.yaml
```

Output:
- raw images under `data/external/web_raw/`
- updated metadata with `local_raw_path` and `download_status`

## Step 4: Filter And Deduplicate

Run:

```bash
python3 scripts/web_collection/filter_candidates.py --config configs/web_collection/web_collection.yaml
```

Outputs:
- `candidates_kept.csv`
- `candidates_rejected.csv`
- `candidates_review_needed.csv`
- `filter_summary.json`

Hard filters reject missing downloads, invalid images, and images below minimum size. Soft flags route blurry, unusual-aspect, or near-duplicate images to review.

## Step 5: Optional Anomaly Triage

Run only after fitted anomaly artifacts exist:

```bash
python3 scripts/web_collection/score_candidates_with_anomaly.py --config configs/web_collection/web_collection.yaml
```

Buckets such as `likely_normal`, `uncertain`, and `likely_anomalous` are review-priority signals, not labels.

## Step 6: Build Human Review Queue

Run:

```bash
python3 scripts/web_collection/build_review_queue.py --config configs/web_collection/web_collection.yaml
```

Output:
- `data/interim/web_curated_candidates/review_queue.csv`

## Step 7: Human Review

Use statuses:
- `pending_review`
- `approved_likely_normal`
- `rejected_irrelevant`
- `rejected_low_quality`
- `rejected_likely_defect`
- `uncertain_holdout`

Only reviewed `approved_likely_normal` candidates should be considered for future anomaly training manifests.
