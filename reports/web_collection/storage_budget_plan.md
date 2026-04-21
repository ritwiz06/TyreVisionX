# Web Collection Storage Budget Plan

Updated: 2026-04-19

Planning budget: `100 GB`

These numbers are planning targets, not guarantees.

## Collection Targets

- Raw candidate target: `12,000` to `15,000` images
- Expected curated likely-good subset after filtering/review: `4,000` to `7,000` images

## Budget Estimate

| Category | Estimated Budget | Notes |
|---|---:|---|
| Raw candidate originals | 45 GB | Preserve original downloads for traceability. |
| Processed/resized review copies | 15 GB | Optional future thumbnails or normalized copies. |
| Curated likely-normal set | 15 GB | Reviewed subset and derived manifests. |
| Metadata, review CSVs, reports | 2 GB | CSV, JSON, markdown, and plots. |
| Anomaly artifacts and embeddings | 15 GB | Feature arrays, scorers, plots, run metadata. |
| Buffer | 8 GB | Failed downloads, experiments, temporary files. |

Total: `100 GB`

## Policy

Do not delete raw originals silently. If space pressure occurs, archive with a documented manifest rather than removing files without traceability.
