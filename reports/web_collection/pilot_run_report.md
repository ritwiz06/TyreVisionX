# Manual Pilot Run Report

Status: `blocked_missing_approved_input`

Input CSV: `not_found`
Output directory: `data/interim/web_candidates/pilot_01`

## Counts

| Stage | Count |
|---|---:|
| candidates | 0 |
| downloaded | 0 |
| kept | 0 |
| review_needed | 0 |
| rejected | 0 |

## Outputs

| Output | Path |
|---|---|
| candidate_metadata | `data/interim/web_candidates/pilot_01/candidate_metadata.csv` |
| downloaded_metadata | `data/interim/web_candidates/pilot_01/candidate_metadata_downloaded.csv` |
| kept | `data/interim/web_candidates/pilot_01/candidates_kept.csv` |
| rejected | `data/interim/web_candidates/pilot_01/candidates_rejected.csv` |
| review_needed | `data/interim/web_candidates/pilot_01/candidates_review_needed.csv` |
| review_queue | `data/interim/web_candidates/pilot_01/review_queue.csv` |
| filter_summary | `data/interim/web_candidates/pilot_01/filter_summary.json` |
| pilot_run_report | `reports/web_collection/pilot_run_report.md` |

## Pending

- Provide a reviewed/approved manual pilot CSV under data/external/manual_candidate_urls/.
- Use approved_tyres_pilot_urls_template.csv as the schema.

## Integrity Notes
- No candidate is auto-labeled as good.
- Human review is required before promotion to a likely-normal manifest.
- Anomaly triage is advisory only when available.
