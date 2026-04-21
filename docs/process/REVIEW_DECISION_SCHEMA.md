# Review Decision Schema

Updated: 2026-04-19

Human review decisions are required before web candidates can be used as likely-normal anomaly training data.

## Required Columns

| Column | Meaning |
|---|---|
| `candidate_id` | Candidate identifier from the review queue. |
| `local_raw_path` | Local copied/downloaded image path. |
| `source_url` | Original image URL or `file://` source. |
| `page_url` | Source page URL when available. |
| `query_text` | Search/query context or manual selection reason. |
| `quality_status` | Filtering output such as `kept` or `review_needed`. |
| `anomaly_score` | Optional advisory score. |
| `anomaly_triage_bucket` | Optional advisory bucket. |
| `human_review_status` | Final human decision status. |
| `review_notes` | Reviewer notes. |
| `reviewer` | Reviewer name or initials. |
| `reviewed_at` | Review timestamp. |
| `decision_notes` | Extra decision rationale. |

## Allowed Human Review Statuses

- `pending_review`
- `approved_likely_normal`
- `rejected_irrelevant`
- `rejected_low_quality`
- `rejected_likely_defect`
- `uncertain_holdout`

Only `approved_likely_normal` can be promoted into a curated likely-normal manifest.
