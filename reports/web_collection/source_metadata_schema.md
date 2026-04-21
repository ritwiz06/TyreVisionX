# Source Metadata Schema

Updated: 2026-04-19

## Required Source-Acquisition Fields

| Field | Required | Meaning |
|---|---:|---|
| `provider` | yes | General provider name, for example `wikimedia_commons` or `manual_google_discovery`. |
| `query_id` | yes | Query identifier from the query catalog. |
| `query_text` | yes | Search text or discovery description. |
| `source_url` | yes for URL rows | Direct image URL or file URI. |
| `page_url` | recommended | Source page where the image was found. |
| `license_name` | recommended | License name shown by provider/source page. |
| `license_url` | recommended | Link to license terms. |
| `attribution_text` | recommended | Creator/source attribution. |
| `retrieval_timestamp` | yes | When the row was collected/imported. |
| `product_type` | yes | `tyre` for TyreVisionX. |
| `view_type` | recommended | `tread`, `sidewall`, `mounted`, `inspection`, etc. |
| `notes` | optional | Researcher notes, uncertainty, or rights concerns. |

## Compatibility Fields

The existing TyreVisionX pipeline also keeps:

- `source_provider`
- `source_query_id`
- `query_family`
- download status
- hashes and quality fields
- anomaly triage fields
- human review fields

`provider` mirrors `source_provider`, and `query_id` mirrors `source_query_id` so old candidate CSVs remain compatible.

## Labeling Rule

No metadata field is a ground-truth label. A candidate becomes usable as likely-normal training data only after human review marks it `approved_likely_normal`.
