# Web Collection Status

Updated: 2026-04-19

## Implemented

- Web collection policy documentation.
- Web curation workflow documentation.
- Query catalog generator.
- Editable seed query catalog with 24 likely-normal tyre queries.
- Manual CSV/JSON URL provider.
- Search-provider and Google Custom Search placeholder interfaces.
- Candidate metadata schema.
- Download/copy script for approved URLs or local files.
- Deduplication and quality-filtering script.
- Optional anomaly-triage status hook.
- Human review queue builder.
- Review guidelines and template CSV.
- Manual pilot input contract and template.
- Manual pilot orchestrator.
- Visual review pack exporter.
- Reviewed-candidate promotion script.
- Source acquisition docs for safe provider/manual discovery.
- Provider source config with safe tiers for manufacturer/internal, Wikimedia Commons, Pexels, Unsplash, Flickr, and manual Google discovery.
- Manual Google-discovery CSV import workflow.
- License/source metadata fields in candidate records.
- Provider smoke-check script.
- Smoke tests for query generation, schema serialization, manual provider loading, filtering, and review queues.

## Pending

- Full provider API collection/download implementation.
- Approved 20-50 row pilot URL/local-file CSV from the researcher.
- Real pilot candidate download/copy run.
- Human review of candidates.
- Anomaly scoring on web candidates after fitted anomaly artifacts exist.

## Current Boundaries

Web candidates are not labels. They are research candidates that require source metadata, filtering, anomaly triage if available, and human review before use.
