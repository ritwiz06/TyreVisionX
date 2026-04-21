# Web Normal Expansion Strategy

Updated: 2026-04-19

## Status

No real web candidate pilot has been collected yet. This strategy defines how reviewed web normals may later expand TyreVisionX anomaly training safely.

## Safe Path

1. Create approved manual pilot CSV.
2. Collect candidate metadata.
3. Copy/download images only from approved rows.
4. Run quality and deduplication filters.
5. Optionally apply anomaly triage as review priority.
6. Human review candidates.
7. Promote only `approved_likely_normal` rows to a curated web-normal manifest.
8. Keep curated web normals separate from benchmark D1/D2/D3 data.

## Forbidden Path

Do not pseudo-label scraped images as good or defect. Do not use low anomaly scores as automatic good labels. Do not use web candidates for defect-vs-good classifier retraining before a human-reviewed dataset exists.

## Why This Matters

The current anomaly baseline has low recall, so it is not reliable enough to label web images. It can later help prioritize review, but it cannot replace review.
