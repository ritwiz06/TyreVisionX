# Anomaly Triage Status

Status: implemented as an advisory hook; not validated on web candidates.

Anomaly triage can update candidate metadata with review-priority buckets after fitted anomaly artifacts are available. It must not create ground-truth labels.

Current default config has anomaly triage disabled. The first D1 anomaly baseline has run, but its recall is low and no real web-candidate pilot has been collected. Therefore, web-candidate anomaly scoring is not validated and must remain advisory only.

Expected future buckets:
- `likely_normal`
- `uncertain`
- `likely_anomalous`

Use these only for review prioritization.

Do not use anomaly triage to auto-label web candidates as `good`, `normal`, `defect`, or `anomaly`.
