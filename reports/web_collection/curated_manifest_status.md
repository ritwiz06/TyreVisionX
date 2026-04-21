# Curated Manifest Status

Status: `pending_human_review`

Target future manifest:

```text
data/manifests/web_curated_tyres_likely_normal_v1.csv
```

Promotion rule:

Only candidates marked `approved_likely_normal` by human review may be promoted.

Model confidence, anomaly scores, query text, and quality filters are not sufficient for promotion without human review.

Current state:

- approved candidates: `0`
- manifest created from approvals: `no`
- reason: no real pilot input CSV and no human review decisions yet

Curated web normals must remain separate from benchmark labels because they come from a different provenance and review process.
