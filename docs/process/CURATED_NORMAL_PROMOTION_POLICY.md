# Curated Normal Promotion Policy

Updated: 2026-04-19

This policy defines how web candidates can become curated likely-normal tyre examples.

## Promotion Rule

Only candidates marked `approved_likely_normal` by human review may be promoted.

The pipeline must not promote candidates based only on:
- query text
- quality filters
- low anomaly score
- AI assistant suggestion
- file name

## Output Manifest

Default output:

```text
data/manifests/web_curated_tyres_likely_normal_v1.csv
```

The manifest uses anomaly-compatible fields:
- `image_path`
- `target = 0`
- `label = 0`
- `label_str = likely_normal_web_reviewed`
- `split = curated_pool`
- `is_normal = true`
- `source_dataset = web_curated_pilot_01`
- `product_type = tyre`
- `candidate_id`
- `source_url`
- `page_url`
- `human_review_status`

## Difference From Benchmark Labels

Curated web normals are reviewed research candidates. They are not the same as benchmark dataset labels because the source conditions, rights, camera viewpoints, and review process differ.

Track them separately from D1/D2/D3 benchmark data.
