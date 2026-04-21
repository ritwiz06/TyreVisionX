# Safe Web Normal Expansion Policy

Updated: 2026-04-19

TyreVisionX may use web-sourced tyre images only as research candidates until human review approves them.

## No Automatic Labeling

The pipeline must not automatically label web candidates as `good`, `normal`, `defect`, or `anomaly`.

Forbidden:
- using query text alone as a good label
- using a low anomaly score as a good label
- using a high anomaly score as a defect label
- pseudo-labeling scraped images for defect-vs-good classifier retraining
- mixing unreviewed web candidates into anomaly training

## Allowed Triage Signals

These signals may prioritize review but must not create labels:
- query family or query priority
- image quality filters
- exact/perceptual duplicate checks
- anomaly score or anomaly triage bucket
- optional supervised relevance score, if later implemented, as advisory only

## Promotion Rule

Only candidates marked `approved_likely_normal` by human review may enter curated web-normal manifests.

## Why Pseudo-Labeling Is Risky

Web images are noisy. They may show damaged tyres, irrelevant objects, edited product photos, unknown usage rights, or duplicated images. Training on pseudo-labeled web data can contaminate the normal set and teach the anomaly model that real defects are normal.

## Safe Future Use

Reviewed web normals can later be used to:
- expand normal-only anomaly training
- test robustness to lighting/viewpoint variation
- build separate curated-web manifests

They must remain separate from benchmark D1/D2/D3 labels.
