# Corruption Benchmark Plan

Updated: 2026-04-20

## Purpose

Evaluate how realistic nuisance changes affect the current best TyreVisionX anomaly variants. This is a stress test, not a new training result.

## Primary Rule

Use the clean validation-selected threshold for each variant. Do not retune on corrupted test data.

## Corruption Families

| Family | Levels | Why It Is Realistic |
|---|---|---|
| Gaussian noise | low, medium | Camera sensor noise and low-light acquisition noise. |
| Gaussian blur | low, medium | Mild motion blur or imperfect focus. |
| JPEG compression | mild | Web or device compression artifacts. |
| Brightness shift | darker, brighter | Lighting variation during capture. |
| Contrast shift | lower, higher | Different cameras, illumination, and exposure settings. |

These are intentionally moderate. Extreme corruptions are not included because they would not represent normal tyre inspection conditions.

## Variants

- `resnet50_knn`
- `resnet50_mahalanobis`
- `resnet18_knn_control`

## Outputs

- `reports/anomaly/corruption_benchmark.csv`
- `reports/anomaly/corruption_benchmark.md`
- `reports/current_status/anomaly_robustness_status.md`
