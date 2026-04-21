# Anomaly Baseline Status

Updated: 2026-04-19

## Implemented In This Prompt

- Canonical anomaly manifest generation under `data/manifests/`.
- Good-only training manifest role: `normal_train`.
- Mixed validation/test manifest roles: `val_mixed`, `test_mixed`.
- Anomaly data contract documentation.
- Manifest-driven anomaly dataset class.
- Frozen ResNet embedding extractor.
- Mahalanobis distance scorer.
- kNN scorer for later comparison.
- Validation-only threshold selection.
- Test evaluation helpers.
- End-to-end pipeline orchestration.
- CLI entry point for fit/calibrate/evaluate.
- CSV/JSON/plot output hooks.
- Smoke tests for manifests, scorers, thresholds, and imports.

## Manifest Outputs Generated

- `data/manifests/D1_anomaly_train_normal.csv`
- `data/manifests/D1_anomaly_val_mixed.csv`
- `data/manifests/D1_anomaly_test_mixed.csv`

Generated counts:

| Manifest | Normal | Anomaly | Total |
|---|---:|---:|---:|
| normal train | 582 | 0 | 582 |
| validation mixed | 125 | 130 | 255 |
| test mixed | 125 | 130 | 255 |

All generated image paths resolved locally during manifest generation.

## Executed D1 Run

The first real D1 anomaly run completed after the local readiness check found cached ResNet-18 weights.

Run directory:

- `artifacts/anomaly/d1_resnet18_mahalanobis_v1`

Weight source:
- backbone: `resnet18`
- source: local torchvision cache, `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`

Test metrics at the validation-selected threshold:

| Metric | Value |
|---|---:|
| AUROC | 0.7194 |
| AUPRC | 0.7212 |
| anomaly recall | 0.2846 |
| anomaly precision | 0.7708 |
| normal FPR | 0.0880 |
| false negatives | 93 |
| false positives | 11 |

Confusion matrix on test (`[[TN, FP], [FN, TP]]`):

```text
[[114, 11],
 [ 93, 37]]
```

This is a real anomaly baseline result, but it is not yet strong enough for a recall-critical inspection claim.

Note: the run directory still contains an older `failure.json` from the earlier blocked attempt. The current authoritative files are `metadata.json`, `metrics_test.json`, prediction CSVs, and plots written in the completed run.

## Recommended Next Experiment

1. Historical reference: review:
   - `metadata.json`
   - `metrics_test.json`
   - `predictions_test.csv`
   - `false_negatives_test.csv`
   - `false_positives_test.csv`
   - score distribution plots

2. The next benchmark stage has now run. See `reports/current_status/anomaly_model_benchmark_status.md`.
3. Current best executed variant is `resnet50_knn`, with test recall `0.8231` and `23` false negatives.
4. Next inspect the remaining false negatives before adding more complex patch-aware methods.
