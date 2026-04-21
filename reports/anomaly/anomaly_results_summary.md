# Anomaly Results Summary

Updated: 2026-04-19

## Status

The first D1 tyre anomaly baseline executed successfully.

This is a real research baseline result, not a random-weight smoke test.

## Run

| Field | Value |
|---|---|
| Run name | `d1_resnet18_mahalanobis_v1` |
| Run directory | `artifacts/anomaly/d1_resnet18_mahalanobis_v1` |
| Backbone | `resnet18` |
| Weight source | local torchvision cache |
| Weight file | `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` |
| Embedding dimension | `512` |
| Scorer | Mahalanobis distance |
| Threshold policy | validation-only recall priority with normal FPR <= 10% |

## Data

| Split | Normal | Anomaly | Total |
|---|---:|---:|---:|
| normal train | 582 | 0 | 582 |
| validation mixed | 125 | 130 | 255 |
| test mixed | 125 | 130 | 255 |

## Validation Calibration

Threshold selected on validation only: `12.147390651173238`

| Metric | Value |
|---|---:|
| anomaly recall | 0.3846 |
| anomaly precision | 0.8333 |
| normal FPR | 0.0800 |
| false negatives | 80 |
| false positives | 10 |

## Test Result

| Metric | Value |
|---|---:|
| AUROC | 0.7194 |
| AUPRC | 0.7212 |
| anomaly recall | 0.2846 |
| anomaly precision | 0.7708 |
| normal FPR | 0.0880 |
| accuracy | 0.5922 |
| F1 | 0.4157 |
| true positives | 37 |
| true negatives | 114 |
| false positives | 11 |
| false negatives | 93 |

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[114, 11],
 [ 93, 37]]
```

## Interpretation

The first anomaly baseline is now runnable and defensible as a baseline, but the result is weak for recall-critical tyre inspection because `93` of `130` defective test tyres were missed at the selected threshold.

This does not invalidate anomaly detection. It means the first frozen global-embedding Mahalanobis baseline is not enough by itself.

## Outputs

- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/metadata.json`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/metrics_test.json`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/predictions_test.csv`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/false_negatives_test.csv`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/false_positives_test.csv`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/confusion_matrix_test.png`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/score_distribution_test.png`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/pr_curve_test.png`

## Next Experiment

1. Inspect false negatives visually.
2. Sweep thresholds beyond the FPR-constrained policy to understand recall/precision trade-offs.
3. Compare Mahalanobis with kNN using the already implemented scorer.
4. Consider patch-level or multi-crop embeddings if defects are local and diluted by global pooling.
