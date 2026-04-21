# Robustness Comparison Summary

Updated: 2026-04-20

## Purpose

Compare clean-trained anomaly variants and the mild noise-robust ResNet50+kNN variant under realistic corruptions. Thresholds are selected on clean validation only.

## Aggregate Test Robustness

| Variant | Mean Recall | Min Recall | Max FN | Mean FN | Mean FP | Mean AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| `resnet18_knn_control` | 0.5392 | 0.4923 | 66 | 59.9 | 10.4 | 0.8414 |
| `resnet50_knn` | 0.8115 | 0.7846 | 28 | 24.5 | 16.8 | 0.9208 |
| `resnet50_knn_noise_robust` | 0.8177 | 0.7923 | 27 | 23.7 | 11.6 | 0.9306 |
| `resnet50_mahalanobis` | 0.7300 | 0.7077 | 38 | 35.1 | 20.5 | 0.8665 |

## Clean Test Reference

| Variant | Clean Recall | Clean FN | Clean FP | Clean AUROC | Clean AUPRC |
|---|---:|---:|---:|---:|---:|
| `resnet50_knn` | 0.8231 | 23 | 15 | 0.9298 | 0.9339 |
| `resnet50_mahalanobis` | 0.7077 | 38 | 20 | 0.8676 | 0.8800 |
| `resnet18_knn_control` | 0.5231 | 62 | 9 | 0.8546 | 0.8394 |
| `resnet50_knn_noise_robust` | 0.8231 | 23 | 13 | 0.9407 | 0.9383 |

## Interpretation

- `resnet50_knn` degrades moderately under realistic corruptions, with the largest recall drops under medium blur and darker lighting.
- The mild noise-robust variant preserves clean recall (`0.8231`) and reduces clean false positives from `15` to `13`.
- The robust variant improves or preserves recall on blur and darker/contrast-lower conditions, but does not eliminate all misses.
- Robustness work helps, but the remaining false negatives still justify patch-aware or local-feature work.
