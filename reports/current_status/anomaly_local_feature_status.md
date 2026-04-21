# Anomaly Local Feature Status

## Current Status
Completed a controlled ResNet50 local-feature benchmark on D1 anomaly manifests.

## Executed Variants
| Variant | Status | Test Recall | FN | FP |
|---|---|---:|---:|---:|
| `resnet50_knn_reference` | Reused existing result | 0.8231 | 23 | 15 |
| `resnet50_knn_threshold_sweep` | Executed | 0.9462 | 7 | 36 |
| `resnet50_multicrop_knn` | Executed | 0.8231 | 23 | 20 |
| `resnet50_patch_grid_knn_fine` | Executed | 0.4692 | 69 | 19 |

## Main Finding
Threshold refinement helped more than the simple local-feature variants. The best current recall-oriented candidate is `resnet50_knn_threshold_sweep`, selected using validation data only.

## Limitation
The current patch-grid method resizes small patches to full ResNet50 input size and compares those embeddings against local crop embeddings. This simple design appears too noisy for reliable tyre defect detection.

## Next Step
The first principled patch-aware method has now been evaluated. It did not beat the threshold-refined ResNet50 kNN result. See `reports/current_status/anomaly_patch_aware_status.md`.
