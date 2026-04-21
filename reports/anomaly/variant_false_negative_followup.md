# Variant False-Negative Follow-Up

Updated: 2026-04-20

## Baseline Comparison

| Variant | Test Recall | False Negatives | False Positives | Change vs Baseline FN |
|---|---:|---:|---:|---:|
| `resnet18_mahalanobis_reference` | 0.2846 | 93 | 11 | 0 |
| `resnet18_knn` | 0.5231 | 62 | 9 | -31 |
| `resnet18_threshold_sweep` | 0.5769 | 55 | 32 | -38 |
| `resnet50_mahalanobis` | 0.7077 | 38 | 20 | -55 |
| `resnet50_knn` | 0.8231 | 23 | 15 | -70 |
| `resnet18_patch_grid_knn` | 0.3462 | 85 | 12 | -8 |

## Interpretation

The main false-negative reduction came from stronger ResNet50 embeddings plus kNN scoring. This suggests the original baseline was limited by both feature quality and the single-Gaussian Mahalanobis assumption.

The ResNet18 patch-grid variant did not meaningfully reduce false negatives. The likely reason is that a simple full-image-plus-quadrants scheme is still too coarse and uses the weaker ResNet18 backbone. Local defects may require finer patch extraction, multi-scale crops, or a backbone with stronger local texture features.

## Did Local Features Help?

Not in this first implementation. `resnet18_patch_grid_knn` missed `85` defects, compared with `62` for pooled `resnet18_knn` and `23` for pooled `resnet50_knn`.

This does not invalidate local-feature methods. It means the first low-risk local-feature attempt was not the right one.

## Remaining Follow-Up

Review the `23` remaining false negatives from `resnet50_knn` and classify them by likely cause:

- small/local crack
- low contrast defect
- defect near image edge
- unusual but normal-looking viewpoint
- possible label noise

If misses are still small or local, the next patch-aware method should use ResNet50 or a dedicated patch embedding strategy rather than simple ResNet18 quadrants.
