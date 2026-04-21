# Patch-Aware False Negative Follow-Up

## Summary
The patch-aware ResNet50 feature-map variants did not reduce false negatives relative to the current high-recall reference.

| Variant | Test Recall | False Negatives | False Positives | Interpretation |
|---|---:|---:|---:|---|
| `resnet50_knn_threshold_sweep_reference` | 0.9462 | 7 | 36 | Current best recall-oriented reference. |
| `resnet50_featuremap_patch_knn` | 0.1000 | 117 | 6 | Too strict; missed most defects. |
| `resnet50_featuremap_patch_knn_threshold_sweep` | 0.4615 | 70 | 30 | Threshold sweep helped but still far below reference. |
| `resnet50_patchcore_lite` | 0.2385 | 99 | 11 | Reduced memory/top-3 scoring did not help. |

## Overlap With High-Recall Reference
Using `resnet50_featuremap_patch_knn_threshold_sweep` as the strongest patch-aware variant:

- Fixed from reference: `2`
- Still missed by both: `5`
- New misses introduced by patch-aware variant: `65`

## Interpretation
Feature-map patch memory is conceptually better than resized image patches, but this first layer4-only implementation did not work well on D1. The likely issue is that layer4 descriptors are too high-level and too spatially coarse for small tyre surface defects. The nearest-neighbor patch scores also appear poorly separated, with many scores near `1.0`.

## Next Practical Action
Do not replace the high-recall pooled reference with the current patch-aware variants. If patch-aware work continues, test layer3 descriptors, layer2+layer3 combinations, better memory reduction, and stronger score normalization before drawing a final conclusion about PatchCore-style methods.
