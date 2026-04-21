# Anomaly Patch-Aware Status

## Current Status
Completed a controlled ResNet50 feature-map patch-aware benchmark on D1 anomaly manifests.

## Executed Variants
| Variant | Test Recall | FN | FP | Status |
|---|---:|---:|---:|---|
| `resnet50_knn_threshold_sweep_reference` | 0.9462 | 7 | 36 | Current high-recall reference. |
| `resnet50_featuremap_patch_knn` | 0.1000 | 117 | 6 | Executed, not useful for recall. |
| `resnet50_featuremap_patch_knn_threshold_sweep` | 0.4615 | 70 | 30 | Executed, still far below reference. |
| `resnet50_patchcore_lite` | 0.2385 | 99 | 11 | Executed, not useful for recall. |

## Main Finding
The high-recall pooled ResNet50 kNN threshold-sweep reference remains the best current anomaly candidate. The first feature-map patch-memory implementation did not reduce false negatives.

## Limitation
This benchmark used ResNet50 `layer4` descriptors. These are spatially coarse and may be too semantic for small local tyre defects. This result should not be interpreted as a rejection of all patch-aware methods.

## Lower/Mid-Level Follow-Up
A second patch-aware follow-up tested lower/mid-level feature maps and robust score normalization.

| Variant | Test Recall | FN | FP | Status |
|---|---:|---:|---:|---|
| `resnet50_knn_threshold_sweep_reference` | 0.9462 | 7 | 36 | Still best recall-oriented candidate. |
| `resnet50_layer3_patch_knn` | 0.1231 | 114 | 12 | Executed, not useful for recall. |
| `resnet50_layer3_patch_knn_threshold_sweep` | 0.3385 | 86 | 41 | Executed, not useful for recall. |
| `resnet50_layer2_layer3_patch_knn_threshold_sweep` | 0.2154 | 102 | 38 | Executed, not useful for recall. |

Conclusion: layer3 and layer2+layer3 patch memory did not improve the D1 tyre anomaly benchmark. The issue now appears to be patch-memory score separation rather than only the coarseness of `layer4`.

## Next Step
Keep `resnet50_knn_threshold_sweep` as the current recall-oriented reference. If patch-aware work continues, inspect patch-score distributions and patch-level nearest-neighbor behavior before adding more variants.
