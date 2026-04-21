# Anomaly Patch Layer Status

## Current Status
Completed the lower/mid-level ResNet50 patch-aware follow-up on D1 anomaly manifests.

## Executed Variants
| Variant | Feature source | Recall | FN | FP | Status |
|---|---|---:|---:|---:|---|
| `resnet50_knn_threshold_sweep_reference` | pooled penultimate embedding | 0.9462 | 7 | 36 | Reused current high-recall reference. |
| `resnet50_layer3_patch_knn` | ResNet50 layer3 feature map | 0.1231 | 114 | 12 | Executed, not useful for recall. |
| `resnet50_layer3_patch_knn_threshold_sweep` | ResNet50 layer3 feature map | 0.3385 | 86 | 41 | Executed, not useful for recall. |
| `resnet50_layer2_layer3_patch_knn_threshold_sweep` | concatenated layer2+layer3 maps | 0.2154 | 102 | 38 | Executed, not useful for recall. |

## Main Finding
Lower/mid-level patch descriptors with robust score normalization did not improve tyre anomaly detection. The current best recall-oriented candidate remains `resnet50_knn_threshold_sweep`.

## Interpretation
The result suggests that the current patch-memory formulation does not separate defect-like local evidence from normal tyre texture well enough. The problem may be score calibration, memory-bank construction, patch aggregation, feature choice, or the absence of spatial localization labels.

## Next Step
Before adding more patch-aware variants, inspect patch-score distributions and nearest-neighbor examples. The next useful prompt should be diagnostic rather than another larger benchmark.
