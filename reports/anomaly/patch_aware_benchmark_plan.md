# Patch-Aware Benchmark Plan

## Objective
Evaluate whether ResNet50 convolutional feature-map patch descriptors can reduce the remaining false negatives from the current high-recall reference, `resnet50_knn_threshold_sweep`.

## Why the Previous Patch Grid Failed
The previous fine patch-grid method cropped small image regions, resized each patch to full image size, and then passed each resized patch through the same full-image embedding model. This can distort tyre texture and create a distribution mismatch: tiny patches do not look like full tyre images, even when they are normal.

## Why Feature-Map Patches Are More Principled
Convolutional feature maps already preserve spatial locations. A feature-map cell describes a local receptive field in the original image while still being computed inside the normal ResNet pipeline. This avoids resizing tiny crops into artificial full images.

## Variants
| Variant | Purpose |
|---|---|
| `resnet50_knn_threshold_sweep_reference` | Current best recall-oriented pooled-embedding reference. |
| `resnet50_featuremap_patch_knn` | Feature-map patch memory with stricter validation FPR policy. |
| `resnet50_featuremap_patch_knn_threshold_sweep` | Same patch descriptors with high-recall validation threshold refinement. |
| `resnet50_patchcore_lite` | Reduced memory-bank variant with top-3 patch-score aggregation. |

## Evaluation Rules
- Fit memory bank only on `D1_anomaly_train_normal.csv`.
- Select thresholds only on `D1_anomaly_val_mixed.csv`.
- Evaluate once on `D1_anomaly_test_mixed.csv`.
- Do not tune on test.
- Report false negatives and false positives explicitly.

## Decision Criteria
Patch-aware scoring is useful only if it reduces false negatives enough to justify added complexity, without making false positives unmanageable.
