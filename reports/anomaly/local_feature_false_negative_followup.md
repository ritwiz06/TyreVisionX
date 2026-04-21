# Local Feature False Negative Follow-Up

## Summary
The controlled local-feature benchmark did not show that local aggregation is the best immediate fix for clean false negatives.

| Variant | Test Recall | False Negatives | False Positives | Interpretation |
|---|---:|---:|---:|---|
| `resnet50_knn_reference` | 0.8231 | 23 | 15 | Current clean reference. |
| `resnet50_knn_threshold_sweep` | 0.9462 | 7 | 36 | Best false-negative reduction, but with higher false positives. |
| `resnet50_multicrop_knn` | 0.8231 | 23 | 20 | No FN improvement over reference. |
| `resnet50_patch_grid_knn_fine` | 0.4692 | 69 | 19 | Worse recall; current patch-grid scoring is not suitable. |

## What Helped
Validation-only threshold refinement helped most. It fixed 16 of the 23 reference false negatives and introduced no new false negatives relative to the reference, but it increased false positives from 15 to 36.

## What Did Not Help
`resnet50_multicrop_knn` did not reduce the total false-negative count. It fixed 8 reference misses but introduced 8 different misses, leaving the total at 23 and increasing false positives to 20.

`resnet50_patch_grid_knn_fine` substantially degraded recall. The likely issue is distribution mismatch: resized tiny patches do not behave like full tyre images in the ResNet50 embedding space, so the kNN score becomes less reliable.

## Implication
The immediate TyreVisionX direction should be threshold/calibration refinement plus a more principled patch-aware method, not the current simple resized patch-grid approach.

## Supporting Outputs
- Threshold-sweep overlap: `reports/anomaly/false_negative_overlap/`
- Multicrop overlap: `reports/anomaly/false_negative_overlap_multicrop/`
- Comparison CSV: `reports/anomaly/local_feature_model_comparison.csv`
