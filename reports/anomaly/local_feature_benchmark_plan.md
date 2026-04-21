# Local Feature Benchmark Plan

## Objective
The current best tyre anomaly variant is `resnet50_knn`, but it still misses too many defective tyres for a recall-critical inspection setting. This benchmark tests whether local evidence reduces false negatives compared with whole-image pooled embeddings.

## Hypothesis
Some tyre defects are small, low contrast, or located near only one region of the image. A whole-image pooled embedding can dilute that local evidence. Multi-crop and patch-grid scoring give small regions their own anomaly scores before aggregating back to the image level.

## Variants
| Variant | Status Goal | Why It Exists |
|---|---|---|
| `resnet50_knn_reference` | Reuse previous result | Clean reference for comparison. |
| `resnet50_knn_threshold_sweep` | Execute | Tests whether validation-only threshold refinement alone reduces false negatives. |
| `resnet50_multicrop_knn` | Execute | Scores full image plus center/corner crops to recover localized defects. |
| `resnet50_patch_grid_knn_fine` | Execute | Scores full image plus a 3x3 local grid for finer regional evidence. |

## Evaluation Rules
- Fit only on `D1_anomaly_train_normal.csv`.
- Select thresholds on `D1_anomaly_val_mixed.csv` only.
- Apply the selected threshold once to `D1_anomaly_test_mixed.csv`.
- Do not tune thresholds on test.
- Emphasize defect/anomaly recall and false-negative count.

## Expected Outputs
- `reports/anomaly/local_feature_model_comparison.csv`
- `reports/anomaly/local_feature_model_comparison.md`
- Per-variant artifacts under `artifacts/anomaly/local_features/`
- False-negative overlap analysis comparing the reference with the top local-feature variant.

## Interpretation
If local-feature variants reduce false negatives without exploding false positives, local evidence becomes the next practical TyreVisionX direction. If they do not help, the next step should be better feature extractors or true patch-aware anomaly methods rather than more global pooling.
