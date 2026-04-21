# Best Variant Recommendation

Updated: 2026-04-20

## Recommendation

Use `resnet50_knn` as the next TyreVisionX anomaly baseline candidate.

This recommendation is tyre-first and conservative. It is based on validation-selected thresholds and test-once reporting, not on test-threshold tuning.

## Why This Variant

`resnet50_knn` produced the strongest executed result in the controlled benchmark:

| Metric | Value |
|---|---:|
| Test AUROC | 0.9298 |
| Test AUPRC | 0.9339 |
| Test anomaly recall | 0.8231 |
| Test anomaly precision | 0.8770 |
| Test normal FPR | 0.1200 |
| Test false negatives | 23 |
| Test false positives | 15 |

Compared with the original ResNet18 + Mahalanobis baseline, this reduces false negatives from `93` to `23`.

## Why Not Claim Solved

The recall is much better, but `23` missed defects remain. For recall-critical tyre inspection, this is still not enough for a production claim. The next work should inspect the remaining false negatives and test patch-aware or multi-crop variants with stronger backbones.

## Why Not Choose Threshold Sweep Alone

The ResNet18 threshold sweep improved recall from `0.2846` to `0.5769`, but it increased normal false positives to `32` and still missed `55` defects. This shows threshold policy matters, but better embeddings and scoring matter more.

## Why Not Choose The Patch-Grid Variant Yet

The low-risk ResNet18 patch-grid kNN variant did not help in this run. It reduced false negatives only from `93` to `85`, worse than pooled ResNet18 kNN. This does not rule out patch-aware methods; it means this simple quadrant-grid implementation is not sufficient.

## Next Best Experiment

1. Inspect false negatives from `artifacts/anomaly/benchmark/resnet50_knn/false_negatives_test.csv`.
2. Run a validation-only threshold sweep for `resnet50_knn` with recall-focused operating points.
3. Try patch-aware scoring with ResNet50 features instead of the current ResNet18 patch-grid variant.
4. Keep test-set threshold tuning forbidden.
