# Patch False Negative Overlap Analysis

## Compared Runs
- Reference: `resnet50_knn_threshold_sweep_reference`
- Patch-aware candidate: `resnet50_featuremap_patch_knn_threshold_sweep`

## Counts
| Group | Count |
|---|---:|
| Fixed by patch-aware candidate | 2 |
| Still missed by both | 5 |
| New misses introduced by patch-aware candidate | 65 |

## Interpretation
The patch-aware candidate fixed 2 of the 7 high-recall reference misses, but introduced 65 new misses. That is not acceptable for the recall-critical anomaly track.

The overlap result suggests that this first feature-map patch memory method changes the decision boundary substantially, but not in a useful direction for D1 tyre defects.

## Detailed Outputs
- `reports/anomaly/patch_false_negative_overlap/false_negatives_fixed_by_candidate.csv`
- `reports/anomaly/patch_false_negative_overlap/false_negatives_still_missed_by_both.csv`
- `reports/anomaly/patch_false_negative_overlap/false_negatives_new_in_candidate.csv`
- `reports/anomaly/patch_false_negative_overlap/false_negative_overlap_contact_sheet.png`
