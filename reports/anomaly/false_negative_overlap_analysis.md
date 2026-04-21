# False Negative Overlap Analysis

## Reference vs Threshold-Sweep Candidate
Compared `resnet50_knn_reference` against `resnet50_knn_threshold_sweep`.

| Group | Count |
|---|---:|
| Fixed by threshold sweep | 16 |
| Still missed by both | 7 |
| New misses introduced | 0 |

This confirms that threshold refinement reduced false negatives without introducing new false-negative cases relative to the reference. The cost is higher false positives.

Detailed outputs:
- `reports/anomaly/false_negative_overlap/false_negatives_fixed_by_candidate.csv`
- `reports/anomaly/false_negative_overlap/false_negatives_still_missed_by_both.csv`
- `reports/anomaly/false_negative_overlap/false_negatives_new_in_candidate.csv`
- `reports/anomaly/false_negative_overlap/false_negative_overlap_contact_sheet.png`

## Reference vs Multicrop
Compared `resnet50_knn_reference` against `resnet50_multicrop_knn`.

| Group | Count |
|---|---:|
| Fixed by multicrop | 8 |
| Still missed by both | 15 |
| New multicrop misses | 8 |

Multicrop changed which tyres were missed but did not reduce the total false-negative count.

Detailed outputs:
- `reports/anomaly/false_negative_overlap_multicrop/false_negatives_fixed_by_candidate.csv`
- `reports/anomaly/false_negative_overlap_multicrop/false_negatives_still_missed_by_both.csv`
- `reports/anomaly/false_negative_overlap_multicrop/false_negatives_new_in_candidate.csv`
- `reports/anomaly/false_negative_overlap_multicrop/false_negative_overlap_contact_sheet.png`

## Interpretation
The strongest immediate improvement is calibration/thresholding, not naive local crop scoring. The still-missed threshold-sweep cases should be visually reviewed before choosing the next patch-aware method.
