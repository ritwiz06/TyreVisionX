# False Negative Overlap Analysis

Reference run: `resnet50_knn_threshold_sweep_reference`
Candidate run: `resnet50_featuremap_patch_knn_threshold_sweep`

| Group | Count | Output |
|---|---:|---|
| Fixed by candidate | 2 | `reports/anomaly/patch_false_negative_overlap/false_negatives_fixed_by_candidate.csv` |
| Still missed by both | 5 | `reports/anomaly/patch_false_negative_overlap/false_negatives_still_missed_by_both.csv` |
| New candidate misses | 65 | `reports/anomaly/patch_false_negative_overlap/false_negatives_new_in_candidate.csv` |

Contact sheet status: Wrote `reports/anomaly/patch_false_negative_overlap/false_negative_overlap_contact_sheet.png`.

## Interpretation Notes
- Fixed cases are defects the reference missed but the candidate caught.
- Still-missed cases are the highest priority for visual review.
- New candidate misses show any regression introduced by the local-feature method.
- This analysis compares decisions only; it does not prove the visual cause without human inspection.
