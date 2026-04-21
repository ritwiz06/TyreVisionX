# High-Recall Error Review

Source run: `artifacts/anomaly/local_features/resnet50_knn_threshold_sweep`

| Error Type | Count | Review Table | Contact Sheet |
|---|---:|---|---|
| False negatives | 7 | `reports/anomaly/high_recall_false_negative_review_table.csv` | `reports/anomaly/high_recall_false_negative_contact_sheet.png` |
| False positives | 36 | `reports/anomaly/high_recall_false_positive_review_table.csv` | `reports/anomaly/high_recall_false_positive_contact_sheet.png` |

## Interpretation Notes
- False negatives are defective tyres still scored below the high-recall threshold.
- False positives are good tyres scored above the high-recall threshold.
- The tables and contact sheets are review aids only; visual failure causes require human inspection.
- Likely patterns to check: subtle cracks, edge defects, glare, low contrast, unusual tread texture, or possible label ambiguity.
