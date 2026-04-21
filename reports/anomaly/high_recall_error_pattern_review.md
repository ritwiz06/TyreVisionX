# High-Recall Error Pattern Review

## Scope
This review covers the current high-recall reference, `resnet50_knn_threshold_sweep`, using the exported test-set review assets:

- `reports/anomaly/high_recall_false_negative_review_table.csv`
- `reports/anomaly/high_recall_false_positive_review_table.csv`
- `reports/anomaly/high_recall_false_negative_contact_sheet.png`
- `reports/anomaly/high_recall_false_positive_contact_sheet.png`

The review is visual and qualitative. It should guide the next model design, but it is not a replacement for metric-based evaluation.

## Current Error Counts
| Error type | Count | Meaning |
|---|---:|---|
| False negatives | 7 | Defective tyres scored below the anomaly threshold. These are the most safety-critical misses. |
| False positives | 36 | Good-labelled tyres scored above the anomaly threshold. These increase review or rejection burden. |

## False-Negative Patterns
The 7 false negatives are all close to the selected threshold (`0.8169`). Several are only slightly below it, which means small threshold changes could recover them but would likely increase false positives.

Visible patterns from the contact sheet:

- Defect evidence appears subtle or low contrast in multiple examples.
- Some images are close-ups where the difference between normal grooves, scuffs, and actual defects is visually ambiguous.
- Several examples show sidewall or tread texture where the defect cue is local rather than whole-image obvious.
- At contact-sheet scale, not every defect is visually obvious. Full-resolution inspection may be needed before claiming exact causes.

## False-Positive Patterns
The 36 false positives are good-labelled images with scores above the recall-priority threshold.

Visible patterns from the contact sheet:

- Many images contain strong tread geometry, deep grooves, sidewall text, or high-frequency texture.
- Some examples include dirt, wear, glare, shadows, or unusual close-up framing.
- A few examples may be label-risk cases: they are labelled good, but visually contain marks or wear that could be considered abnormal under a stricter inspection definition.
- The false-positive load is an expected tradeoff of the recall-priority threshold sweep.

## Implication for Patch-Aware Design
The error review justifies testing local evidence, but it does not prove that any patch method will help. The remaining misses look local/subtle, while many false positives are also local/high-texture. A patch-aware method must separate actual defect-like local evidence from normal tyre texture, dirt, lettering, and tread structure.

The lower/mid-level patch-memory follow-up did not solve this. This suggests the next patch-aware work should inspect patch-score distributions and memory-neighbor behavior before adding more feature layers or aggregation rules.

## Current Decision
Keep `resnet50_knn_threshold_sweep` as the recall-oriented reference. Do not replace it with the current patch-aware variants.
