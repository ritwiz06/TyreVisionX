# Day 5 Regularization and Stability Observations

## Objective
Improve the tiny CNN baseline with augmentation, BatchNorm, and Dropout, then compare defect-focused metrics.

## Short-Run Summary (3-epoch, seed=42)
- Baseline was best on this short run for recall-critical behavior.
- It produced recall `0.9923` with only `1` false negative.
- `+ Augmentation` improved precision and reduced false positives, with a small recall drop.
- `+ BatchNorm` and `+ Dropout` were stabilized after fixes but did not beat baseline in this short-run setup.
- See the later "Additional Run Details" section for the updated 10-epoch multi-seed conclusion.

## Run Settings
- Dataset manifest: `data/processed/D1_manifest.csv`
- Split for final comparison: `test`
- Image size: `224`
- Epochs: `3`
- Batch size: `16`
- Optimizer: `AdamW`
- Seed: `42`

## Comparison Table (After Fix)
| Model | Recall | Precision | FN count | FP count |
|---|---:|---:|---:|---:|
| Baseline | 0.9923 | 0.7247 | 1 | 49 |
| + Augmentation | 0.9846 | 0.7665 | 2 | 39 |
| + BatchNorm | 0.9077 | 0.7329 | 12 | 43 |
| + Dropout | 0.9538 | 0.6392 | 6 | 70 |

Source: `artifacts/day5/step3_step4_comparison.csv`

## Key Findings
- `+ Augmentation` improved precision and reduced false positives while keeping recall high.
- BatchNorm/Dropout collapse was fixed by switching to logits output + `BCEWithLogitsLoss` and lowering LR to `3e-4`.
- `+ BatchNorm` is now stable but does not beat `+ Augmentation` on this 3-epoch setup.
- `+ Dropout (0.3)` improved recall over the failed state but increased false positives noticeably.

## Error Analysis (Current)
- Baseline false negatives: `1` sample (`artifacts/day5/step3_step4_baseline/false_negatives.csv`)
- +Aug false negatives: `2` samples (`artifacts/day5/step3_step4_aug/false_negatives.csv`)
- +BN false negatives: `12` samples (`artifacts/day5/step3_step4_aug_batchnorm_fix/false_negatives.csv`)
- +Dropout false negatives: `6` samples (`artifacts/day5/step3_step4_aug_batchnorm_dropout03_fix/false_negatives.csv`)

Use notebook `notebooks/day5_regularization_bn_dropout.ipynb` to inspect:
- False negatives on tire images
- False positives on tire images
- Per-run prediction behavior and confidence patterns

## Hypothesis for BN/Dropout Failure
The previous failure came from an unstable optimization path (sigmoid-probability head + BCE). Using logits with `BCEWithLogitsLoss` plus lower LR stabilized BatchNorm and Dropout runs.

## Next Actions
1. Try BatchNorm runs with lower LR (`3e-4` or `1e-4`) and/or warmup.
2. Switch to `BCEWithLogitsLoss` and remove output sigmoid for improved numerical stability.
3. Add threshold tuning for recall-critical operation after selecting the best stable model.

---

## Additional Run Details — Longer Training + Threshold Sweep + Multi-Seed

### Protocol (New)
- Training horizon: `10` epochs with early stopping (`patience=3`)
- Seeds: `7`, `123`, `999`
- Models: Baseline, +Augmentation, +BatchNorm, +Dropout
- Threshold selection: sweep `0.10` to `0.90` (step `0.01`) on **validation**; choose max precision under recall >= `0.99` (fallback: best recall)
- Final metrics reported on **test** split at chosen threshold
- PR curves and AUPRC computed on test split

Artifacts:
- `artifacts/day5/longrun_seed_sweep/summary_runs.csv`
- `artifacts/day5/longrun_seed_sweep/summary_by_model.csv`
- `artifacts/day5/longrun_seed_sweep/pr_curves_*.png`

### Aggregated Results (mean +/- std across 3 seeds)
| Model | Recall | Precision | FN | FP | AUPRC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9436 +/- 0.0844 | 0.7458 +/- 0.0533 | 7.33 +/- 10.97 | 43.00 +/- 14.73 | 0.9017 +/- 0.0114 |
| + Augmentation | 0.9410 +/- 0.0291 | 0.7787 +/- 0.0301 | 7.67 +/- 3.79 | 35.00 +/- 7.00 | 0.9468 +/- 0.0079 |
| + BatchNorm | 0.8641 +/- 0.1039 | 0.7660 +/- 0.1302 | 17.67 +/- 13.50 | 39.00 +/- 28.51 | 0.8873 +/- 0.0478 |
| + Dropout | 0.8949 +/- 0.1089 | 0.7370 +/- 0.0721 | 13.67 +/- 14.15 | 43.67 +/- 18.93 | 0.8875 +/- 0.0157 |

### Interpretation (Long-Run)
- Baseline has the highest **mean recall** by a small margin, but is unstable across seeds (high recall std and one seed with low recall `0.8462`).
- +Augmentation is the most **stable** and gives the best precision/FP trade-off plus highest AUPRC.
- +BatchNorm and +Dropout remain less reliable for recall-critical deployment on this setup.

### Updated Decision
- If priority is strict defect recall with acceptable variance: Baseline is competitive but unstable.
- If priority is robust deployment behavior across seeds and better precision/FP control: **+Augmentation is the better current candidate**.
- Recommended next step: keep +Augmentation and perform operating-threshold calibration on validation for target recall bands (`>=0.99`, `>=0.97`, `>=0.95`) before selecting production threshold.
