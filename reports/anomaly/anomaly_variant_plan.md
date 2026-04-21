# Anomaly Variant Plan

Updated: 2026-04-19

## Purpose

The first D1 anomaly baseline ran successfully, but test anomaly recall was low (`0.2846`) and false negatives were high (`93`). The next variants should stay simple and defensible before adding heavier methods.

## Variant Comparison

| Variant | Status | Why Try It | Evaluation Rule |
|---|---|---|---|
| Mahalanobis baseline | implemented and executed | Current reference point using global ResNet18 embeddings. | Keep existing validation-selected threshold and test-once result. |
| kNN scorer | planned, config added | kNN may capture local neighborhood density better than a single Gaussian covariance assumption. | Fit on normal train embeddings, calibrate on validation, evaluate test once. |
| Threshold sweep | planned, config added | Current policy controlled normal FPR but missed many defects. Sweep recall/FPR trade-offs on validation. | Choose threshold on validation only; apply selected threshold once to test. |
| Multi-crop / patch-aware features | planned later | Global pooling may dilute small cracks or edge-local defects. | Add only after reviewing false negatives and simple scorer variants. |

## Evaluation Rules

- Train/fit only on `D1_anomaly_train_normal.csv`.
- Calibrate thresholds only on `D1_anomaly_val_mixed.csv`.
- Evaluate once on `D1_anomaly_test_mixed.csv`.
- Emphasize defect/anomaly recall and false-negative count.
- Report normal false-positive rate because excessive rejection of good tyres also matters.
- Do not compare variants using test-set threshold tuning.

## Current Baseline Result

| Metric | Value |
|---|---:|
| AUROC | 0.7194 |
| AUPRC | 0.7212 |
| anomaly recall | 0.2846 |
| anomaly precision | 0.7708 |
| normal FPR | 0.0880 |
| false negatives | 93 |
| false positives | 11 |

## Next Action

Review the false-negative contact sheet and table first. If misses are visually small/local defects, prioritize patch-aware or multi-crop embeddings after kNN and threshold sweep.
