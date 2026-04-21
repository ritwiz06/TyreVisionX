# Anomaly Robustness Guide

Updated: 2026-04-20

## Why Robustness Matters

Tyre images can vary because of lighting, camera noise, slight blur, or compression. A useful anomaly baseline should not collapse when these realistic nuisance changes appear.

## Evaluation Corruption vs Training Augmentation

Evaluation corruption tests how a fixed model behaves under realistic image changes. Training augmentation changes the normal-only training embeddings to include mild variations.

TyreVisionX keeps these separate so the benchmark stays interpretable.

## Clean Validation Threshold

The main robustness benchmark uses the threshold already selected on clean validation data. This prevents corrupted test tuning and shows the real robustness gap.

## Why Not Aggressive Noise

Too much noise can make defective regions disappear or make normal texture look anomalous. In good-only anomaly detection, overly broad augmentation can teach the model that abnormal-looking images are normal.

## How To Interpret Results

- Small recall gap: model is reasonably robust to that corruption.
- Large recall gap: corruption makes defects harder to score as anomalous.
- Rising false positives: normal tyres become incorrectly rejected.
- Lower AUROC/AUPRC: score ranking is degraded, not only the threshold.

For TyreVisionX, false negatives are the priority because missed defects are safety-critical.
