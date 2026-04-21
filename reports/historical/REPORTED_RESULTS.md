# Historical Reported Results

Updated: 2026-04-19

This file preserves prior reported TyreVisionX results without overstating them. Unless marked otherwise, these are historical reported results from earlier repository work and were not recomputed by the cleanup prompt.

## Result Status Labels

- Historical reported result: recorded in existing reports, progress logs, README text, or artifacts from earlier work.
- Recomputed in current cleaned repo: rerun after this cleanup using the cleaned canonical pipeline.
- Pending recomputation: not rerun after cleanup.

## Day 3 Frozen ResNet50 Baseline

Source: `README.md`, `PROGRESS_LOG.md`, and `reports/day3_baseline_observations.md`.

Status: historical reported result; pending recomputation in the cleaned canonical pipeline.

Reported setup:
- model: frozen or partially unfrozen ResNet50 feature extractor
- split: D1 test split
- training: 3-epoch comparison noted in prior reports

Reported metrics:
- accuracy: `0.9804`
- defect recall: `0.9692`
- defect F1: `0.9805`
- AUROC: `0.9985`

Notes:
- These metrics are useful historical context.
- They should not be presented as newly verified by this cleanup.
- The result should be recomputed or replaced by a canonical `src/train.py` / `src/evaluate.py` run before thesis-level claims.

## Day 5 SimpleCNN Regularization Comparison

Source: `reports/day5_regularization_observations.md`, `PROGRESS_LOG.md`, and available `artifacts/day5/` summaries.

Status: historical reported result; available local artifacts exist, but not recomputed by this cleanup prompt.

Short-run reported conclusion:
- baseline had the strongest recall in a 3-epoch seed-42 run
- augmentation improved precision and false-positive behavior
- BatchNorm and Dropout needed stabilization

Long-run multi-seed reported conclusion:
- baseline had the highest mean recall but high seed variance
- augmentation had the most stable precision/AUPRC trade-off

Long-run reported aggregate:

| Model | Recall mean | Precision mean | FN mean | FP mean | AUPRC mean |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9436 | 0.7458 | 7.33 | 43.00 | 0.9017 |
| + Augmentation | 0.9410 | 0.7787 | 7.67 | 35.00 | 0.9468 |
| + BatchNorm | 0.8641 | 0.7660 | 17.67 | 39.00 | 0.8873 |
| + Dropout | 0.8949 | 0.7370 | 13.67 | 43.67 | 0.8875 |

## Dataset Counts Previously Reported

Source: `reports/dataset_report.md` and earlier progress notes.

Status: historical reported result; pending recomputation in this cleanup.

Reported counts:
- total images: `1698`
- good: `832`
- defect: `866`
- defect ratio: approximately `0.5112`

Reported split counts:

| Split | Defect | Good |
|---|---:|---:|
| train | 606 | 582 |
| val | 130 | 125 |
| test | 130 | 125 |

## Current Cleanup Recomputed Results

No model training or dataset-stat recomputation was performed during this cleanup prompt.

