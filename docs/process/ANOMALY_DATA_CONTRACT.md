# Anomaly Data Contract

Updated: 2026-04-19

This document defines the manifest contract for TyreVisionX anomaly-detection work.

## Canonical Manifest Location

Use `data/manifests/` for anomaly and future supervised manifests.

`data/processed/` is legacy compatibility only. Existing legacy scripts may still use it, but new anomaly code should not.

## Manifest Roles

The anomaly baseline uses three roles:

- `normal_train`: good/normal images only
- `val_mixed`: held-out good and defect images for threshold calibration
- `test_mixed`: held-out good and defect images for final evaluation

Default D1 outputs:

- `data/manifests/D1_anomaly_train_normal.csv`
- `data/manifests/D1_anomaly_val_mixed.csv`
- `data/manifests/D1_anomaly_test_mixed.csv`

## Required Columns

| Column | Meaning |
|---|---|
| `image_path` | Path to image, preferably repo-relative |
| `target` | Generic anomaly target; `0 = normal`, `1 = anomaly` |
| `label` | Original supervised label when available; for D1, `0 = good`, `1 = defect` |
| `label_str` | Human-readable source label, such as `good` or `defect` |
| `split` | Role split, such as `train`, `val`, or `test` |
| `is_normal` | Boolean; `true` for normal training/good samples |
| `source_dataset` | Source dataset identifier, such as `D1` |
| `product_type` | Product category, currently `tyre` |

## Generic vs Tyre-Specific Parts

Generic:
- `image_path`
- `target`
- `split`
- `is_normal`
- `source_dataset`
- `product_type`
- normal-only training and mixed validation/test pattern

Tyre-specific for now:
- source label names: `good`, `defect`
- source dataset ID: `D1`
- product type: `tyre`
- current image folder assumptions

## Why This Contract Supports Future Products

Future products can reuse the anomaly pipeline by changing manifests:

- keep `target = 0` for normal examples
- keep `target = 1` for anomalous examples
- set `product_type` to the product name
- set `source_dataset` to the dataset ID

The anomaly model should not need tyre-specific folder names once manifests are created.

