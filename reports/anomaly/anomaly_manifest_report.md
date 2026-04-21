# Anomaly Manifest Report

Status: generated

Source manifest: `/Users/ritik/Documents/Project TDA/TyreVisionX/data/manifests/D1_tyrenet_manifest.csv`

Canonical decision: anomaly manifests are written under `data/manifests/`.
`data/processed/` remains legacy compatibility only.

Image paths unresolved during generation: `0`

## normal_train

Rows: `582`

| Target | Count |
|---|---:|
| normal | 582 |
| anomaly | 0 |

## val_mixed

Rows: `255`

| Target | Count |
|---|---:|
| normal | 125 |
| anomaly | 130 |

## test_mixed

Rows: `255`

| Target | Count |
|---|---:|
| normal | 125 |
| anomaly | 130 |

## Outputs

- normal train: `/Users/ritik/Documents/Project TDA/TyreVisionX/data/manifests/D1_anomaly_train_normal.csv`
- validation mixed: `/Users/ritik/Documents/Project TDA/TyreVisionX/data/manifests/D1_anomaly_val_mixed.csv`
- test mixed: `/Users/ritik/Documents/Project TDA/TyreVisionX/data/manifests/D1_anomaly_test_mixed.csv`

## Data Contract

Required columns include `image_path`, `target`, `split`, `is_normal`, `source_dataset`, and `product_type`.
`target = 0` means normal/good and `target = 1` means anomaly/defect.
