# Anomaly Baseline Report

Status: implemented and executed once on D1 with cached ResNet-18 weights

## Method

The first anomaly baseline uses frozen CNN embeddings:

- backbone: ResNet-18
- default weights: ImageNet pretrained weights
- embedding: pooled penultimate feature vector
- primary anomaly score: Mahalanobis distance to the good-only training embedding distribution
- optional secondary score in code: k-nearest-neighbor distance

Higher scores mean the image is farther from the normal training distribution.

## Data Contract

The baseline expects:

- `data/manifests/D1_anomaly_train_normal.csv`
- `data/manifests/D1_anomaly_val_mixed.csv`
- `data/manifests/D1_anomaly_test_mixed.csv`

Training uses only normal/good images. Validation and test are mixed good/defect manifests.

## Threshold Policy

The threshold is chosen only on validation data.

Default policy:

- maximize anomaly/defect recall
- subject to normal false-positive rate <= 10%
- if infeasible, fall back to maximizing F1

The selected threshold is then applied once to the test split.

## Current Execution Status

The D1 anomaly manifests were generated successfully:

| Manifest | Normal | Anomaly | Total |
|---|---:|---:|---:|
| train normal | 582 | 0 | 582 |
| validation mixed | 125 | 130 | 255 |
| test mixed | 125 | 130 | 255 |

The full preferred run completed using cached ResNet-18 weights.

Run artifact directory:

- `artifacts/anomaly/d1_resnet18_mahalanobis_v1`

Weight source:

- `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`

Validation threshold:

- selected on validation only
- threshold: `12.147390651173238`
- policy: maximize anomaly recall subject to normal FPR <= 10%
- fallback used: `false`

## Test Result

| Metric | Value |
|---|---:|
| AUROC | 0.7194 |
| AUPRC | 0.7212 |
| anomaly recall | 0.2846 |
| anomaly precision | 0.7708 |
| normal FPR | 0.0880 |
| accuracy | 0.5922 |
| F1 | 0.4157 |
| true positives | 37 |
| true negatives | 114 |
| false positives | 11 |
| false negatives | 93 |

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[114, 11],
 [ 93, 37]]
```

Interpretation:

- The baseline controls normal false positives near the requested validation policy.
- It misses many defective tyres at the selected threshold.
- This is a useful first tyre-specific anomaly baseline, but not a deployable recall-critical detector.

Offline compatibility remains available:
- `feature_extractor.weights_path` can point to a local `.pth` file in `configs/anomaly/anomaly_baseline.yaml`.
- Leave it as `null` when using torchvision's normal weight loading in an online or pre-cached environment.

Run:

```bash
python scripts/data/create_anomaly_manifests.py
python scripts/anomaly/run_anomaly_baseline.py --config configs/anomaly/anomaly_baseline.yaml
```

Current completed run outputs:

- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/metadata.json`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/metrics_test.json`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/predictions_test.csv`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/false_negatives_test.csv`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/confusion_matrix_test.png`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/score_distribution_test.png`
- `artifacts/anomaly/d1_resnet18_mahalanobis_v1/pr_curve_test.png`

## Limitations

- This is an image-level anomaly baseline, not localization or segmentation.
- It depends on a good-only training manifest.
- It is not a universal manufacturing platform yet; it is designed so future products can reuse the manifest contract and pipeline.
- Current recall is too low for inspection use; false-negative review is the next critical step.
