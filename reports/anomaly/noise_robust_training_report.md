# Noise-Robust Training Report

Updated: 2026-04-20

## Variant

- Backbone: ResNet50
- Scorer: kNN
- Training data: normal-only D1 train manifest
- Added normal-training augmentations: light Gaussian noise, slight blur, light brightness shift, light contrast shift
- Threshold calibration: clean validation only

## Clean Test Result

| Variant | Recall | FN | FP | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| `resnet50_knn` | 0.8231 | 23 | 15 | 0.9298 | 0.9339 |
| `resnet50_knn_noise_robust` | 0.8231 | 23 | 13 | 0.9407 | 0.9383 |

## Interpretation

The robust-trained variant preserved clean recall and reduced clean false positives. It did not reduce clean false negatives, so it should be treated as a robustness improvement, not the final recall solution.
