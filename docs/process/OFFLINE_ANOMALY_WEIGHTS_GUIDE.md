# Offline Anomaly Weights Guide

Updated: 2026-04-19

TyreVisionX uses a frozen CNN backbone to turn tyre images into embedding vectors for anomaly scoring. For a real research run, that backbone needs meaningful pretrained weights.

## Why Pretrained Weights Matter

Pretrained weights come from a model that has already learned useful visual features such as edges, textures, and object parts. For TyreVisionX, those features are reused to compare good tyres against possible defects.

Random weights are not meaningful for anomaly results because the embedding space has not learned visual structure. A random-weight run can be used only as a software smoke test.

## Preferred Offline Setup

For the default config:

```yaml
feature_extractor:
  backbone: resnet18
  pretrained: true
  weights_path: null
```

Place the torchvision ResNet18 file here:

```text
~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

Then run:

```bash
python3 scripts/anomaly/check_anomaly_readiness.py
python3 scripts/anomaly/run_anomaly_baseline.py --config configs/anomaly/anomaly_baseline.yaml
```

## Explicit Local Weights Path

You can also set:

```yaml
feature_extractor:
  backbone: resnet18
  pretrained: true
  weights_path: /absolute/path/to/resnet18-f37072fd.pth
```

This is useful when the machine has no network access and the weights are stored outside the normal Torch cache.

## Classifier Checkpoints Vs Pretrained Weights

A classifier checkpoint may include a final classification layer trained for `good` vs `defect`. A generic pretrained ResNet checkpoint is cleaner for the first anomaly baseline because the anomaly method only needs frozen feature embeddings, not the classifier head.

TyreVisionX may later reuse supervised checkpoints as feature extractors, but only when the checkpoint format and backbone compatibility are verified clearly.

## Readiness Checker

Run:

```bash
python3 scripts/anomaly/check_anomaly_readiness.py
```

It reports:
- configured backbone
- expected torchvision weight file
- found local weight sources
- selected compatible source
- whether the D1 anomaly run can execute now
- exact blocker if not
