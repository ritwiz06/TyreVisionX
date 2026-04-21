# Local Feature Anomaly Guide

## What Local Features Mean
Local features are image representations computed from regions of an image rather than only from the full image. For tyre inspection, this matters because a crack or sidewall defect may occupy only a small part of the tyre.

## Why TyreVisionX Uses Them Now
The best current anomaly model, `resnet50_knn`, is stronger than the first baseline but still has too many clean false negatives. That suggests the model may miss subtle local evidence after whole-image pooling.

## Multi-Crop Inference
Multi-crop inference evaluates the full image plus several overlapping crops, such as center and corner crops. Each crop receives an anomaly score. TyreVisionX aggregates those scores with a conservative rule: the image score is the maximum crop score. If any meaningful region looks unusual, the whole image should be reviewed as anomalous.

## Patch-Grid Embeddings
Patch-grid scoring splits the image into a fine grid, extracts an embedding for each patch, and compares each patch against normal tyre patches. TyreVisionX currently uses a 3x3 grid plus the full image. This is still simple enough to explain, but more local than whole-image scoring.

## Multi-Scale Scoring
Multi-scale scoring means evaluating both full-image and local regions at different sizes. This prompt keeps the implementation focused on multi-crop and patch-grid scoring first. More complex multi-scale methods are planned only if these simple variants show evidence of helping.

## Validation-Only Threshold Sweep
A threshold converts an anomaly score into a normal/anomaly decision. TyreVisionX chooses thresholds on validation data only, then applies the chosen threshold once to test data. This avoids quietly tuning to the test set.

## Why Bigger Models Are Not Automatically Next
Larger backbones can help, but the current failure pattern points toward local defects. A larger global embedding may still dilute small cracks. Local-feature scoring tests the more direct hypothesis first.

## How This Is Implemented
- `src/anomaly/multicrop.py` builds local crop tensors.
- `src/anomaly/local_features.py` extracts and aggregates crop embeddings.
- `src/anomaly/local_benchmark.py` runs the controlled benchmark.
- `scripts/anomaly/run_local_feature_benchmark.py` is the CLI entry point.
