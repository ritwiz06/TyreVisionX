# Anomaly Model Guide

Updated: 2026-04-20

## Why Pretrained Backbones Matter

A pretrained backbone is an image model that already learned visual features from a large dataset. TyreVisionX uses frozen pretrained backbones to turn tyre images into embedding vectors without training a large model from scratch.

This is useful because the local D1 dataset is small. A random-weight model would produce arbitrary features and should not be treated as a research result.

## ResNet18 vs ResNet50

ResNet18 is smaller and faster. It is the current reference because it already ran on D1.

ResNet50 is deeper and produces richer features. It may separate normal and defective tyres better, but it can still miss small local defects if the final embedding pools the whole image into one vector.

## EfficientNet, ConvNeXt, and ViT

EfficientNet, ConvNeXt, and Vision Transformer models are useful future backbones. They are not automatically better for TyreVisionX unless:

- pretrained weights are locally available,
- extractor support is implemented,
- thresholds are selected on validation only,
- false negatives improve without unacceptable false positives.

For this prompt they are readiness-checked only if local weights exist; no downloads are forced.

## Mahalanobis vs kNN

Mahalanobis scoring fits one normal embedding distribution and measures distance from it. It is simple, but it assumes normal embeddings behave like one compact cloud.

kNN scoring measures distance to nearby normal training examples. It may work better when normal tyre appearances form several clusters, such as tread, sidewall, and mounted views.

## Threshold Sweep

A threshold sweep tries many validation thresholds and records recall/precision/FPR trade-offs. TyreVisionX must choose thresholds on validation data only and then apply the chosen threshold once to the test split.

## Local Features

Global pooled embeddings can dilute small cracks because one vector summarizes the whole image. A patch-aware method scores local regions and can catch defects that occupy only part of the tyre image.

The first low-risk local-feature variant in this repo uses a patch grid: the full image plus four quadrant crops. It fits on normal patches and scores an image by the most anomalous patch.

## Why Bigger Is Not Automatically Better

A larger model can produce stronger features, but it can also emphasize ImageNet semantics rather than tyre defects. For tyre inspection, better recall comes from the right feature scale, threshold policy, and data quality, not only model size.
