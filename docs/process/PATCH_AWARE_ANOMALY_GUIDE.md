# Patch-Aware Anomaly Guide

## Big-Picture Idea
A tyre defect can be tiny compared with the full image. If the model compresses the entire tyre into one vector, a small crack can be diluted by all the normal-looking rubber around it. Patch-aware anomaly detection keeps local evidence alive for longer.

## Convolutional Feature Maps
A convolutional neural network does not only produce one final prediction. Inside the network, it produces feature maps: grids of learned features. Each grid cell summarizes a region of the original image. Early layers capture edges and textures. Deeper layers capture larger, more semantic patterns.

## Receptive Fields
The receptive field of a feature-map cell is the area of the original image that influenced that cell. In ResNet50, a deep feature-map location may represent a local tyre region rather than the entire image. This makes it useful for local defect evidence.

## Local Descriptors
A local descriptor is the feature vector at one feature-map location. If a ResNet50 feature map has shape `C x H x W`, then each of the `H * W` locations gives one `C`-dimensional patch descriptor.

## Memory Bank Anomaly Detection
The memory bank stores patch descriptors from normal training tyres. At test time, each new patch asks: "How close am I to the most similar normal patch?" If a patch is far from the normal memory, it is suspicious.

## Nearest-Neighbor Patch Scoring
For each test patch descriptor, TyreVisionX computes the distance to its nearest normal patch descriptor. A larger distance means more anomalous. The image-level score is then aggregated from all patch scores.

## PatchCore-Style Intuition
PatchCore methods are based on a simple idea: keep a representative memory of normal local features, then flag image regions that do not match that memory. Full PatchCore adds more engineering, such as layer combination and coreset selection. TyreVisionX starts with a lightweight version to test whether the idea helps this dataset.

## Image-Level Aggregation
TyreVisionX must still make one decision per tyre image. The benchmark uses aggregation rules such as:

- `max`: one highly anomalous patch can make the image anomalous.
- `top3_mean`: average the most suspicious few patches to reduce single-patch noise.

## Why Resized Small Image Patches Can Fail
The previous patch-grid test resized small tyre regions into full 224x224 images. That changes texture scale and context. A normal local rubber patch may look strange when blown up, causing noisy scores.

Feature-map patches are more principled because they are produced naturally by the convolutional network while processing the original image.

## Calibration Still Matters
Even with better local features, a threshold is still needed. A low threshold catches more defects but creates more false positives. A high threshold reduces false alarms but can miss defects. TyreVisionX chooses thresholds on validation data only and reports final test behavior once.

## TyreVisionX Interpretation
If patch-aware memory reduces false negatives compared with `resnet50_knn_threshold_sweep`, it supports the idea that missed tyre defects are local. If it does not, then the remaining misses may require better data, label review, stronger backbones, or supervised localization rather than simple patch memory.
