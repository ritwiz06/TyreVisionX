# Anomaly Robustness Status

Updated: 2026-04-20

## Status

The corruption robustness benchmark and mild noise-robust ResNet50+kNN variant both executed.

## Executed Corruptions

- clean
- gaussian_noise_low
- gaussian_noise_medium
- gaussian_blur_low
- gaussian_blur_medium
- jpeg_compression_mild
- brightness_darker
- brightness_brighter
- contrast_lower
- contrast_higher

## Main Finding

`resnet50_knn` degrades moderately under realistic corruptions. The mild noise-robust variant preserved clean recall and improved several corrupted conditions, but did not reduce clean false negatives.

## Current Recommendation

The next local-feature stage has now run. Threshold refinement improved recall more than the simple local crop/patch variants. Use `resnet50_knn_threshold_sweep` as the current recall-oriented candidate while reviewing its higher false-positive load.
