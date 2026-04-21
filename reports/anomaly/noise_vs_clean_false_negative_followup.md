# Noise vs Clean False-Negative Follow-Up

Updated: 2026-04-20

## Summary

The mild noise-robust variant did not reduce clean false negatives: both clean-trained and noise-robust `resnet50_knn` missed `23` test defects on clean images.

Under corruptions, the robust variant reduced some false-negative spikes:

| Corruption | Clean-Trained FN | Noise-Robust FN |
|---|---:|---:|
| clean | 23 | 23 |
| gaussian_noise_low | 25 | 26 |
| gaussian_noise_medium | 19 | 22 |
| gaussian_blur_low | 24 | 23 |
| gaussian_blur_medium | 28 | 23 |
| jpeg_compression_mild | 26 | 27 |
| brightness_darker | 28 | 23 |
| brightness_brighter | 22 | 22 |
| contrast_lower | 26 | 23 |
| contrast_higher | 24 | 25 |

## Interpretation

Noise-aware normal training helped robustness for some nuisance shifts but did not fix the core clean-image miss pattern. The remaining clean false negatives likely need better local defect evidence, not only noise robustness.
