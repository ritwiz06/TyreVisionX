# Robustness Recommendation

Updated: 2026-04-20

## Recommendation

Keep `resnet50_knn` as the main anomaly baseline, and keep the mild noise-robust variant as a useful candidate because it preserves clean recall while reducing false positives and improving several corrupted conditions.

However, robustness training did not solve the recall-critical problem. The next major improvement should be patch-aware or local-feature ResNet50 work, while retaining mild robustness augmentation as a secondary regularization option.

## Evidence

- Clean-trained `resnet50_knn`: clean recall `0.8231`, FN `23`, FP `15`.
- Noise-robust `resnet50_knn`: clean recall `0.8231`, FN `23`, FP `13`.
- Robust variant under medium blur: recall `0.8231`, FN `23`, compared with clean-trained recall `0.7846`, FN `28`.
- Robust variant under darker brightness: recall `0.8231`, FN `23`, compared with clean-trained recall `0.7846`, FN `28`.

## Practical Decision

Use mild robustness augmentation in future ResNet50 kNN experiments, but prioritize visual inspection of remaining false negatives and a stronger local-feature method next.
