# Anomaly Model Comparison

This table compares variants using validation-selected thresholds. Test metrics are reported once per executed variant.

| Variant | Status | Backbone | Scorer | Embedding | Recall | Precision | Normal FPR | FN | FP | AUROC | AUPRC |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet18_mahalanobis_reference` | reused_existing_result | resnet18 | mahalanobis | pooled_penultimate | 0.2846 | 0.7708 | 0.0880 | 93 | 11 | 0.7194 | 0.7212 |
| `resnet18_knn` | executed | resnet18 | knn | pooled_penultimate | 0.5231 | 0.8831 | 0.0720 | 62 | 9 | 0.8546 | 0.8394 |
| `resnet18_threshold_sweep` | executed | resnet18 | mahalanobis | pooled_penultimate | 0.5769 | 0.7009 | 0.2560 | 55 | 32 | 0.7194 | 0.7212 |
| `resnet50_mahalanobis` | executed | resnet50 | mahalanobis | pooled_penultimate | 0.7077 | 0.8214 | 0.1600 | 38 | 20 | 0.8676 | 0.8800 |
| `resnet50_knn` | executed | resnet50 | knn | pooled_penultimate | 0.8231 | 0.8770 | 0.1200 | 23 | 15 | 0.9298 | 0.9339 |
| `resnet18_patch_grid_knn` | executed | resnet18 | knn | patch_grid | 0.3462 | 0.7895 | 0.0960 | 85 | 12 | 0.7872 | 0.7407 |
