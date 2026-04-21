# Patch-Aware Model Comparison

Thresholds are selected on validation only and applied once to test.

| Variant | Status | Layer | Method | Memory Patches | Recall | Precision | FN | FP | AUROC | AUPRC |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet50_knn_threshold_sweep_reference` | reused_existing_result | pooled_penultimate | pooled_embedding_reference |  | 0.9462 | 0.7736 | 7 | 36 | 0.9298 | 0.9339 |
| `resnet50_featuremap_patch_knn` | executed | layer4 | featuremap_patch_knn | 10000 | 0.1000 | 0.6842 | 117 | 6 | 0.6523 | 0.6105 |
| `resnet50_featuremap_patch_knn_threshold_sweep` | executed | layer4 | featuremap_patch_knn_threshold_sweep | 10000 | 0.4615 | 0.6667 | 70 | 30 | 0.6523 | 0.6105 |
| `resnet50_patchcore_lite` | executed | layer4 | patchcore_lite | 5000 | 0.2385 | 0.7381 | 99 | 11 | 0.6461 | 0.6086 |
