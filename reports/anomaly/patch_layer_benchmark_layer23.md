# Patch-Aware Model Comparison

Thresholds are selected on validation only and applied once to test.

| Variant | Status | Layer | Method | Normalized | Memory Patches | Recall | Precision | FN | FP | AUROC | AUPRC |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet50_knn_threshold_sweep_reference` | reused_existing_result | pooled_penultimate | pooled_embedding_reference | False |  | 0.9462 | 0.7736 | 7 | 36 | 0.9298 | 0.9339 |
| `resnet50_layer2_layer3_patch_knn_threshold_sweep` | executed | layer2_layer3 | layer2_layer3_patch_knn_threshold_sweep | True | 12000 | 0.2154 | 0.4242 | 102 | 38 | 0.3932 | 0.4528 |
