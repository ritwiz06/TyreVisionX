# Local Feature Model Comparison

Thresholds are selected on validation only and applied once to test.

| Variant | Status | Local Mode | Crops | Recall | Precision | FN | FP | AUROC | AUPRC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet50_knn_reference` | reused_existing_result | none | 1 | 0.8231 | 0.8770 | 23 | 15 | 0.9298 | 0.9339 |
| `resnet50_knn_threshold_sweep` | executed | none | 1 | 0.9462 | 0.7736 | 7 | 36 | 0.9298 | 0.9339 |
| `resnet50_multicrop_knn` | executed | multicrop | 6 | 0.8231 | 0.8425 | 23 | 20 | 0.9170 | 0.8852 |
| `resnet50_patch_grid_knn_fine` | executed | patch_grid_fine | 10 | 0.4692 | 0.7625 | 69 | 19 | 0.7575 | 0.7353 |
