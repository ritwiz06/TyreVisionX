# Patch Layer Best Variant Recommendation

## Recommendation
Keep `resnet50_knn_threshold_sweep_reference` as the current best recall-oriented anomaly candidate.

The lower/mid-level patch-aware variants did not reduce false negatives and should not replace the pooled ResNet50 kNN threshold-sweep reference.

## Comparison
| Variant | Recall | Precision | FN | FP | Recommendation |
|---|---:|---:|---:|---:|---|
| `resnet50_knn_threshold_sweep_reference` | 0.9462 | 0.7736 | 7 | 36 | Keep as current reference. |
| `resnet50_layer3_patch_knn` | 0.1231 | 0.5714 | 114 | 12 | Reject for recall-critical use. |
| `resnet50_layer3_patch_knn_threshold_sweep` | 0.3385 | 0.5176 | 86 | 41 | Reject for recall-critical use. |
| `resnet50_layer2_layer3_patch_knn_threshold_sweep` | 0.2154 | 0.4242 | 102 | 38 | Reject for recall-critical use. |

## Why The Patch Variants Are Not Recommended
The patch variants were designed to test whether lower/mid-level ResNet50 feature-map descriptors could recover local subtle tyre defects that whole-image pooled embeddings missed. In this run, they did not provide useful separation between good and defective tyre images.

The failure is not only a false-positive tradeoff. The main issue is low defect recall: the layer3 and layer2+layer3 methods missed far more defects than the pooled reference.

## Practical Next Step
Do not add more patch variants blindly. First inspect:

- validation/test patch-score distributions,
- nearest normal patch neighbors for missed defect regions,
- whether memory-bank sampling removes rare but important normal texture,
- whether feature normalization is suppressing useful defect contrast.

If patch-aware work continues, it should be a diagnostic prompt rather than another broad benchmark prompt.
