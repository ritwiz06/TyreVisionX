# Local Feature Best Variant Recommendation

## Recommendation
Use `resnet50_knn_threshold_sweep` as the next operational tyre anomaly candidate, not the current local-feature variants.

## Why
The benchmark target was false-negative reduction. The validation-only threshold sweep reduced test false negatives from 23 to 7 and improved recall from 0.8231 to 0.9462. This is the strongest observed recall improvement in this prompt.

## Tradeoff
The improvement comes with more false positives:

- Reference FP: 15
- Threshold-sweep FP: 36

For tyre inspection, this tradeoff is more acceptable than missed defects, but it is not yet production-ready. A professor-facing interpretation should describe this as a recall-prioritized research candidate.

## Local-Feature Result
The simple local-feature methods were not the winner:

- `resnet50_multicrop_knn` kept the same FN count as the reference.
- `resnet50_patch_grid_knn_fine` failed badly on recall.

This does not disprove local evidence. It shows that naive resized crop/patch kNN is not enough. The next local-feature step should be more principled, such as patch embeddings from convolutional feature maps, PatchCore-style memory banks, or region-aware feature aggregation.

## Next Experiment
Run a calibrated high-recall `resnet50_knn` threshold setting as the current best candidate, then test a true patch-aware method that avoids resizing tiny tyre regions into full-image inputs.
