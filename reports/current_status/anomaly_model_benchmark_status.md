# Anomaly Model Benchmark Status

Updated: 2026-04-20

## Status

The controlled D1 anomaly benchmark executed successfully for all locally runnable first-tier variants.

## Executed Variants

- `resnet18_mahalanobis_reference` reused from the completed baseline run.
- `resnet18_knn`
- `resnet18_threshold_sweep`
- `resnet50_mahalanobis`
- `resnet50_knn`
- `resnet18_patch_grid_knn`

## Local Backbone Availability

Runnable now:

- ResNet18: `/Users/ritik/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
- ResNet50: `artifacts/.torch/hub/checkpoints/resnet50-11ad3fa6.pth`

Pending:

- EfficientNet-B0: local weights not found and extractor support not implemented.
- ConvNeXt-Tiny: local weights not found and extractor support not implemented.
- ViT-B/16: local weights not found and extractor support not implemented.

## Best Current Variant

`resnet50_knn` is the current best executed variant:

- Test recall: `0.8231`
- False negatives: `23`
- False positives: `15`
- AUROC: `0.9298`
- AUPRC: `0.9339`

This is a major improvement over the original baseline, but not enough for a recall-critical production claim.

## Robustness Follow-Up

The next robustness stage has now run. See `reports/current_status/anomaly_robustness_status.md`.

Summary:
- clean-trained `resnet50_knn` degrades moderately under realistic corruptions
- mild noise-robust `resnet50_knn` preserves clean recall and reduces false positives
- clean false negatives remain `23`, so patch-aware/local-feature work is still needed

## Local-Feature Follow-Up

The local-feature stage has now run. See `reports/current_status/anomaly_local_feature_status.md`.

Summary:
- validation-only threshold refinement reduced false negatives from `23` to `7`
- multicrop did not reduce false negatives
- fine patch-grid scoring performed worse under the current implementation
- current best recall-oriented candidate is `resnet50_knn_threshold_sweep`

## Outputs

- `reports/anomaly/model_comparison.csv`
- `reports/anomaly/model_comparison.md`
- `reports/anomaly/best_variant_recommendation.md`
- `reports/anomaly/variant_false_negative_followup.md`
- `reports/anomaly/backbone_readiness_report.md`
- `reports/anomaly/backbone_readiness_report.json`
- artifacts under `artifacts/anomaly/benchmark/`
