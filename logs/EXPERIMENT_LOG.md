# Experiment Log

This log summarizes experiment results already recorded in committed reports or progress notes. It does not add new results.

| Date / Phase | Experiment ID | Dataset | Model | Source | Status | Result Summary | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Day 3 | `simple_cnn_v1` | D1 | SimpleCNN | `reports/day3_baseline_observations.md` | Historical baseline | test accuracy `0.7647`, defect F1 `0.8101`, AUROC `0.8631` | One-epoch baseline with high false positives |
| Day 3 | `simple_cnn_v2` | D1 | SimpleCNN | `reports/day3_baseline_observations.md` | Historical baseline | test accuracy `0.8353`, defect F1 `0.8552`, AUPRC `0.8650` | Two-epoch follow-up |
| Day 3 | `frozen_resnet18_v1` | D1 | Frozen ResNet18 | `reports/day3_baseline_observations.md` | Historical baseline | test accuracy `0.9333`, defect F1 `0.9354`, AUROC `0.9802` | Stronger transfer-learning baseline |
| Day 3 | `frozen_resnet18_unfreeze_v2` | D1 | ResNet18 with layer4 unfrozen | `reports/day3_baseline_observations.md` | Historical baseline | defect recall `0.9846`, defect F1 `0.9660`, AUROC `0.9933` | Good recall-critical trade-off |
| Day 3 | `frozen_resnet50_unfreeze_v2_3ep` | D1 | ResNet50 with layer4 unfrozen | `reports/day3_baseline_observations.md` | Historical baseline | accuracy `0.9804`, defect recall `0.9692`, defect F1 `0.9805`, AUROC `0.9985` | Best documented Day 3 accuracy, but recall trade-off remained important |
| Day 4 | `simple_cnn_phase1_sigmoid_v1` | D1 | Tiny SimpleCNN | `reports/day4_partc_tiny_cnn_observations.md` | Historical baseline | defect recall `0.9923`, defect F1 `0.7633`, AUROC `0.8824` | High recall, many false positives |
| Day 5 | `step3_step4_comparison` | D1 | SimpleCNN variants | `reports/day5_regularization_observations.md` | Historical baseline study | short-run comparison showed augmentation improved precision while baseline kept highest recall | Three-epoch comparison after BN/Dropout fix |
| Day 5 | `longrun_seed_sweep` | D1 | SimpleCNN variants | `reports/day5_regularization_observations.md` | Historical baseline study | baseline had highest mean recall, augmentation had best stability and AUPRC trade-off | Multi-seed long-run comparison |

## Notes
- The canonical cleaned pipeline has been stabilized structurally in this refactor, but no new canonical-pipeline experiment results were generated here.
- D2 and D3 are not logged yet because tracked manifests are not populated in this checkout.
