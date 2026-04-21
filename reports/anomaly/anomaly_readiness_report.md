# Anomaly Readiness Report

Config: `configs/anomaly/anomaly_baseline.yaml`
Configured backbone: `resnet18`
Expected torchvision weight: `resnet18-f37072fd.pth`
Can run real anomaly baseline now: `True`

## Chosen Weight Source

- source type: `torchvision_cache`
- backbone: `resnet18`
- path: `/Users/ritik/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
- notes: Matches configured torchvision backbone filename.

## Manifest Checks

| Manifest | Exists |
|---|---:|
| normal_train_manifest | `True` |
| validation_manifest | `True` |
| test_manifest | `True` |

## Candidate Weight Sources

| Type | Backbone | Compatible | Path | Notes |
|---|---|---:|---|---|
| torchvision_cache | resnet18 | `True` | `/Users/ritik/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` | Matches configured torchvision backbone filename. |
| fallback_torchvision_cache | resnet50 | `True` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/.torch/hub/checkpoints/resnet50-11ad3fa6.pth` | Supported fallback only if config is changed to backbone=resnet50. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/.torch/hub/checkpoints/resnet50-11ad3fa6.pth` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/frozen_resnet18_unfreeze_v2/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/frozen_resnet18_unfreeze_v2/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/frozen_resnet18_v1/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/frozen_resnet18_v1/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/frozen_resnet50_unfreeze_v1/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/frozen_resnet50_unfreeze_v1/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/frozen_resnet50_unfreeze_v2_3ep/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/frozen_resnet50_unfreeze_v2_3ep/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/simple_cnn_phase1_sigmoid_v1/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/simple_cnn_phase1_sigmoid_v1/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/simple_cnn_v1/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/simple_cnn_v1/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/simple_cnn_v2/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day3_baseline/simple_cnn_v2/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/augmentation/seed_123/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/augmentation/seed_123/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/augmentation/seed_7/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/augmentation/seed_7/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/augmentation/seed_999/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/augmentation/seed_999/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/baseline/seed_123/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/baseline/seed_123/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/baseline/seed_7/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/baseline/seed_7/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/baseline/seed_999/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/baseline/seed_999/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/batchnorm/seed_123/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/batchnorm/seed_123/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/batchnorm/seed_7/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/batchnorm/seed_7/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/batchnorm/seed_999/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/batchnorm/seed_999/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/dropout/seed_123/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/dropout/seed_123/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/dropout/seed_7/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/dropout/seed_7/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/dropout/seed_999/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/longrun_seed_sweep/dropout/seed_999/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/simple_cnn_control_noaug_nowd/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/simple_cnn_control_noaug_nowd/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/simple_cnn_day5_aug_nowd/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/simple_cnn_day5_aug_nowd/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/simple_cnn_day5_aug_wd1e4/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/simple_cnn_day5_aug_wd1e4/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug_batchnorm/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug_batchnorm/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug_batchnorm_dropout03/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug_batchnorm_dropout03/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug_batchnorm_dropout03_fix/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug_batchnorm_dropout03_fix/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug_batchnorm_fix/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_aug_batchnorm_fix/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_baseline/best.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |
| project_checkpoint | unknown | `False` | `/Users/ritik/Documents/Project TDA/TyreVisionX/artifacts/day5/step3_step4_baseline/last.pt` | Found locally, but not selected automatically unless proven ResNet-backbone compatible. |

## Blocker

No blocker found. A real run is feasible.
