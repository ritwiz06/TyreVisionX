# Project Status

Date: 2026-04-06

TyreVisionX is an early-stage academic research prototype for tyre defect inspection. After the cleanup in this refactor, the repository has a clearer canonical training and evaluation path, an archived legacy baseline namespace, and more honest project-facing documentation.

## Implemented Now
### Canonical active path
- training entry point: `src/train.py`
- evaluation entry point: `src/evaluate.py`
- canonical dataset config: `configs/data/datasets.yaml`
- canonical train configs:
  - `configs/train/train_resnet18.yaml`
  - `configs/train/train_resnet34.yaml`
- canonical transform configs:
  - `configs/aug/light.yaml`
  - `configs/aug/strong.yaml`

### Shared pipeline behavior
- training and evaluation now use the same runtime dataset-loading contract
- the canonical path uses manifest sets from `configs/data/datasets.yaml`
- export config now has an explicit input size in the canonical train configs
- registry import in `src/train.py` is fixed

### Supporting research utilities
- FastAPI inference endpoint
- Streamlit QA app
- batch inference script
- baseline reports and research notes
- tests for dataset loading, transforms, metrics, and service behavior

## Cleanup Applied In This Refactor
- added canonical config subdirectories under `configs/data/`, `configs/train/`, and `configs/aug/`
- removed the default ResNet manifest override from the active training config path
- added `src/legacy/` namespace wrappers for archived baseline components
- kept root-level config files as compatibility copies instead of deleting them
- normalized manifest preparation and fold preparation defaults toward the canonical dataset config
- added test bootstrapping so local pytest runs find `src`
- added a dataset-config test that covers dataset-root-relative manifest resolution
- updated README and status/reporting files to match actual repository maturity

## Partial / Experimental
### Multi-dataset support
- D1 is populated in tracked manifests
- D2 and D3 remain configured but empty in tracked manifests
- cross-dataset robustness is therefore still a planned research direction rather than a validated current result

### CNN→GNN path
- available as an optional model path
- still experimental and dependency-gated through `torch_geometric`

### Inference registry workflow
- code path is present and cleaner than before
- still dependent on actual experiment outputs being present under `artifacts/experiments/`

## Future Planned Work
- anomaly detection
- localization and segmentation
- stronger cross-dataset evaluation once D2/D3 manifests are populated
- multi-view 3D reconstruction
- defect projection to mesh
- knowledge-graph-style reasoning

## Remaining Known Gaps
- the repository still uses compatibility copies instead of fully removing old config paths
- the older notebook structure has not yet been replaced by the full professor-ready notebook set from the cleanup plan
- historical baseline scripts remain in the root namespace as compatibility paths, even though `src/legacy/` now exists
- service tests depend on the `python-multipart` package; they now skip cleanly when that dependency is not installed

## Honest Status Statement
TyreVisionX now has one clear canonical classification pipeline and clearer separation between active and legacy code. It remains an early research repository, with the strongest current support in binary classification on D1 and research reporting, not in broad multi-dataset validation or production deployment.
