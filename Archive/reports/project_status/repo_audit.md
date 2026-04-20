# TyreVisionX Repository Audit

Date: 2026-04-06

This document records the repository audit captured before the cleanup refactor completed on 2026-04-06. The pre-cleanup findings are preserved here, and the applied changes are summarized later in `Cleanup Applied On 2026-04-06`.

## Scope
This audit is based on repository inspection only. It does not invent missing artifacts or assume unpublished experiment results.

Checked areas:
- repository structure
- README and project docs
- training, evaluation, export, service, and app entry points
- configs, dataset scripts, manifests, tests, and artifact directories

## Executive Summary
At audit time, TyreVisionX was an early-stage research prototype centered on binary tyre defect classification. The repository contained:

- one active ResNet-based classification path under `src/train.py` and `src/evaluate.py`
- one older parallel baseline path under `src/train_baseline.py`, `src/eval_baseline.py`, and `scripts/day5_seed_sweep.py`
- partial supporting inference surfaces via FastAPI, Streamlit, and batch inference
- future-facing GNN, detection, and segmentation hooks that are not yet validated production paths

The main repository risk is not missing code but inconsistency between paths:
- dataset manifests exist in two incompatible conventions
- the default ResNet-18 config still points to the older manifest convention
- evaluation ignores the training manifest override and always uses `configs/data.yaml`
- docs describe a cleaner and more mature pipeline than the repository currently provides

## Canonical Current Pipeline
At audit time, the canonical active pipeline was the config-driven ResNet classification path:

1. Dataset config in `configs/data.yaml`
2. Training entry point in `src/train.py`
3. Evaluation entry point in `src/evaluate.py`
4. Model construction in `src/models/resnet_classifier.py`
5. Dataset combination utilities in `src/data/datasets.py`
6. Albumentations config-based transforms in `src/data/transforms.py`

This was the most coherent path because it supported:
- YAML-driven model configuration
- dataset selection through `configs/data.yaml`
- combined dataset loading
- report generation under `artifacts/reports/<exp_name>/`
- registry-aware inference surfaces

Important caveat at audit time:
- `configs/train_resnet18.yaml` still injects `data.manifest_csv: data/processed/D1_manifest.csv`, so the default training command uses a legacy manifest override instead of the cleaner `configs/data.yaml` path.

## Legacy / Parallel Pipeline
The legacy parallel pipeline is the baseline branch:

- `src/train_baseline.py`
- `src/eval_baseline.py`
- `scripts/day5_seed_sweep.py`
- `src/dataset.py`
- `src/transforms.py`
- `src/models/simple_cnn.py`
- `src/models/feature_extractor.py`

Characteristics:
- CLI-argument driven instead of YAML-config driven
- tied to `data/processed/D1_manifest.csv`
- optimized around Day 3 / Day 5 baseline experiments
- still useful for historical comparison and artifact interpretation
- not aligned with the intended consolidated architecture

## Implemented Now
### Core research functionality
- Binary classification for `good` vs `defect`
- ResNet-18/34 training path
- older SimpleCNN and frozen-feature baselines
- stratified split utilities
- manifest-backed datasets
- evaluation reports with confusion matrix and ROC/PR plots

### Supporting interfaces
- FastAPI `/classify` inference endpoint
- Streamlit QA app for single/batch upload and manifest evaluation
- batch inference script with optional Grad-CAM output
- ONNX and TorchScript export script

### Project support assets
- tests for dataset loading, transforms, metrics, and service endpoint
- CI workflow running `pytest -q`
- research notes and baseline reports in `reports/`
- dataset download and manifest preparation scripts

## Partial / Experimental
### CNN to GNN path
- `src/models/cnn_gnn.py` is wired behind `model.gnn.enabled`
- this path depends on `torch_geometric`
- no validated artifact directory or documented completed experiment was found

### Multi-dataset support
- `configs/data.yaml` names D1, D2, and D3
- only D1 currently has a populated manifest
- `data/manifests/D2_crack_manifest.csv` and `data/manifests/D3_quality_manifest.csv` contain header only
- cross-dataset evaluation is therefore scaffolded, not established

### Inference packaging
- FastAPI and Streamlit can load registry entries or a fallback dummy model
- no populated `artifacts/experiments/` directory was found in the current workspace
- registry-backed inference exists in code, but not as a verified reproducible artifact flow in this checkout

## Future Planned Work
- detection via YOLOv8 stub in `src/models/yolo_stub.md`
- segmentation via Mask R-CNN / U-Net stub in `src/models/segmentation_stub.md`
- broader anomaly detection and localization directions named in project guidance
- multi-view 3D reconstruction, defect projection, and knowledge graph reasoning from `AGENTS.md`

These are roadmap items, not implemented deliverables in the current repository state.

## Findings
### 1. Dataset path conventions are inconsistent
Two manifest conventions are present:

- `data/processed/D1_manifest.csv` stores repo-relative paths like `data/raw/defect/...`
- `data/manifests/D1_tyrenet_manifest.csv` stores dataset-root-relative paths like `defect/...` and relies on `configs/data.yaml` root resolution

Impact:
- the active codebase must maintain two dataset loaders
- training and evaluation can silently target different manifest sources
- docs are harder to trust because both conventions are described or implied in different places

### 2. Default training config mixes active and legacy logic
`configs/train_resnet18.yaml` contains both:
- `data.manifest_csv: data/processed/D1_manifest.csv`
- `data.config_file: configs/data.yaml`

In `src/train.py`, `manifest_csv` takes precedence and bypasses `configs/data.yaml`.

Impact:
- `make train` uses the older single-manifest path
- `src/evaluate.py` later rebuilds the dataset only from `data.config_file`
- the train/eval pair is not guaranteed to use the same manifest definition

### 3. Training and evaluation disagree on data source selection
`src/train.py` supports:
- `manifest_csv`
- or `config_file` plus combined dataset loading

`src/evaluate.py` supports only:
- `config_file` plus combined dataset loading

Impact:
- a checkpoint trained from `data/processed/D1_manifest.csv` may be evaluated against `data/manifests/D1_tyrenet_manifest.csv`
- this is the most important current pipeline inconsistency

### 4. Dataset loading logic is duplicated
Both of these exist:
- `src/data/datasets.py::TyreManifestDataset`
- `src/dataset.py::TyreManifestDataset`

Both of these exist:
- `src/data/transforms.py`
- `src/transforms.py`

Impact:
- duplicated behavior
- increased maintenance burden
- higher chance of subtle divergence in path resolution, metadata shape, and augmentation behavior

### 5. Manifest generation docs do not match the active multi-dataset config
README says manifests live under `data/manifests/`, but `scripts/prepare_manifests.py` defaults to:
- `data/processed/D1_manifest.csv`

Impact:
- user-facing setup guidance points at one location
- the manifest creation script defaults to another
- the active ResNet-18 config reinforces the older location

### 6. Multi-dataset claims are ahead of implementation
README frames D1, D2, and D3 as part of Phase 1 and mentions cross-dataset robustness metrics.

Current repository state:
- D1 manifest populated
- D2 manifest header only
- D3 manifest header only
- no experiment artifacts under `artifacts/experiments/`

Impact:
- documentation overstates operational maturity
- cross-dataset evaluation is planned/scaffolded, not currently demonstrated in this checkout

### 7. README includes hard-coded results and maturity claims that should be treated as historical, not canonical status
README includes Day 3 baseline metrics and improvement hypotheses.

Risk:
- these sections read like current validated repo status
- they are tied to older baseline artifacts under `artifacts/day3_baseline/`, not the active config-driven path

### 8. Runtime issue in training registry hook
`src/train.py` calls `register_model(...)` but does not import it.

Impact:
- training can still finish because the call is wrapped in `try/except`
- registry update silently fails instead of working as documented

### 9. Export config handling is only partial
`src/export.py` looks for `config["data"]["input_size"]`.

Current train configs do not define `input_size`.

Observed behavior:
- the script falls back to `384`
- that is workable, but not explicit or aligned with the augmentation config files

### 10. Some repository layout docs are stale
Planned structure docs recommend:
- `src/tyrevisionx/`
- `src/legacy/`
- dedicated architecture and roadmap docs
- centralized logs and status reporting

Current state at audit time:
- source remains flat under `src/`
- no `src/tyrevisionx/`
- no `src/legacy/`
- requested status docs were not present before this audit

## Documentation Accuracy Audit
### Accurate enough now
- repo contains configs, scripts, tests, apps, and CI
- binary classification is the main implemented task
- GNN, detection, and segmentation are not hard dependencies for the default path

### Needs correction or qualification
- multi-dataset Phase 1 claims should be downgraded to partial
- cross-dataset reporting should be described as planned or scaffolded
- manifest location guidance should be unified
- README should distinguish active ResNet path from baseline experiment path
- baseline metrics in README should be labeled as historical observations, not current repo status

## Verification Notes
Static verification performed:
- top-level module import check for `src.train`, `src.evaluate`, `src.cli`, `src.train_baseline`, and `src.eval_baseline`
- manifest inventory and line counts
- artifact directory inspection
- README versus file-tree comparison

Not verified in this audit:
- full training run
- full evaluation run
- notebook execution
- service behavior against a real trained checkpoint

## Safe Refactor Order
1. Choose one canonical manifest convention and document it.
2. Make `src/train.py` and `src/evaluate.py` consume the same dataset source contract.
3. Remove or isolate the legacy dataset and transform modules behind a clearly named legacy path.
4. Update README and scripts so default commands follow the canonical path only.
5. Fix the registry import and verify `artifacts/experiments/` plus `artifacts/registry/` behavior.
6. After the pipeline is stable, reorganize package structure into active versus legacy namespaces.

## Proposed Fixes
- Remove `data.manifest_csv` from the default canonical ResNet config or make evaluation honor it too.
- Standardize on `data/manifests/*.csv` plus dataset-root-relative image paths.
- Consolidate dataset loading into one `TyreManifestDataset` implementation.
- Consolidate transform logic into one active module.
- Reclassify `train_baseline.py`, `eval_baseline.py`, and `day5_seed_sweep.py` as legacy research baselines.
- Update README to distinguish:
  - implemented now
  - partial / experimental
  - future planned work
- Add explicit placeholders when artifacts or reports are absent instead of implying they exist.

## Cleanup Applied On 2026-04-06
The following low-risk cleanup work was applied after this audit:

- added canonical config directories under `configs/data/`, `configs/train/`, and `configs/aug/`
- removed the default active manifest override from the canonical ResNet configuration
- updated `src/train.py` and `src/evaluate.py` to use a shared runtime dataset-loading helper
- fixed the missing `register_model` import in `src/train.py`
- added `src/legacy/` namespace wrappers so archived baseline code has a clear home
- normalized default manifest and fold-preparation paths toward `configs/data/datasets.yaml`
- updated tests to cover the shared runtime dataset contract
- added local pytest path bootstrapping and a clean skip for service tests when multipart support is absent

## Remaining Follow-up
- root-level compatibility copies and compatibility import paths still exist and should be retired only after another verification pass
- the full professor-ready notebook set recommended in the cleanup plan is still outstanding
- D2 and D3 remain unpopulated in tracked manifests, so multi-dataset claims must remain partial
- no new experiment results were generated as part of this cleanup
- active and compatibility config copies should be kept in sync until the old paths are formally removed

## Remaining Risks
- Existing historical artifacts and notebooks may depend on the legacy manifest convention.
- Refactoring dataset paths without migration notes could break reproducibility for earlier experiments.
- D2 and D3 support remains mostly structural until manifests and evaluation artifacts are real and versioned.
- Inference surfaces may continue to fall back to dummy behavior if registry and experiment outputs are not formalized.
