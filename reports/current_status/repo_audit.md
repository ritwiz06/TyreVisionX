# Current Repository Audit

Timestamp: 2026-04-19

This audit was created during the cleanup/scaffolding prompt. It is based on static inspection and file organization work, not a fresh training run.

## Files and Areas Inspected

- `README.md`
- `PROGRESS_LOG.md`
- `pyproject.toml`
- `requirements.txt`
- `Makefile`
- `configs/`
- `src/`
- `scripts/`
- `notebooks/`
- `reports/`
- `tests/`
- existing `docs/` and `logs/`

## Canonical Supervised Pipeline Found

The most coherent active supervised path is:

- configs: `configs/data/datasets.yaml`, `configs/train/train_resnet18.yaml`, `configs/train/train_resnet34.yaml`
- training: `src/train.py`
- evaluation: `src/evaluate.py`
- data loading: `src/data/datasets.py`
- transforms: `src/data/transforms.py`
- models: `src/models/resnet_classifier.py`

## Legacy / Parallel Paths Found

The following files support older Day 3/Day 5 baseline work:

- `src/train_baseline.py`
- `src/eval_baseline.py`
- `src/dataset.py`
- `src/transforms.py`
- `src/models/simple_cnn.py`
- `src/models/feature_extractor.py`
- `scripts/day5_seed_sweep.py`
- legacy wrappers under `src/legacy/`

These were preserved because they contain useful historical work. They should not be deleted without a dedicated migration prompt.

## Structure Issues Found

- Notebook folders used older names (`00_overview`, `01_data`, `02_models`, `03_results`, `04_analysis`) and root-level notebooks.
- Reports were split between root report files and `reports/project_status/`.
- The requested `reports/current_status/`, `reports/historical/`, `reports/anomaly/`, and `reports/web_collection/` folders did not exist.
- The requested work/process logging convention did not exist.
- Anomaly and web-data configs did not exist.
- `data/raw/` is populated locally, while the canonical config still points to `data/D1_tyrenet`.
- Both `data/manifests/` and `data/processed/` manifest conventions exist.

## Code Irregularities Noted

- Duplicate dataset/transform modules remain across active and legacy paths.
- `src/evaluation/` and `src/training/` namespaces were missing and are currently placeholders only.
- `__pycache__` files are present in the working tree. They are generated files and should not be part of research review.
- `Archive/tests/` contains older duplicate test modules that can shadow active tests during pytest collection.
- The cleanup did not modify training code because deeper refactors could destabilize historical imports.

## Changes Made Safely

- Added documentation folders and current status docs.
- Added prompt logging framework.
- Added timestamped logs for this prompt.
- Added anomaly/web scaffolding.
- Reorganized useful notebooks into the target notebook folders with `legacy_` prefixes.
- Created professor-readable notebook index/plan notebooks.
- Refreshed `README.md`.
- Added pytest configuration to ignore archived duplicate tests without deleting the archive.

## Changes Intentionally Not Made

- No experiment metrics were fabricated.
- No full anomaly model was implemented.
- No web collection code was implemented.
- No historical baseline code was deleted.
- No dataset files were deleted or moved.
- No aggressive package restructure was attempted.

## Recommended Follow-Up

1. Run `pytest -q` and fix any current import or dependency issues.
2. Decide whether `data/manifests/` or `data/processed/` is the only future manifest convention.
3. Recompute the supervised canonical baseline and update `reports/historical/REPORTED_RESULTS.md`.
4. Implement the anomaly baseline in a separate prompt using `configs/anomaly/anomaly_baseline.yaml`.

## Verification Performed

- Notebook JSON validation passed for all notebooks under `notebooks/`.
- Import check passed for `src.train`, `src.evaluate`, `src.anomaly`, `src.data.datasets`, and `src.data.transforms`.
- `pytest -q` passed with `5 passed`, `1 skipped`, and `1 warning`.
- The warning came from albumentations attempting an online version check in a restricted-network environment.
