# Repository Architecture

Updated: 2026-04-19

This document explains the intended TyreVisionX repository structure after cleanup. The goal is to make the active supervised path clear while preserving historical experiments and creating a clean place for anomaly-detection research.

## Current High-Level Layout

```text
TyreVisionX/
  configs/
    aug/                 # existing supervised augmentation configs
    data/                # existing supervised dataset config
    train/               # existing supervised training configs
    anomaly/             # planned anomaly baseline config
    web_collection/      # planned web-data curation config
  data/
    manifests/           # canonical manifest-style CSVs
    external/            # future third-party/raw external sources
    interim/             # future intermediate data products
    processed/           # legacy processed manifest path
    raw/                 # local raw image data, gitignored
  docs/
    architecture/        # architecture and repo organization
    codex/               # future prompt rules and durable context
    project/             # status and roadmap
    process/             # process guidance
  logs/
    work_logs/           # timestamped execution logs
    process_logs/        # timestamped beginner/process explanations
    experiment_logs/     # future experiment logging
  notebooks/
    00_project_overview/
    01_data_audit/
    02_supervised_baseline/
    03_anomaly_baseline/
    04_web_data_curation/
    05_results/
  reports/
    historical/          # historical reported results
    current_status/      # current cleanup and audit state
    anomaly/             # planned anomaly reports
    web_collection/      # planned web curation reports
  scripts/
    anomaly/             # future anomaly scripts
    data/                # future data scripts
    utilities/           # future general utilities
  src/
    anomaly/             # future anomaly code
    data/                # canonical data loading for supervised track
    evaluation/          # future evaluation package namespace
    legacy/              # historical wrappers / compatibility code
    models/
    training/            # future training package namespace
    utils/
  tests/
```

## Canonical Supervised Pipeline

The current canonical supervised pipeline is:

1. Dataset config: `configs/data/datasets.yaml`
2. Training config: `configs/train/train_resnet18.yaml` or `configs/train/train_resnet34.yaml`
3. Dataset loader: `src/data/datasets.py`
4. Transform builder: `src/data/transforms.py`
5. Model builder: `src/models/resnet_classifier.py`
6. Training entry point: `python -m src.train --config configs/train/train_resnet18.yaml`
7. Evaluation entry point: `python -m src.evaluate --checkpoint <checkpoint> --split test`

This is the path future supervised work should stabilize first.

## Historical / Legacy Path

The repository also contains a Day 3 / Day 5 baseline path:

- `src/train_baseline.py`
- `src/eval_baseline.py`
- `src/dataset.py`
- `src/transforms.py`
- `src/models/simple_cnn.py`
- `src/models/feature_extractor.py`
- `scripts/day5_seed_sweep.py`

These files are preserved because they contain useful baseline work and reported historical results. They are not deleted, and they should be treated as historical/experimental unless intentionally promoted later.

## Anomaly Track

The anomaly track is currently a scaffold:

- config: `configs/anomaly/anomaly_baseline.yaml`
- code namespace: `src/anomaly/`
- script namespace: `scripts/anomaly/`
- reports: `reports/anomaly/`
- notebook plan: `notebooks/03_anomaly_baseline/anomaly_direction_plan.ipynb`

The planned anomaly baseline will train only on good tyre images and assign anomaly scores to images that differ from learned normal appearance.

## Web-Data Curation Track

The web-data track is currently a scaffold:

- config: `configs/web_collection/web_collection.yaml`
- reports: `reports/web_collection/`
- notebook namespace: `notebooks/04_web_data_curation/`

No web collection implementation is claimed in this cleanup.

## Risk Notes

- Dataset path conventions still include both `data/manifests/` and legacy `data/processed/`.
- Historical metrics exist in reports and artifacts, but this cleanup did not recompute them.
- The full anomaly model is intentionally not implemented yet.
- Future refactors should be tested against `pytest -q` and at least one dry-run import check.

