# Interactive Report Outline

Date: 2026-04-06

## Purpose
This file describes the professor-facing interactive reporting view that the repository can support with its current artifacts and status files. It is intentionally honest about what is available now versus what is still pending.

## What An Interactive View Can Show Now
### Current status cards
- project stage: early-stage academic research prototype
- canonical pipeline: config-driven binary classification
- legacy path: archived baseline comparison path
- populated tracked dataset manifests: D1 only
- tracked D2/D3 manifests: present but empty

### Available result sources
- `reports/project_status/current_status.md`
- `reports/project_status/repo_audit.md`
- `logs/EXPERIMENT_LOG.md`
- `reports/day3_baseline_observations.md`
- `reports/day4_partc_tiny_cnn_observations.md`
- `reports/day5_regularization_observations.md`
- `artifacts/day3_baseline/*/eval_report.json`
- `artifacts/day5/longrun_seed_sweep/summary_by_model.csv`

### Available analysis sources
- misclassification CSV files under `artifacts/day3_baseline/`
- false-negative CSV files under `artifacts/day3_baseline/`
- test prediction CSV files under `artifacts/day5/step3_step4_*`

## What Should Be Labeled Pending
- fresh canonical-pipeline experiment results under `artifacts/experiments/`
- canonical D1 report generated after the cleanup refactor
- cross-dataset D2/D3 result comparison
- localization outputs
- segmentation outputs
- anomaly-detection outputs
- multi-view 3D outputs
- knowledge-reasoning outputs

## Suggested Sections
### 1. Project snapshot
- current maturity level
- implemented now
- partial / experimental
- future planned work

### 2. Dataset readiness
- manifest availability by dataset
- tracked row counts
- split availability
- pending datasets flagged clearly

### 3. Historical result summary
- Day 3 baseline table
- Day 5 stability table
- note that these are historical baseline results, not fresh canonical-pipeline outputs

### 4. Error analysis section
- links to confusion matrices
- counts of false negatives and false positives where tracked artifacts exist
- pending placeholders where canonical artifacts do not yet exist

### 5. Future work panel
- anomaly detection
- localization
- segmentation
- multi-view 3D
- knowledge reasoning

## Professor-Facing Narrative
The interactive report should make one main point clear:

TyreVisionX currently has a cleaner and more reproducible structure for binary classification research, but its strongest available tracked results are still historical baseline experiments rather than fresh outputs from the newly cleaned canonical pipeline.

## If A Future Dashboard Is Added
A lightweight dashboard could safely read:
- markdown status files from `docs/`, `logs/`, and `reports/project_status/`
- experiment summary CSV files already present in `artifacts/`
- evaluation JSON files where they exist

It should degrade gracefully by showing `Pending` instead of blank plots or invented metrics whenever expected artifacts are absent.
