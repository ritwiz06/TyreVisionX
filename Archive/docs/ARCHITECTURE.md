# Architecture

Date: 2026-04-06

## Overview
TyreVisionX currently centers on one canonical image-classification pipeline and one archived legacy baseline path.

The active architectural goal is:
- manifest-driven data ingestion
- supervised binary classification
- reproducible evaluation and error analysis
- research-facing reporting

## Active Pipeline
### 1. Configuration layer
Canonical config locations:
- `configs/data/datasets.yaml`
- `configs/train/train_resnet18.yaml`
- `configs/train/train_resnet34.yaml`
- `configs/aug/light.yaml`
- `configs/aug/strong.yaml`

Compatibility copies remain at the root of `configs/`, but new work should use the canonical subdirectories.

### 2. Data layer
Primary module:
- `src/data/datasets.py`

Responsibilities:
- load manifest CSVs
- resolve split filtering
- resolve dataset-root-relative image paths
- support one direct-manifest compatibility fallback
- support multi-manifest loading through the dataset config

Primary transform module:
- `src/data/transforms.py`

### 3. Model layer
Current implemented model families:
- ResNet classifiers via `src/models/resnet_classifier.py`
- optional CNN→GNN path via `src/models/cnn_gnn.py`

Historical baseline models still exist for comparison:
- `src/models/simple_cnn.py`
- `src/models/feature_extractor.py`

### 4. Training layer
Primary entry point:
- `src/train.py`

Key behavior:
- reads a train config
- resolves datasets through the shared runtime data contract
- trains a classifier
- saves `best.pt`, `config.yaml`, metrics, and class index
- optionally registers the run in the model registry

### 5. Evaluation layer
Primary entry point:
- `src/evaluate.py`

Key behavior:
- loads the saved checkpoint config
- resolves the evaluation dataset through the same runtime data contract as training
- computes metrics and confusion matrix
- writes ROC, PR, and confusion artifacts

### 6. Serving and reporting
Implemented surfaces:
- `src/service_fastapi.py`
- `src/app_streamlit.py`
- `scripts/demo_infer.py`
- `src/export.py`

These are useful utilities, but they depend on trained experiment outputs being present in the local workspace.

## Legacy Pipeline
Archived legacy namespace:
- `src/legacy/`

Legacy-compatible components:
- `src/legacy/train_baseline.py`
- `src/legacy/eval_baseline.py`
- `src/legacy/dataset.py`
- `src/legacy/transforms.py`

Historical root-level compatibility imports remain:
- `src/train_baseline.py`
- `src/eval_baseline.py`
- `src/dataset.py`
- `src/transforms.py`

Reason:
- preserve reproducibility for Day 3 / Day 5 baseline work
- avoid breaking older scripts while making the active path obvious

## Data Contract
The active pipeline now uses one shared contract:

1. Prefer `data.config_file`
2. Load dataset definitions from `configs/data/datasets.yaml`
3. Select dataset manifests by configured dataset ID
4. Resolve image paths against the dataset root when manifest paths are dataset-root-relative

Compatibility fallback:
- `data.manifest_csv` still works for older runs, but it is no longer the default active path

## Status By Layer
Implemented now:
- config-driven classification pipeline
- dataset manifests and split management
- baseline reporting
- serving utilities

Partial or experimental:
- multi-dataset evaluation
- CNN→GNN path
- registry-backed serving with real experiment outputs

Planned:
- anomaly detection
- localization and segmentation
- multi-view learning and 3D reconstruction
- knowledge graph / root-cause reasoning
