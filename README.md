# TyreVisionX

TyreVisionX is an early-stage academic research prototype for tyre defect inspection. The repository currently supports supervised binary classification of tyre images and basic research reporting utilities. It is not a production inspection system, and several future directions remain experimental or planned rather than validated.

## Current Maturity
- Stage: early academic research prototype
- Validated focus: manifests, supervised binary classification, evaluation, error analysis, research reporting
- Not yet validated as a full platform for multi-dataset robustness, localization, segmentation, or deployment-grade serving

## Research Motivation
The project targets defect-sensitive tyre inspection where missed defects matter more than cosmetic false alarms. The current codebase is structured to support reproducible classification experiments first, then expand into robustness studies, anomaly detection, and localization-oriented work.

## Implemented Now
- Config-driven training and evaluation for binary classification
- ResNet-18 and ResNet-34 classification configs
- Canonical manifest-driven dataset loading under `src/data/`
- Confusion matrix and ROC/PR report generation
- FastAPI inference endpoint and Streamlit QA app
- Batch inference and export utilities
- Historical baseline path for SimpleCNN and frozen-feature experiments
- Project status reports, decisions log, changelog, and experiment log

## Current Limitations
- D1 is the only dataset with a populated tracked manifest in this checkout
- D2 and D3 remain scaffolded in config but not populated with tracked manifests
- Historical baseline results are stronger documented than current canonical experiment artifacts
- Service tests require optional multipart support in the environment
- Legacy baseline code is retained for reproducibility and comparison, not as the canonical pipeline

## Repository Structure
```text
TyreVisionX/
├── configs/
│   ├── aug/
│   │   ├── light.yaml
│   │   └── strong.yaml
│   ├── data/
│   │   └── datasets.yaml
│   ├── train/
│   │   ├── train_resnet18.yaml
│   │   └── train_resnet34.yaml
│   ├── aug_light.yaml            # compatibility copy
│   ├── aug_strong.yaml           # compatibility copy
│   ├── data.yaml                 # compatibility copy
│   ├── train_resnet18.yaml       # compatibility copy
│   └── train_resnet34.yaml       # compatibility copy
├── data/
│   ├── manifests/
│   ├── processed/
│   ├── raw/
│   └── README.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PROJECT_STATUS.md
│   ├── codex/
│   └── project/
├── logs/
│   ├── CHANGELOG.md
│   ├── DECISIONS.md
│   ├── EXPERIMENT_LOG.md
│   └── TODO_NEXT.md
├── reports/
│   ├── project_status/
│   │   ├── current_status.md
│   │   └── repo_audit.md
│   └── research_notes/
│       └── concepts_used.md
├── scripts/
├── src/
│   ├── data/                     # canonical data path
│   ├── legacy/                   # archived baseline namespace
│   ├── models/
│   ├── utils/
│   ├── train.py                  # canonical training entry point
│   ├── evaluate.py               # canonical evaluation entry point
│   ├── service_fastapi.py
│   └── app_streamlit.py
├── tests/
├── artifacts/                    # generated outputs only, mostly gitignored
├── AGENTS.md
├── PROGRESS_LOG.md
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Canonical Pipeline
The canonical current pipeline is the config-driven classification path:

1. Dataset config: `configs/data/datasets.yaml`
2. Training config: `configs/train/train_resnet18.yaml` or `configs/train/train_resnet34.yaml`
3. Training entry point: `python -m src.train`
4. Evaluation entry point: `python -m src.evaluate`
5. Dataset loading: `src/data/datasets.py`
6. Transforms: `src/data/transforms.py`

Legacy baseline components are archived under `src/legacy/` and kept for historical reproducibility.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
```

## Data Preparation Workflow
Dataset guidance:
- D1 TyreNet should be placed under `data/D1_tyrenet/`
- D2 should be placed under `data/D2_tire_crack/`
- D3 should be placed under `data/D3_tyre_quality/`

Download guidance:
```bash
python scripts/download_datasets.py
```

Prepare the canonical D1 manifest:
```bash
python scripts/prepare_manifests.py \
  --dataset_root data/D1_tyrenet \
  --dataset_id D1 \
  --good_dir good \
  --defect_dir defect \
  --out_csv data/manifests/D1_tyrenet_manifest.csv
```

Refresh split columns for configured manifests:
```bash
python scripts/prepare_folds.py --config configs/data/datasets.yaml
```

## Training Workflow
Default canonical training command:
```bash
make train
```

Explicit command:
```bash
python -m src.train --config configs/train/train_resnet18.yaml
```

Alternative active model:
```bash
python -m src.train --config configs/train/train_resnet34.yaml
```

## Evaluation Workflow
```bash
python -m src.evaluate \
  --checkpoint artifacts/experiments/resnet18_tyrenet_v1/best.pt \
  --split test
```

The evaluation path uses the same runtime dataset contract as training. Direct-manifest fallback is still supported for compatibility, but the canonical path is the dataset config plus manifest set under `configs/data/datasets.yaml`.

## Reports and Notebooks
Project-level status files:
- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `reports/project_status/repo_audit.md`
- `reports/project_status/current_status.md`
- `logs/DECISIONS.md`
- `logs/EXPERIMENT_LOG.md`

Currently present notebooks:
- `notebooks/dataset_exploration.ipynb`
- `notebooks/day3_baseline_training.ipynb`
- `notebooks/day5_regularization_bn_dropout.ipynb`

The professor-ready notebook set recommended in the cleanup plan is not yet fully generated in this refactor.

## Serving and Utilities
FastAPI:
```bash
make serve
```

Streamlit QA app:
```bash
make app
```

Batch inference:
```bash
python scripts/demo_infer.py \
  --model artifacts/experiments/resnet18_tyrenet_v1/best.pt \
  --input_dir path/to/images \
  --output_csv artifacts/demo_results.csv
```

Export:
```bash
make export
```

## Current Results Summary
Only historical results already documented in committed reports are summarized here.

Historical baseline observations:
- `reports/day3_baseline_observations.md` records a strong D1 baseline for `frozen_resnet50` with partial fine-tuning over 3 epochs:
  - accuracy `0.9804`
  - defect recall `0.9692`
  - defect F1 `0.9805`
  - AUROC `0.9985`
- `reports/day5_regularization_observations.md` records that, across the longer multi-seed SimpleCNN comparison:
  - baseline had the highest mean recall but high variance
  - augmentation had the most stable precision / AUPRC trade-off

These are historical baseline results from the legacy comparison path, not claims about the cleaned canonical pipeline after this refactor.

## Experimental and Planned Work
Experimental or partial:
- CNN→GNN option in `src/models/cnn_gnn.py`
- multi-dataset configuration across D1, D2, D3
- registry-backed model loading when experiment outputs are present

Planned:
- anomaly detection studies
- localization and segmentation
- cross-dataset robustness experiments with populated D2/D3 manifests
- multi-view 3D reconstruction
- defect projection to mesh
- knowledge-graph-style reasoning

## Discussion Topics For Professor Meetings
- Which metric should be primary for the next research milestone: recall, F1, or calibrated operating threshold?
- Whether D2/D3 should be integrated next for robustness, or whether D1 error analysis should be deepened first
- Whether the next milestone should stay in classification or branch into anomaly detection / localization
- How much historical baseline code should remain user-facing versus archived-only

## Maintenance Notes
- The active path is `src/train.py` plus `src/evaluate.py`.
- The legacy baseline path is archived under `src/legacy/`.
- Root-level config files are retained as compatibility copies; new work should use `configs/aug/`, `configs/data/`, and `configs/train/`.
- Do not add reported metrics unless they are backed by committed reports or reproducible artifacts.
