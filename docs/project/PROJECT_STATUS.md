# TyreVisionX Project Status

Updated: 2026-04-19

TyreVisionX is an early-stage graduate research repository for tyre inspection using computer vision and machine learning. The repository currently contains a supervised binary classification track and historical baseline experiments. The new research direction adds an anomaly-detection track that trains only on good tyre images. The first D1 anomaly baseline has now executed with locally cached ResNet-18 weights.

## Implemented Now

### Supervised Binary Classification
- Task: classify tyre images as `good = 0` or `defect = 1`.
- Canonical training entry point: `src/train.py`.
- Canonical evaluation entry point: `src/evaluate.py`.
- Canonical data modules: `src/data/datasets.py` and `src/data/transforms.py`.
- Canonical configs:
  - `configs/data/datasets.yaml`
  - `configs/train/train_resnet18.yaml`
  - `configs/train/train_resnet34.yaml`
  - `configs/aug/light.yaml`
  - `configs/aug/strong.yaml`

### Historical Baseline Work
- SimpleCNN and frozen ResNet baseline scripts exist and are preserved.
- Day 3 and Day 5 reports contain historical reported results.
- These results are useful for research notes, but they are not newly recomputed by this cleanup prompt.

### Anomaly Baseline Scaffolding and Initial Implementation
- Good-only anomaly manifest generation is implemented.
- A first frozen-CNN embedding anomaly baseline is implemented under `src/anomaly/`.
- Mahalanobis scoring, kNN scoring, validation thresholding, and test-report export code exist.
- The first D1 run completed using local cached ResNet-18 weights.
- Test result: AUROC `0.7194`, AUPRC `0.7212`, anomaly recall `0.2846`, false negatives `93`.
- This is a valid first baseline but not sufficient for recall-critical inspection.
- False-negative analysis tooling and review artifacts now exist.
- A controlled anomaly benchmark now compares ResNet18/ResNet50, Mahalanobis/kNN, threshold sweep, and a first patch-grid variant.
- Current best executed anomaly candidate is `resnet50_knn`: recall `0.8231`, false negatives `23`, AUROC `0.9298`, AUPRC `0.9339`.
- This is improved, but still not sufficient for a production inspection claim.
- A corruption robustness benchmark now exists. `resnet50_knn` degrades moderately under realistic nuisance corruptions.
- A mild noise-robust `resnet50_knn` variant preserved clean recall and reduced false positives, but did not reduce clean false negatives.
- A ResNet50 local-feature benchmark now exists. Validation-only threshold refinement reduced false negatives to `7`; simple multicrop did not reduce false negatives; the current fine patch-grid variant performed worse.
- Current best recall-oriented anomaly candidate is `resnet50_knn_threshold_sweep`: recall `0.9462`, false negatives `7`, false positives `36`.
- A first feature-map patch-aware benchmark now exists. The tested layer4 patch-memory variants did not beat the threshold-swept reference.
- A lower/mid-level patch-aware follow-up now exists. The tested layer3 and layer2+layer3 patch-memory variants with robust score normalization also did not beat the threshold-swept pooled ResNet50 kNN reference.
- The high-recall pooled ResNet50 kNN reference remains the best current anomaly candidate.

### Documentation and Reporting
- Project status, architecture, roadmap, audit, and historical-result documents are now separated.
- Timestamped Codex work/process logs are now required for future prompts.
- Notebook folders now follow a research workflow structure.

### Web Data Curation Foundation
- Web collection policy and workflow docs now exist.
- A seed query catalog for likely-normal tyre images now exists.
- Manual CSV/JSON candidate URL import is implemented.
- Provider stubs exist for future approved search APIs.
- Candidate metadata, deduplication, quality filtering, anomaly-triage status, and human-review queue scaffolding are implemented.
- Manual pilot orchestration, local-file input support, review-pack export, and promotion-path code are implemented.
- A safe source-acquisition layer now exists with source-tier docs, provider source config, manual Google-discovery CSV import, provider smoke checks, and source/license metadata fields.
- No real approved 20-50 image pilot input CSV has been provided yet, so no real web-candidate ingestion or review run has been completed.

## Partial / Experimental

### Legacy Baseline Path
- Files such as `src/train_baseline.py`, `src/eval_baseline.py`, `src/dataset.py`, and `src/transforms.py` remain for compatibility with earlier Day 3/Day 5 experiments.
- Similar wrappers exist in `src/legacy/`.
- This path should not be treated as the canonical supervised pipeline for future research without review.

### Multi-Dataset Work
- D1 is the only clearly populated local dataset in this checkout.
- D2 and D3 are configured or scaffolded, but not validated as active research datasets here.
- Cross-dataset robustness remains planned.

### CNN to GNN Direction
- `src/models/cnn_gnn.py` exists as an optional, dependency-gated experiment path.
- It is not currently the main research baseline.

### Inference Apps
- FastAPI and Streamlit code exists.
- These apps require trained checkpoint artifacts to be useful and are not the current research priority.

## Planned / Future Work

### Anomaly Detection Next Steps
- Review the remaining `resnet50_knn_threshold_sweep` false negatives and higher false-positive load.
- Use validation-only threshold refinement as the current recall-oriented candidate setting.
- If patch-aware work continues, first diagnose why the current patch-memory scores are weak. The layer4, layer3, and layer2+layer3 patch-memory runs did not reduce false negatives.
- Use anomaly scores only as triage signals, not ground-truth labels.
- Do not use the current anomaly model to auto-label web candidates.

### External Dataset Readiness
- A Roboflow Universe external dataset registry now exists for careful shortlist tracking and import preparation.
- No external dataset has been downloaded, imported into canonical manifests, or merged with D1.
- Classification datasets are near-term candidates only after license/provenance review. Detection and segmentation datasets are later-phase localization resources.

### Web Data Curation
- Provide and run a small approved manual URL/local-file import.
- Deduplicate, quality-filter, and metadata-track collected images.
- Use anomaly scoring only as a review-priority signal after anomaly artifacts exist.
- Human-review candidates before any are used as likely-normal anomaly training data.

### Localization and Segmentation
- Add defect localization after classification/anomaly baselines are stable.
- Candidate future methods: YOLO-style detection, U-Net or Mask R-CNN segmentation.

### Multi-View / 3D / Reasoning
- Explore multi-view tyre inspection and possible 3D reconstruction after image-level baselines are reliable.
- Later research may connect detected/anomalous regions to knowledge reasoning about defect types and causes.

## Current Research Position

The repository is research-ready enough for supervised-baseline review, first anomaly-baseline discussion, and small manual web-curation experiments. It is not yet ready to claim a validated anomaly detector, production inspection pipeline, large-scale web-data collection system, or localization model.
