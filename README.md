# TyreVisionX

TyreVisionX is an early-stage graduate research repository for tyre inspection using computer vision and machine learning.

The project began as a supervised binary classifier for tyre images:

- `good = 0`
- `defect = 1`

The current research direction adds a planned good-only anomaly-detection track. That future track will learn normal tyre appearance from good images and score unusual images as likely anomalies. It may later help curate additional likely-good tyre images collected from approved web sources.

This repository is not a production inspection system. It is a research codebase with implemented supervised-baseline work, historical experiments, an executed first anomaly baseline, and early web-curation tooling.

## Current Status

Implemented now:
- supervised binary classification pipeline
- manifest-based data loading
- ResNet training/evaluation configs
- historical SimpleCNN and frozen-feature baseline work
- first good-only anomaly baseline code path and D1 execution
- evaluation utilities and baseline reports
- organized project docs, notebooks, and logs

Partial or experimental:
- legacy Day 3 / Day 5 baseline scripts
- multi-dataset support beyond the current local D1 data
- CNN to GNN experiment path
- FastAPI and Streamlit utilities that depend on available checkpoints

Planned:
- anomaly false-negative review and stronger anomaly experiments
- web-data curation workflow execution with real approved candidate URLs
- localization and segmentation
- multi-view / 3D / knowledge-reasoning research

## Cleaned Repository Structure

```text
TyreVisionX/
  configs/
    aug/                  # existing supervised augmentation configs
    data/                 # supervised dataset config
    train/                # supervised train configs
    anomaly/              # anomaly config scaffold
    web_collection/       # web-curation config scaffold
  data/
    manifests/            # manifest CSVs
    external/             # future external/raw collected data
    interim/              # future intermediate data products
    processed/            # legacy processed manifest path
    raw/                  # local raw image data, gitignored
  docs/
    architecture/
    codex/
    process/
    project/
  logs/
    work_logs/
    process_logs/
    experiment_logs/
  notebooks/
    00_project_overview/
    01_data_audit/
    02_supervised_baseline/
    03_anomaly_baseline/
    04_web_data_curation/
    05_results/
  reports/
    current_status/
    historical/
    anomaly/
    web_collection/
  scripts/
    anomaly/
    data/
    utilities/
  src/
    anomaly/
    data/
    evaluation/
    legacy/
    models/
    training/
    utils/
  tests/
```

Some older files and folders remain for compatibility and historical traceability. They are preserved intentionally rather than deleted.

## Canonical Supervised Track

Use this path for current supervised classification work:

- dataset config: `configs/data/datasets.yaml`
- train configs: `configs/train/train_resnet18.yaml`, `configs/train/train_resnet34.yaml`
- data loader: `src/data/datasets.py`
- transforms: `src/data/transforms.py`
- training: `src/train.py`
- evaluation: `src/evaluate.py`

Train:

```bash
python -m src.train --config configs/train/train_resnet18.yaml
```

Evaluate:

```bash
python -m src.evaluate \
  --checkpoint artifacts/experiments/resnet18_tyrenet_v1/best.pt \
  --split test
```

## Historical Baseline Work

The older Day 3 / Day 5 baseline path is preserved for research history:

- `src/train_baseline.py`
- `src/eval_baseline.py`
- `src/dataset.py`
- `src/transforms.py`
- `src/models/simple_cnn.py`
- `src/models/feature_extractor.py`
- `scripts/day5_seed_sweep.py`
- `src/legacy/`

Historical results are summarized in:

- `reports/historical/REPORTED_RESULTS.md`
- `reports/day3_baseline_observations.md`
- `reports/day5_regularization_observations.md`

Important: these are historical reported results unless explicitly labeled as recomputed after the cleanup. This cleanup did not train or recompute metrics.

## Anomaly Track

Current status: first baseline implemented and executed on D1 with cached ResNet-18 weights.

Created for future anomaly work:

- `configs/anomaly/anomaly_baseline.yaml`
- `src/anomaly/`
- `scripts/anomaly/`
- `reports/anomaly/`
- `notebooks/03_anomaly_baseline/anomaly_direction_plan.ipynb`
- `notebooks/03_anomaly_baseline/anomaly_baseline_results.ipynb`

Implemented first anomaly baseline:
- train only on good tyre images
- extract frozen CNN embeddings
- compute a normality/anomaly score
- select thresholds on validation data
- report final behavior once on test data

The default method is pretrained ResNet-18 pooled embeddings plus Mahalanobis distance. The first completed D1 run used cached ResNet-18 weights from the local Torch cache.

First D1 anomaly result:
- AUROC: `0.7194`
- AUPRC: `0.7212`
- anomaly recall: `0.2846`
- anomaly precision: `0.7708`
- false negatives: `93`
- false positives: `11`

This is a real baseline result, but recall is too low for a recall-critical tyre inspection claim. See `reports/anomaly/anomaly_results_summary.md`.

Controlled anomaly benchmark update:
- `resnet18_mahalanobis_reference`: reused baseline, recall `0.2846`, FN `93`
- `resnet18_knn`: recall `0.5231`, FN `62`
- `resnet18_threshold_sweep`: recall `0.5769`, FN `55`
- `resnet50_mahalanobis`: recall `0.7077`, FN `38`
- `resnet50_knn`: recall `0.8231`, FN `23`
- `resnet18_patch_grid_knn`: recall `0.3462`, FN `85`

Current next-best anomaly candidate: `resnet50_knn`. It is improved but still not production-ready.

Robustness update:
- realistic corruption benchmark completed for `resnet50_knn`, `resnet50_mahalanobis`, `resnet18_knn_control`, and `resnet50_knn_noise_robust`
- `resnet50_knn` degraded moderately under tested corruptions; worst tested recall was `0.7846`
- mild noise-robust `resnet50_knn` preserved clean recall (`0.8231`) and reduced clean false positives from `15` to `13`
- robustness training helped nuisance stability but did not reduce clean false negatives, so patch-aware/local-feature work remains the next main anomaly direction

Local-feature benchmark update:
- `resnet50_knn_threshold_sweep`: recall `0.9462`, FN `7`, FP `36`
- `resnet50_multicrop_knn`: recall `0.8231`, FN `23`, FP `20`
- `resnet50_patch_grid_knn_fine`: recall `0.4692`, FN `69`, FP `19`

Current best recall-oriented anomaly candidate: `resnet50_knn_threshold_sweep`. This was selected by validation-only threshold refinement and improves false negatives, but it increases false positives. The simple local crop/patch variants did not beat threshold refinement.

Patch-aware benchmark update:
- `resnet50_knn_threshold_sweep_reference`: recall `0.9462`, FN `7`, FP `36`
- `resnet50_featuremap_patch_knn`: recall `0.1000`, FN `117`, FP `6`
- `resnet50_featuremap_patch_knn_threshold_sweep`: recall `0.4615`, FN `70`, FP `30`
- `resnet50_patchcore_lite`: recall `0.2385`, FN `99`, FP `11`

Current decision: the threshold-swept pooled ResNet50 kNN reference remains best. The first feature-map patch-memory implementation did not improve tyre anomaly detection.

Lower/mid-level patch-aware follow-up:
- `resnet50_layer3_patch_knn`: recall `0.1231`, FN `114`, FP `12`
- `resnet50_layer3_patch_knn_threshold_sweep`: recall `0.3385`, FN `86`, FP `41`
- `resnet50_layer2_layer3_patch_knn_threshold_sweep`: recall `0.2154`, FN `102`, FP `38`

Current decision after the lower/mid-level follow-up: the threshold-swept pooled ResNet50 kNN reference still remains best. The layer3 and layer2+layer3 patch-memory variants with robust score normalization did not reduce false negatives. The next anomaly step should diagnose why patch-memory scores are weak before adding more patch variants.

## External Dataset Registry

Roboflow Universe has been audited only as a possible external data source. No Roboflow dataset has been downloaded or merged into D1.

Near-term candidates for careful supervised/anomaly support after license and provenance review:
- Good Tire Bad Tire
- Tires Defects

Later-phase candidates:
- Tire Tread, because it is detection/localization oriented.
- defect by Hemant, because it is detection/localization oriented.

Blocked/pending candidates:
- Tire Quality, pending explicit license verification.
- tire / College segmentation, pending exact source and license verification.

External datasets must stay separate from D1 until their licenses, labels, task type, and augmentation/version inflation risks are audited.

## Web-Data Curation Track

Current status: research-grade scaffold and manual/provider-stub pipeline implemented. A first manual pilot orchestrator now exists, but no approved pilot CSV has been provided yet, so no real candidate images have been ingested or reviewed.

Created for web curation:

- `configs/web_collection/web_collection.yaml`
- `configs/web_collection/query_catalog.yaml`
- `src/web_collection/`
- `scripts/web_collection/`
- `docs/process/WEB_COLLECTION_POLICY.md`
- `docs/process/WEB_CURATION_WORKFLOW.md`
- `reports/web_collection/`
- `notebooks/04_web_data_curation/`

The implemented workflow supports editable query catalogs, manual CSV/JSON URL import, future approved provider API stubs, candidate metadata tracking, download/copy metadata updates, deduplication and quality filtering, optional anomaly-triage status hooks, and human-review queue generation.

Source acquisition now includes a safe provider/manual discovery layer:
- `configs/web_collection/provider_sources.yaml`
- `docs/process/WEB_SOURCE_ACQUISITION_GUIDE.md`
- `docs/process/GOOGLE_MANUAL_DISCOVERY_WORKFLOW.md`
- `scripts/web_collection/import_manual_google_discovery.py`
- `scripts/web_collection/provider_smoke_check.py`

Google remains manual discovery only. Official-provider adapters are scaffolded for Wikimedia Commons, Pexels, Unsplash, and Flickr; Pexels/Unsplash/Flickr require user-provided API credentials.

Important: web candidates are not automatically labeled good. They remain research candidates until filtered and reviewed by a human. Do not scrape Google HTML; use manual import or approved provider APIs.

The current anomaly baseline is not reliable enough to label web images automatically. Model scores may only assist review prioritization.

First manual pilot:
- input template: `data/external/manual_candidate_urls/approved_tyres_pilot_urls_template.csv`
- guide: `docs/process/MANUAL_PILOT_INPUT_GUIDE.md`
- orchestrator: `scripts/web_collection/run_manual_pilot.py`
- current status: `reports/current_status/manual_pilot_status.md`

## Notebook Map

- `notebooks/00_project_overview/project_overview.ipynb`: project orientation
- `notebooks/01_data_audit/data_manifest_audit.ipynb`: manifest and dataset audit
- `notebooks/02_supervised_baseline/supervised_baseline_review.ipynb`: supervised baseline review
- `notebooks/03_anomaly_baseline/anomaly_direction_plan.ipynb`: anomaly plan
- `notebooks/03_anomaly_baseline/anomaly_baseline_results.ipynb`: anomaly baseline status/results placeholders
- `notebooks/03_anomaly_baseline/anomaly_model_benchmark.ipynb`: controlled anomaly benchmark summary
- `notebooks/03_anomaly_baseline/anomaly_corruption_benchmark.ipynb`: corruption robustness benchmark summary
- `notebooks/03_anomaly_baseline/anomaly_local_feature_benchmark.ipynb`: local-feature benchmark summary
- `notebooks/03_anomaly_baseline/false_negative_overlap_review.ipynb`: false-negative overlap review
- `notebooks/03_anomaly_baseline/high_recall_error_review.ipynb`: high-recall FN/FP review pack
- `notebooks/03_anomaly_baseline/high_recall_error_pattern_review.ipynb`: qualitative high-recall error pattern notes
- `notebooks/03_anomaly_baseline/anomaly_patch_aware_benchmark.ipynb`: patch-aware benchmark summary
- `notebooks/03_anomaly_baseline/anomaly_patch_layer_benchmark.ipynb`: lower/mid-level patch-aware follow-up
- `notebooks/03_anomaly_baseline/patch_false_negative_overlap_review.ipynb`: patch-aware overlap review
- `notebooks/04_web_data_curation/query_catalog_review.ipynb`: query catalog review
- `notebooks/04_web_data_curation/filtering_and_review.ipynb`: filtering and review workflow
- `notebooks/04_web_data_curation/review_workflow_demo.ipynb`: human review status workflow
- `notebooks/04_web_data_curation/manual_pilot_results.ipynb`: first manual pilot status/results
- `notebooks/04_web_data_curation/manual_pilot_review_pack.ipynb`: visual review pack workflow
- `notebooks/05_results/results_index.ipynb`: report/result index

Older useful notebooks were moved into the new folders with `legacy_` prefixes.

## Documentation Map

- `docs/project/PROJECT_STATUS.md`: implemented, partial, and planned work
- `docs/architecture/REPO_ARCHITECTURE.md`: repo layout and canonical paths
- `docs/project/ROADMAP.md`: research roadmap
- `reports/current_status/repo_audit.md`: current cleanup audit
- `reports/historical/REPORTED_RESULTS.md`: historical reported results
- `docs/codex/BASE_CONTEXT.md`: durable project context for future prompts
- `docs/codex/PROMPT_CONTRACT.md`: future Codex prompt requirements
- `docs/process/WEB_COLLECTION_POLICY.md`: web candidate collection boundaries
- `docs/process/WEB_CURATION_WORKFLOW.md`: step-by-step web curation pipeline
- `docs/project/PLATFORM_STRATEGY.md`: grounded future platform strategy

## Logging Conventions

Every substantial future prompt should create:

```text
logs/work_logs/WORK_LOG_<YYYYMMDD_HHMMSS>.md
logs/process_logs/PROCESS_LOG_<YYYYMMDD_HHMMSS>.md
```

And update:

```text
logs/work_logs/LATEST.md
logs/process_logs/LATEST.md
```

The log rules are defined in `docs/codex/PROMPT_CONTRACT.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or:

```bash
make setup
```

## Testing

```bash
pytest -q
```

If dependencies are missing in the local environment, install from `requirements.txt` first.

## Next Steps

1. Run tests after cleanup and fix any environment-specific issues.
2. Choose one future manifest convention for supervised and anomaly work.
3. Recompute a supervised baseline using the canonical `src/train.py` / `src/evaluate.py` path.
4. Create good-only training manifests for anomaly detection.
5. Review anomaly false negatives and compare kNN/threshold alternatives.
6. Create a manual URL CSV and run the web curation pipeline on a small approved sample.
7. Review candidates manually before using any likely-normal web images for anomaly training.
