# TyreVisionX Base Context for Future Codex Prompts

Updated: 2026-04-19

TyreVisionX is an early-stage graduate research repository for tyre inspection using AI/ML and computer vision. It started as a supervised binary classifier for tyre images (`good = 0`, `defect = 1`). The current professor-directed research expansion is to add a good-only anomaly-detection track and later use that baseline to help curate additional likely-good tyre images collected from approved web sources.

## Current State

Implemented:
- supervised binary classification code
- canonical ResNet training/evaluation path
- historical SimpleCNN and feature-extractor experiments
- data manifest tooling
- baseline reports and notebooks
- first good-only anomaly baseline code path
- D1 anomaly manifests under `data/manifests/`
- web-image curation policy, query catalog, manual-provider scaffold, filtering, and review queue code
- manual tyre pilot orchestration and promotion-path code
- safe source-acquisition docs, provider source config, manual Google-discovery CSV import, and provider smoke-check scaffolding

Partial/experimental:
- multi-dataset support
- CNN to GNN work
- inference services
- legacy Day 3/Day 5 baseline path

Planned:
- false-negative review and stronger anomaly follow-up after the first D1 run
- first approved 20-50 tyre web-candidate ingestion and human review run
- localization and segmentation
- multi-view / 3D / knowledge-reasoning research

## Canonical Supervised Path

Use this path for future supervised work unless a prompt says otherwise:

- `configs/data/datasets.yaml`
- `configs/train/train_resnet18.yaml`
- `configs/train/train_resnet34.yaml`
- `src/train.py`
- `src/evaluate.py`
- `src/data/datasets.py`
- `src/data/transforms.py`

## Historical / Legacy Path

Preserve but do not treat as canonical:

- `src/train_baseline.py`
- `src/eval_baseline.py`
- `src/dataset.py`
- `src/transforms.py`
- `src/models/simple_cnn.py`
- `src/models/feature_extractor.py`
- `scripts/day5_seed_sweep.py`
- `src/legacy/`

## Result Policy

Do not fabricate results. If artifacts are missing or a run was not executed in the current prompt, label the result as historical reported or pending recomputation.

## Current Anomaly Path

The first anomaly implementation uses:

- manifests:
  - `data/manifests/D1_anomaly_train_normal.csv`
  - `data/manifests/D1_anomaly_val_mixed.csv`
  - `data/manifests/D1_anomaly_test_mixed.csv`
- config: `configs/anomaly/anomaly_baseline.yaml`
- CLI: `scripts/anomaly/run_anomaly_baseline.py`
- modules: `src/anomaly/`

The preferred first run uses pretrained ResNet-18 embeddings plus Mahalanobis distance. As of 2026-04-19, it executed on D1 using cached local weights:

- run dir: `artifacts/anomaly/d1_resnet18_mahalanobis_v1`
- AUROC: `0.7194`
- AUPRC: `0.7212`
- anomaly recall: `0.2846`
- false negatives: `93`

This is a real baseline, but not strong enough for recall-critical tyre inspection.

False-negative analysis artifacts now exist under `reports/anomaly/`. The web pipeline must not use this anomaly model for automatic labeling; scores are advisory for review prioritization only.

As of 2026-04-20, a controlled anomaly benchmark has also run. The current best executed tyre-first variant is ResNet50 frozen embeddings plus kNN scoring:

- run dir: `artifacts/anomaly/benchmark/resnet50_knn`
- AUROC: `0.9298`
- AUPRC: `0.9339`
- anomaly recall: `0.8231`
- false negatives: `23`

This is a major improvement over the first baseline, but it is still not production-ready for recall-critical inspection.

As of 2026-04-20, a corruption robustness benchmark has also run. `resnet50_knn` degrades moderately under realistic corruptions. A mild noise-robust ResNet50+kNN variant preserved clean recall and reduced false positives, but did not reduce clean false negatives.

As of 2026-04-20, a controlled local-feature benchmark has also run:

- `resnet50_knn_reference`: recall `0.8231`, false negatives `23`, false positives `15`
- `resnet50_knn_threshold_sweep`: recall `0.9462`, false negatives `7`, false positives `36`
- `resnet50_multicrop_knn`: recall `0.8231`, false negatives `23`, false positives `20`
- `resnet50_patch_grid_knn_fine`: recall `0.4692`, false negatives `69`, false positives `19`

The current best recall-oriented anomaly candidate is `resnet50_knn_threshold_sweep`. The simple local crop/patch variants did not reduce false negatives enough, so the next local-feature direction should be a more principled patch-aware method rather than naive resized patch scoring.

As of 2026-04-20, a first feature-map patch-aware benchmark has also run:

- `resnet50_knn_threshold_sweep_reference`: recall `0.9462`, false negatives `7`, false positives `36`
- `resnet50_featuremap_patch_knn`: recall `0.1000`, false negatives `117`, false positives `6`
- `resnet50_featuremap_patch_knn_threshold_sweep`: recall `0.4615`, false negatives `70`, false positives `30`
- `resnet50_patchcore_lite`: recall `0.2385`, false negatives `99`, false positives `11`

The high-recall pooled ResNet50 kNN reference remains best. The first layer4 feature-map patch-memory implementation did not improve recall, so future patch-aware work should focus on better feature layers and score normalization rather than treating this run as a final rejection of local evidence.

As of 2026-04-20, that lower/mid-level follow-up has also run:

- `resnet50_layer3_patch_knn`: recall `0.1231`, false negatives `114`, false positives `12`
- `resnet50_layer3_patch_knn_threshold_sweep`: recall `0.3385`, false negatives `86`, false positives `41`
- `resnet50_layer2_layer3_patch_knn_threshold_sweep`: recall `0.2154`, false negatives `102`, false positives `38`

The pooled `resnet50_knn_threshold_sweep` reference still remains the best recall-oriented tyre anomaly candidate. Layer3 and layer2+layer3 patch-memory variants with robust score normalization did not improve false negatives. Future patch-aware work should diagnose patch score behavior before adding more variants.

External-data note: a Roboflow Universe registry and import-preparation scaffold now exists, but no Roboflow dataset has been downloaded or merged into D1. External data must remain separate until source, license, task type, labels, and augmentation/version risks are audited.

## Current Web-Curation Path

The web curation implementation uses:

- config: `configs/web_collection/web_collection.yaml`
- query catalog: `configs/web_collection/query_catalog.yaml`
- modules: `src/web_collection/`
- scripts: `scripts/web_collection/`
- policy docs:
  - `docs/process/WEB_COLLECTION_POLICY.md`
  - `docs/process/WEB_CURATION_WORKFLOW.md`

Only manual CSV/JSON import and provider stubs are implemented. Do not scrape Google HTML. Web images are research candidates, not ground-truth labels, until human review approves them.

The first manual pilot path exists:

- input folder: `data/external/manual_candidate_urls/`
- input template: `approved_tyres_pilot_urls_template.csv`
- CLI: `scripts/web_collection/run_manual_pilot.py`
- review pack exporter: `scripts/web_collection/export_review_pack.py`
- promotion CLI: `scripts/web_collection/promote_reviewed_candidates.py`

As of 2026-04-19, no real approved pilot CSV has been provided.

## Beginner-Friendly Requirement

Future prompts should explain important ML/repo concepts clearly enough for a beginner or professor review. Use plain language and connect each concept to TyreVisionX.
