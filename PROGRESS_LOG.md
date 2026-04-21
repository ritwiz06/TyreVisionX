# TyreVisionX Progress Log

## Day 2 — Dataset understanding & task definition
- [x] Implemented scripts/prepare_manifests.py (CSV + stratified splits)
- [x] Implemented src/dataset.py (TyreManifestDataset + helpers)
- [x] Implemented src/transforms.py (light/strong/eval)
- [x] Created reports/dataset_report.md template
- [x] Created notebooks/dataset_exploration.ipynb (stats, visuals, QC checks)
- [ ] Run notebook on D1 and paste key stats into dataset_report.md
- [ ] Decide RGB vs grayscale + img_size + augmentation preset
- [ ] List suspected mislabeled/ambiguous samples (filenames)
- [x] .gitignore verified/updated (data + artifacts ignored)

## How to run Day 2
```bash
python scripts/prepare_manifests.py --dataset_root data/raw/D1_tyrenet --good_dir good --defect_dir defect --out_csv data/processed/D1_manifest.csv
# then open notebooks/dataset_exploration.ipynb and run all cells
```

## Day 3 — Baseline training
- [x] Implemented SimpleCNN baseline model
- [x] Added baseline training CLI (src/train_baseline.py)
- [x] Added baseline evaluation CLI (src/eval_baseline.py)
- [x] Created notebooks/day3_baseline_training.ipynb (runnable)
- [x] Created reports/day3_baseline_observations.md template
- Ran 1-epoch baseline at 224 with light aug; metrics saved under artifacts/day3_baseline/simple_cnn_v1.
- Ran Option B feature-extractor baseline (`frozen_resnet18`) for 2 epochs; artifacts saved under `artifacts/day3_baseline/frozen_resnet18_v1`.
- Added stronger follow-up support: `frozen_resnet50` option and `--unfreeze_last_block`.
- Added training logs for `train loss`, `val loss`, `val recall_defect`, and `train-val gap`.
- Added evaluation failure analysis exports: misclassified CSV/grid + false-negative CSV/grid.
- Ran stronger follow-up (`frozen_resnet18 --unfreeze_last_block`) and evaluated on test split; artifacts saved under `artifacts/day3_baseline/frozen_resnet18_unfreeze_v2`.
- Fixed pretrained ResNet50 cache issue by using project-local cache dirs (`TORCH_HOME`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`) and downloaded `resnet50-11ad3fa6.pth` into `artifacts/.torch/hub/checkpoints/`.
- Ran requested pretrained ResNet50 follow-up (`frozen_resnet50 --unfreeze_last_block`) and evaluated on test split; artifacts saved under `artifacts/day3_baseline/frozen_resnet50_unfreeze_v1`.
- Ran cleaner 3-epoch ResNet50 comparison (`frozen_resnet50 --unfreeze_last_block --epochs 3`) and evaluated on test split; artifacts saved under `artifacts/day3_baseline/frozen_resnet50_unfreeze_v2_3ep`.
- Added Day 3 baseline analysis + data-driven Day 4-5 improvement hypotheses to `README.md`.

## Day 4 — Part C (Tiny Phase-1 baseline)
- Updated `SimpleCNN` to the tiny Phase-1 architecture (Conv16 -> Conv32 -> Flatten -> Linear -> Sigmoid binary output) for fast baseline iteration.
- Updated baseline train/eval + metrics to support binary sigmoid outputs.
- Ran a smoke training under `artifacts/day3_baseline/simple_cnn_phase1_sigmoid_v1`.
- Recorded Day-4 observations in `reports/day4_partc_tiny_cnn_observations.md`.

## Day 5 — Baseline improvement (regularization + augmentation)
- [x] Added explicit Day-5 training augmentation preset in `src/transforms.py`:
  - Random crop (`RandomResizedCrop`, mild scale/ratio)
  - Horizontal flip
  - Small rotation (`±5°`)
  - Brightness/contrast jitter
- [x] Kept evaluation pipeline deterministic (`Resize + Normalize + ToTensorV2` only).
- [x] Verified optimizer already supports `weight_decay` via CLI in `src/train_baseline.py`.
- [x] Added albumentations 1.x/2.x compatibility for `RandomResizedCrop` and `CoarseDropout`.
- [x] Ran controlled 3-epoch comparison runs:
  - Control: `artifacts/day5/simple_cnn_control_noaug_nowd` (`preset=none`, `weight_decay=0`)
  - Aug-only: `artifacts/day5/simple_cnn_day5_aug_nowd` (`preset=day5`, `weight_decay=0`)
  - Aug+WD: `artifacts/day5/simple_cnn_day5_aug_wd1e4` (`preset=day5`, `weight_decay=1e-4`)
- Validation trend snapshot (epoch3):
  - Control: `val_recall_defect=0.9692`, `gap(train-val)=0.0214`
  - Aug-only: `val_recall_defect=0.7923`, `gap(train-val)=0.1067`
  - Aug+WD: `val_recall_defect=0.9385`, `gap(train-val)=0.1370`
- Test split summary (best checkpoints):
  - Control: recall(defect)=`0.9923`, FN=`1`, FP=`49`, F1(defect)=`0.8377`
  - Aug-only: recall(defect)=`0.7923`, FN=`27`, FP=`14`, F1(defect)=`0.8340`
  - Aug+WD: recall(defect)=`0.9462`, FN=`7`, FP=`31`, F1(defect)=`0.8662`
- Day-5 takeaway:
  - Augmentation + weight decay improved precision/F1 tradeoff and reduced false positives vs control.
  - Recall dropped vs control, so for recall-critical deployment keep threshold tuning and/or defect-weighted loss in next iteration.
- Step 3/4 run set completed (`3` epochs each, fresh outputs):
  - Baseline: `artifacts/day5/step3_step4_baseline`
  - + Augmentation: `artifacts/day5/step3_step4_aug`
  - + BatchNorm: `artifacts/day5/step3_step4_aug_batchnorm`
  - + Dropout(0.3): `artifacts/day5/step3_step4_aug_batchnorm_dropout03`
- Step 5 comparison (test split):
  - Baseline: recall=`0.9923`, precision=`0.7247`, FN=`1`, FP=`49`
  - + Augmentation: recall=`0.9846`, precision=`0.7665`, FN=`2`, FP=`39`
  - + BatchNorm: recall=`0.0000`, precision=`0.0000`, FN=`130`, FP=`0`
  - + Dropout: recall=`0.0000`, precision=`0.0000`, FN=`130`, FP=`0`
- Step 6 error-analysis artifacts exported per run:
  - `predictions_test.csv`
  - `misclassified.csv`
  - `false_negatives.csv`
  - `false_positives.csv`
- Added Day-5 report: `reports/day5_regularization_observations.md`
- Added Day-5 notebook for FN/FP visualization: `notebooks/day5_regularization_bn_dropout.ipynb`
- Fixed BN/Dropout failure:
  - Switched SimpleCNN training path to logits (`output_logits`) + `BCEWithLogitsLoss`.
  - Added model/eval compatibility for `output_logits` in `model_info.json`.
  - Tuned BN/Dropout runs at `lr=3e-4`:
    - `artifacts/day5/step3_step4_aug_batchnorm_fix`
    - `artifacts/day5/step3_step4_aug_batchnorm_dropout03_fix`
- Updated comparison after fix:
  - +BatchNorm: recall=`0.9077`, precision=`0.7329`, FN=`12`, FP=`43`
  - +Dropout: recall=`0.9538`, precision=`0.6392`, FN=`6`, FP=`70`
- Implemented long-run seed sweep automation (`scripts/day5_seed_sweep.py`):
  - 10 epochs + early stopping
  - threshold sweep (0.10-0.90) for recall-priority threshold selection
  - PR curve + AUPRC export per run
  - seeds: `7`, `123`, `999`
- Completed long-run runs across all variants:
  - Baseline, +Augmentation, +BatchNorm, +Dropout
  - outputs under `artifacts/day5/longrun_seed_sweep/`
  - summary tables:
    - `summary_runs.csv`
    - `summary_by_model.csv`
- Long-run conclusion:
  - Baseline has highest mean recall but high seed variance.
  - +Augmentation is the most stable and has strongest precision/AUPRC trade-off.

## 2026-04-19 — Cleanup, research structure, and anomaly scaffolding
- [x] Added current project status, architecture, roadmap, audit, and historical-results docs.
- [x] Added durable Codex context and prompt contract docs.
- [x] Added timestamped work/process logs for the cleanup prompt.
- [x] Reorganized useful notebooks into the new research workflow folders with `legacy_` prefixes.
- [x] Created professor-readable overview, data audit, supervised review, anomaly plan, and results index notebooks.
- [x] Added anomaly baseline scaffolding under `configs/anomaly/`, `src/anomaly/`, `scripts/anomaly/`, and `reports/anomaly/`.
- [x] Added web-collection scaffolding under `configs/web_collection/` and `reports/web_collection/`.
- [x] Refreshed README to distinguish implemented supervised work, historical results, and planned anomaly/web tracks.
- [x] Updated `.gitignore` so required logs, reports, and notebooks are trackable.
- [x] Added pytest config to ignore archived duplicate tests without deleting `Archive/`.
- [ ] Recompute supervised baseline using the cleaned canonical pipeline.
- [ ] Create good-only manifests for the anomaly baseline.

## 2026-04-19 — First anomaly baseline implementation
- [x] Adopted `data/manifests/` as canonical anomaly manifest location.
- [x] Created `scripts/data/create_anomaly_manifests.py`.
- [x] Generated D1 anomaly manifests:
  - `data/manifests/D1_anomaly_train_normal.csv`
  - `data/manifests/D1_anomaly_val_mixed.csv`
  - `data/manifests/D1_anomaly_test_mixed.csv`
- [x] Implemented anomaly modules under `src/anomaly/`:
  - manifest dataset
  - frozen ResNet embedding extractor
  - Mahalanobis scorer
  - kNN scorer
  - threshold selection
  - evaluation/reporting helpers
  - pipeline orchestration
- [x] Added CLI: `scripts/anomaly/run_anomaly_baseline.py`.
- [x] Added anomaly data contract, baseline report, status report, and results notebook.
- [x] Added smoke tests for anomaly manifests, scorers, thresholds, and imports.
- [x] Ran tests: `10 passed`, `1 skipped`, `2 warnings`.
- [ ] Full preferred pretrained anomaly run pending cached ResNet-18 weights.

## 2026-04-19 — Web-image curation foundation
- [x] Synced docs to reflect that anomaly manifests/code exist while the preferred run is blocked by missing cached ResNet-18 weights.
- [x] Added web collection policy and workflow docs.
- [x] Added grounded platform strategy doc.
- [x] Added editable query catalog generation for likely-normal tyre candidates.
- [x] Generated initial query catalog with 24 queries under `configs/web_collection/query_catalog.yaml`.
- [x] Implemented `src/web_collection/` modules:
  - metadata schemas
  - manual CSV/JSON provider
  - provider stubs for future approved search APIs
  - candidate IO
  - exact/perceptual dedupe and quality filters
  - human review queue helper
- [x] Added scripts under `scripts/web_collection/`:
  - generate query catalog
  - collect candidate metadata
  - download/copy candidates
  - filter candidates
  - anomaly triage status hook
  - build review queue
- [x] Added web collection reports, storage budget plan, filtering template, review guidelines, and review queue template.
- [x] Added optional local `feature_extractor.weights_path` support for future offline anomaly runs.
- [ ] Real web candidate ingestion pending an approved manual URL file or provider credentials.
- [ ] Anomaly triage on web candidates pending fitted anomaly artifacts and candidate scoring extension.

## 2026-04-19 — Manual tyre web-candidate pilot preparation
- [x] Verified Prompt 4 had not yet been executed; latest logs still pointed to the web-foundation stage.
- [x] Confirmed no real approved pilot CSV exists under `data/external/manual_candidate_urls/`.
- [x] Created manual pilot input guide and approved URL/local-path CSV template.
- [x] Added manual pilot orchestrator: `scripts/web_collection/run_manual_pilot.py`.
- [x] Added reviewed-candidate promotion script: `scripts/web_collection/promote_reviewed_candidates.py`.
- [x] Added review-pack exporter: `scripts/web_collection/export_review_pack.py`.
- [x] Added `src/web_collection/pilot.py` for reusable pilot orchestration, local-file ingestion, review queue creation, and promotion logic.
- [x] Ran the pilot orchestrator; result was blocked honestly because no approved input CSV exists.
- [x] Added blocked-status reports and empty pilot CSV templates under `reports/web_collection/`.
- [x] Added review decision schema and curated-normal promotion policy docs.
- [ ] Real 20-50 image pilot pending user-provided approved CSV.
- [ ] Visual review pack pending real review queue with local images.
- [ ] Curated likely-normal manifest pending human-reviewed `approved_likely_normal` decisions.

## 2026-04-19 — D1 anomaly baseline readiness and execution
- [x] Verified latest work/process logs point to Prompt 4 logs.
- [x] Added offline readiness checker: `scripts/anomaly/check_anomaly_readiness.py`.
- [x] Added offline weights guide: `docs/process/OFFLINE_ANOMALY_WEIGHTS_GUIDE.md`.
- [x] Found compatible cached ResNet18 weights:
  - `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
- [x] Ran D1 anomaly baseline with ResNet18 ImageNet embeddings + Mahalanobis scoring:
  - run dir: `artifacts/anomaly/d1_resnet18_mahalanobis_v1`
  - threshold: `12.147390651173238`
  - test AUROC: `0.7194`
  - test AUPRC: `0.7212`
  - test anomaly recall: `0.2846`
  - test anomaly precision: `0.7708`
  - test FN: `93`
  - test FP: `11`
- [x] Added anomaly readiness and results reports.
- [ ] Review false negatives visually before choosing the next anomaly method.

## 2026-04-19 — Anomaly false-negative analysis and safe web-normal policy
- [x] Added false-negative analysis script and report.
- [x] Generated false-negative review table and contact sheet from the completed D1 anomaly run.
- [x] Added safe anomaly variant plan.
- [x] Added configs for planned kNN and threshold-sweep variants.
- [x] Added no-auto-label web-normal expansion policy.
- [x] Added model-assisted triage policy.
- [x] Added review-priority confidence utility for web candidates.
- [x] Updated web anomaly triage status to reflect the D1 anomaly run while keeping web scoring advisory only.
- [ ] Run kNN scorer comparison.
- [ ] Run validation-only threshold sweep for higher recall.

## 2026-04-19 — Safe web source acquisition layer
- [x] Added source-acquisition docs for approved/manual tyre candidate discovery.
- [x] Added provider source config: `configs/web_collection/provider_sources.yaml`.
- [x] Added manual Google-discovery workflow and CSV template.
- [x] Added provider adapter scaffolds for Wikimedia Commons, Pexels, Unsplash, Flickr, and manual Google discovery.
- [x] Added provider smoke-check script.
- [x] Extended candidate metadata schema with source/license fields while keeping legacy aliases.
- [x] Added source strategy, metadata schema, and pilot source plan reports.
- [x] Added smoke tests for provider config parsing, manual Google import, API-key failure clarity, and metadata completeness.
- [ ] First real 30-50 source-acquisition pilot pending researcher-approved candidate CSV.

## 2026-04-20 — Manual Google discovery pack
- [x] Generated 40 browser-ready manual Google query links grouped by tread, sidewall, mounted tyre, inspection-like, and industrial/off-highway views.
- [x] Added manual browsing checklist for approve/reject decisions.
- [x] Created prefilled pilot CSV: `data/external/manual_candidate_urls/approved_tyres_pilot_urls_001_prefilled.csv`.
- [x] Added notebook: `notebooks/04_web_data_curation/manual_google_discovery_pack.ipynb`.
- [x] Added candidate selection rubric for likely-normal tyre images.
- [x] Added tests for discovery-pack generation, CSV columns, and query grouping.
- [ ] User must manually browse, fill source/page/license fields, and keep rows as candidates until review.

## 2026-04-20 — Controlled anomaly model benchmark
- [x] Added backbone readiness checker for ResNet18, ResNet50, EfficientNet-B0, ConvNeXt-Tiny, and ViT-B/16.
- [x] Confirmed local runnable weights for ResNet18 and ResNet50.
- [x] Marked EfficientNet-B0, ConvNeXt-Tiny, and ViT-B/16 as pending local weights/extractor support.
- [x] Added benchmark harness and config: `configs/anomaly/anomaly_benchmark.yaml`.
- [x] Reused the existing ResNet18 + Mahalanobis reference result.
- [x] Executed `resnet18_knn`, `resnet18_threshold_sweep`, `resnet50_mahalanobis`, `resnet50_knn`, and `resnet18_patch_grid_knn`.
- [x] Current best executed variant: `resnet50_knn` with recall `0.8231`, FN `23`, AUROC `0.9298`, AUPRC `0.9339`.
- [x] Added comparison, recommendation, and false-negative follow-up reports.
- [ ] Inspect remaining `resnet50_knn` false negatives visually.

## 2026-04-20 — Anomaly corruption robustness benchmark
- [x] Added realistic corruption benchmark config and guide.
- [x] Implemented corruption families: low/medium Gaussian noise, low/medium blur, mild JPEG compression, brightness shifts, and contrast shifts.
- [x] Ran corruption benchmark for `resnet50_knn`, `resnet50_mahalanobis`, and `resnet18_knn_control`.
- [x] Implemented and ran mild noise-robust `resnet50_knn` trained on normal-only augmented embeddings.
- [x] Evaluated the noise-robust variant under the same corruption suite.
- [x] Clean-trained `resnet50_knn` degraded moderately; worst tested recall was `0.7846`.
- [x] Noise-robust `resnet50_knn` preserved clean recall `0.8231` and reduced clean FP from `15` to `13`.
- [x] Added robustness comparison, recommendation, and false-negative follow-up reports.
- [x] Next anomaly priority started: inspect remaining false negatives and build patch-aware/local-feature ResNet50 variant.

## 2026-04-20 — ResNet50 local-feature anomaly benchmark
- [x] Added local-feature benchmark config: `configs/anomaly/anomaly_local_feature_benchmark.yaml`.
- [x] Implemented local-feature utilities:
  - `src/anomaly/multicrop.py`
  - `src/anomaly/local_features.py`
  - `src/anomaly/local_benchmark.py`
  - `scripts/anomaly/run_local_feature_benchmark.py`
- [x] Executed local-feature benchmark variants:
  - `resnet50_knn_reference`: recall `0.8231`, FN `23`, FP `15`
  - `resnet50_knn_threshold_sweep`: recall `0.9462`, FN `7`, FP `36`
  - `resnet50_multicrop_knn`: recall `0.8231`, FN `23`, FP `20`
  - `resnet50_patch_grid_knn_fine`: recall `0.4692`, FN `69`, FP `19`
- [x] Added local-feature reports, notebooks, and false-negative overlap analysis.
- [x] Current recall-oriented anomaly candidate: `resnet50_knn_threshold_sweep`.
- [x] Exported review assets for the remaining `7` false negatives and `36` false positives.

## 2026-04-20 — Patch-aware ResNet50 anomaly benchmark
- [x] Added high-recall error-review exporter and generated FN/FP tables/contact sheets.
- [x] Added patch-aware benchmark config: `configs/anomaly/anomaly_patch_aware_benchmark.yaml`.
- [x] Implemented feature-map patch extraction and patch memory scoring:
  - `src/anomaly/feature_map_patches.py`
  - `src/anomaly/patch_memory.py`
  - `src/anomaly/patchcore_lite.py`
  - `src/anomaly/patch_benchmark.py`
  - `scripts/anomaly/run_patch_aware_benchmark.py`
- [x] Executed patch-aware variants:
  - `resnet50_knn_threshold_sweep_reference`: recall `0.9462`, FN `7`, FP `36`
  - `resnet50_featuremap_patch_knn`: recall `0.1000`, FN `117`, FP `6`
  - `resnet50_featuremap_patch_knn_threshold_sweep`: recall `0.4615`, FN `70`, FP `30`
  - `resnet50_patchcore_lite`: recall `0.2385`, FN `99`, FP `11`
- [x] Added patch-aware reports, notebooks, and false-negative overlap analysis.
- [x] Current best candidate remains `resnet50_knn_threshold_sweep`.
- [ ] Review high-recall FP/FN contact sheets before another model iteration.

## 2026-04-20 — Lower/mid-level patch-aware follow-up and Roboflow registry
- [x] Reviewed high-recall FP/FN contact sheets for `resnet50_knn_threshold_sweep`.
- [x] Added lower/mid-level patch-aware configs:
  - `configs/anomaly/anomaly_patch_aware_layer3.yaml`
  - `configs/anomaly/anomaly_patch_aware_layer23.yaml`
- [x] Added robust patch-score normalization and layer-specific feature-map extraction support.
- [x] Executed lower/mid-level patch-aware variants:
  - `resnet50_knn_threshold_sweep_reference`: recall `0.9462`, FN `7`, FP `36`
  - `resnet50_layer3_patch_knn`: recall `0.1231`, FN `114`, FP `12`
  - `resnet50_layer3_patch_knn_threshold_sweep`: recall `0.3385`, FN `86`, FP `41`
  - `resnet50_layer2_layer3_patch_knn_threshold_sweep`: recall `0.2154`, FN `102`, FP `38`
- [x] Current best recall-oriented anomaly candidate remains `resnet50_knn_threshold_sweep`.
- [x] Added Roboflow external dataset registry and import-preparation scaffold.
- [x] Confirmed no Roboflow dataset was downloaded or merged into D1.
