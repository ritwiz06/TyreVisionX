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
