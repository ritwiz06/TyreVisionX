# Day 3 Baseline Observations

## Run Configuration
- Model: SimpleCNN
- Image size: 224
- Augmentation preset: light
- Epochs: 1
- Batch size: 16
- Optimizer: AdamW (default)

## Metrics (Test Split)
- Accuracy: 0.7647
- Precision (macro): 0.8296
- Recall (macro): 0.7603
- F1 (macro): 0.7504
- F1 (defect): 0.8101
- AUROC: 0.8631

## Confusion Matrix Notes
- False negatives (defect predicted as good): 2
- False positives (good predicted as defect): 58

## Failure Cases
- Example filenames to review: _TBD_

## Observations & Next Steps
- One-epoch baseline shows decent defect F1 but high false positives; run longer training and inspect error cases.

## Run 2 (SimpleCNN v2)

### Run Configuration
- Image size: 224
- Augmentation preset: light
- Epochs: 2
- Batch size: 16
- Output dir: artifacts/day3_baseline/simple_cnn_v2

### Metrics (Test Split)
- Accuracy: 0.8353
- Precision (macro): 0.8559
- Recall (macro): 0.8329
- F1 (macro): 0.8321
- F1 (defect): 0.8552
- AUROC: 0.8893
- AUPRC: 0.8650

### Confusion Matrix Notes
- False negatives (defect predicted as good): 6
- False positives (good predicted as defect): 36

## Run 3 (Option B: Frozen ResNet18 + Linear Head)

### Run Configuration
- Model: frozen_resnet18 (pretrained backbone, frozen weights, trainable linear head)
- Image size: 224
- Augmentation preset: light
- Epochs: 2
- Batch size: 16
- Output dir: artifacts/day3_baseline/frozen_resnet18_v1

### Metrics (Test Split)
- Accuracy: 0.9333
- Precision (macro): 0.9337
- Recall (macro): 0.9331
- F1 (macro): 0.9333
- F1 (defect): 0.9354
- AUROC: 0.9802
- AUPRC: 0.9811

### Confusion Matrix Notes
- False negatives (defect predicted as good): 7
- False positives (good predicted as defect): 10

### Interpretation
- Defect/non-defect appears strongly separable with pretrained frozen features on D1.
- This is a good signal for moving to stronger backbones or fine-tuning in the next step.

## Run 4 (Stronger Follow-up: ResNet18 + Unfrozen Layer4)

### Run Configuration
- Model: frozen_resnet18 with `unfreeze_last_block=true` (feature extractor + partial fine-tuning)
- Image size: 224
- Augmentation preset: light
- Epochs: 2
- Batch size: 16
- Output dir: artifacts/day3_baseline/frozen_resnet18_unfreeze_v2

### Training Dynamics
- Epoch 1: Train loss 0.2706 | Val loss 0.1176 | Val recall(defect) 0.9385 | Gap(train-val) 0.1529
- Epoch 2: Train loss 0.1153 | Val loss 0.1082 | Val recall(defect) 0.9769 | Gap(train-val) 0.0071

### Metrics (Test Split)
- Accuracy: 0.9647
- Precision (macro): 0.9657
- Recall (macro): 0.9643
- F1 (macro): 0.9647
- Precision (defect): 0.9481
- Recall (defect): 0.9846
- F1 (defect): 0.9660
- AUROC: 0.9933
- AUPRC: 0.9934

### Confusion Matrix Notes
- Matrix: [[118, 7], [2, 128]]
- False negatives (defect predicted as good): 2
- False positives (good predicted as defect): 7

### Failure Analysis Artifacts
- Misclassified rows: `artifacts/day3_baseline/frozen_resnet18_unfreeze_v2/misclassified.csv`
- False negatives rows: `artifacts/day3_baseline/frozen_resnet18_unfreeze_v2/false_negatives.csv`
- Misclassified visual grid: `artifacts/day3_baseline/frozen_resnet18_unfreeze_v2/misclassified_grid.png`
- False-negative visual grid: `artifacts/day3_baseline/frozen_resnet18_unfreeze_v2/false_negative_grid.png`

## Run 5 (Requested ResNet50 Run: Pretrained + Unfrozen Layer4)

### Run Configuration
- Model: frozen_resnet50 with `unfreeze_last_block=true`
- Image size: 224
- Augmentation preset: light
- Epochs: 1
- Batch size: 16
- Output dir: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v1`

### Training Dynamics
- Epoch 1: Train loss 0.2906 | Val loss 0.0774 | Val recall(defect) 1.0000 | Gap(train-val) 0.2132

### Metrics (Test Split)
- Accuracy: 0.9529
- Precision (macro): 0.9564
- Recall (macro): 0.9522
- F1 (macro): 0.9528
- Precision (defect): 0.9214
- Recall (defect): 0.9923
- F1 (defect): 0.9556
- AUROC: 0.9961
- AUPRC: 0.9965

### Confusion Matrix Notes
- Matrix: [[114, 11], [1, 129]]
- False negatives (defect predicted as good): 1
- False positives (good predicted as defect): 11

### Failure Analysis Artifacts
- Misclassified rows: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v1/misclassified.csv`
- False negatives rows: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v1/false_negatives.csv`
- Misclassified visual grid: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v1/misclassified_grid.png`
- False-negative visual grid: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v1/false_negative_grid.png`

## Run 6 (Cleaner Comparison: ResNet50, 3 Epochs)

### Run Configuration
- Model: frozen_resnet50 with `unfreeze_last_block=true`
- Image size: 224
- Augmentation preset: light
- Epochs: 3
- Batch size: 16
- Output dir: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v2_3ep`

### Training Dynamics
- Epoch 1: Train loss 0.2867 | Val loss 0.1076 | Val recall(defect) 1.0000 | Gap(train-val) 0.1791
- Epoch 2: Train loss 0.0920 | Val loss 0.0646 | Val recall(defect) 0.9692 | Gap(train-val) 0.0274
- Epoch 3: Train loss 0.0789 | Val loss 0.0507 | Val recall(defect) 0.9615 | Gap(train-val) 0.0282

### Metrics (Test Split)
- Accuracy: 0.9804
- Precision (macro): 0.9804
- Recall (macro): 0.9806
- F1 (macro): 0.9804
- Precision (defect): 0.9921
- Recall (defect): 0.9692
- F1 (defect): 0.9805
- AUROC: 0.9985
- AUPRC: 0.9986

### Confusion Matrix Notes
- Matrix: [[124, 1], [4, 126]]
- False negatives (defect predicted as good): 4
- False positives (good predicted as defect): 1

### Failure Analysis Artifacts
- Misclassified rows: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v2_3ep/misclassified.csv`
- False negatives rows: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v2_3ep/false_negatives.csv`
- Misclassified visual grid: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v2_3ep/misclassified_grid.png`
- False-negative visual grid: `artifacts/day3_baseline/frozen_resnet50_unfreeze_v2_3ep/false_negative_grid.png`

### Comparison vs Run 5 (1 Epoch)
- 3-epoch run improves overall accuracy and macro scores.
- 3-epoch run reduces false positives (11 -> 1) but increases false negatives (1 -> 4).
- Because production priority is defect recall, this tradeoff suggests threshold tuning and recall-weighted loss are needed before selecting this checkpoint.
