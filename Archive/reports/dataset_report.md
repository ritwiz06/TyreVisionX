# TyreVisionX Dataset Report (Day 2)

## ML Task Definition
- Task: supervised image classification
- Input: tyre image (RGB)
- Output: defect label (binary; good=0, defect=1)

## Dataset Summary
- Total images: 1698
- Class counts: good 832, defect 866
- Split ratios: train 0.70, val 0.15, test 0.15
- Split counts per class: train (defect 606, good 582), val (defect 130, good 125), test (defect 130, good 125)
- Imbalance notes: near-balanced; defect ~51.1% of total

## Image Properties
- Resolution range (min/max): 1100x1100 to 2268x2268
- Lighting/contrast: _TBD_ (inspect visually)
- Texture/defect subtlety: _TBD_ (inspect visually)
- Defect localization difficulty: _TBD_ (inspect visually)

## Ground Truth Quality
- Ambiguous examples: _TBD_ (list file paths)
- Suspected mislabeled samples: _TBD_ (list file paths)
- Notes on label noise: _TBD_

## Input Decisions
- Color space: RGB (default)
- img_size: 384 (default; consider 224 for faster iteration)
- Normalization: ImageNet mean/std

## Metrics & Risks
- Primary focus: Recall(defect) and F1(defect)
- Risk: false negatives (defects missed) are high impact
- Secondary: AUROC, precision, macro F1

## Concept Map (TyreVisionX)
| ML Concept | TyreVisionX Mapping |
| --- | --- |
| Positive class | defect (1) |
| Negative class | good (0) |
| Label source | folder name (good/defect) |
| Primary metric | Recall(defect), F1(defect) |
| Error cost | false negatives are critical |

## Next Steps (Day 3)
1. Train baseline ResNet-18 with light augmentations.
2. Evaluate on test split; record F1(defect), Recall(defect), AUROC.
3. Review failure cases and update Ground Truth Quality section.
