# Concepts Used In TyreVisionX

Date: 2026-04-06

## How To Read This File
Each concept is summarized in three ways:
- simple explanation
- why it matters in TyreVisionX
- status: used now, partial, or planned

## Core Concepts
| Concept | Simple Explanation | Why It Matters Here | Status |
| --- | --- | --- | --- |
| CNN | A convolutional neural network learns visual patterns from images. | It is the main model family used for tyre image classification. | Used now |
| Convolution / filters | Small learned pattern detectors slide across the image. | Useful for edges, grooves, cracks, and texture disruptions on tyre surfaces. | Used now |
| Pooling | A layer that shrinks features while keeping strong signals. | Helps reduce sensitivity to exact pixel location in baseline CNNs. | Used now |
| Transfer learning | Reusing a model that already learned general image features. | ResNet backbones are initialized from ImageNet to reduce data requirements. | Used now |
| ResNet | A CNN with skip connections that helps deeper models train well. | Current canonical configs use ResNet-18 and ResNet-34. | Used now |
| Augmentation | Controlled image variation during training. | Helps test whether models can handle lighting, crop, and viewpoint shifts. | Used now |
| Generalization | Performing well on new images, not just training data. | Important because inspection models must work beyond memorized samples. | Used now |
| Overfitting | Learning the training set too narrowly. | A constant risk in small academic image datasets. | Used now |
| Class imbalance | One class appears more often than another. | The current D1 setup is near-balanced, but recall weighting is still important. | Used now |
| CrossEntropyLoss | Standard classification loss for multi-class logits. | Used in the active ResNet-based classification path. | Used now |
| BCE / BCEWithLogitsLoss | Binary losses for defect vs non-defect prediction. | Used in the historical SimpleCNN baseline path, especially after stability fixes. | Used now |
| Confusion matrix | Counts true/false positives and negatives. | Important because missed defects are higher-risk than extra flags. | Used now |
| Precision / recall / F1 | Precision measures false alarms, recall measures misses, F1 balances them. | Defect recall and defect F1 are the most important current metrics. | Used now |
| AUROC / AUPRC | Metrics that summarize ranking quality across thresholds. | Useful when threshold choice matters for recall-sensitive operation. | Used now |
| Threshold tuning | Choosing a decision cutoff instead of always using 0.5. | Mentioned repeatedly in baseline reports as a next step for recall-critical use. | Partial |
| Misclassification analysis | Reviewing false negatives and false positives. | Central for deciding whether the model is missing subtle cracks or overreacting to benign texture. | Used now |

## Near-Term Research Concepts
| Concept | Simple Explanation | Why It Matters Here | Status |
| --- | --- | --- | --- |
| Limited-data learning | Making useful models from small or uneven datasets. | TyreVisionX is still operating with modest, research-scale data. | Partial |
| Cross-dataset robustness | Checking if a model trained on one dataset still works on another. | D2 and D3 exist in config but are not yet populated in tracked manifests. | Partial |
| Anomaly detection | Finding unusual patterns without relying fully on closed label sets. | Useful if defect classes become too diverse for simple binary supervision. | Planned |
| Defect localization | Identifying where the defect is, not just whether one exists. | Important for inspection trust and operator review. | Planned |
| Segmentation | Predicting a pixel-level defect region. | Needed for precise defect extent rather than image-level classification only. | Planned |

## Longer-Term Concepts
| Concept | Simple Explanation | Why It Matters Here | Status |
| --- | --- | --- | --- |
| GNN / CNN→GNN | A graph neural network reasons over relationships between learned regions or parts. | The repo includes an experimental hybrid path for richer spatial reasoning. | Partial |
| Multi-view learning | Combining several views of the same physical tyre. | Real inspection cells may capture more than one angle. | Planned |
| 3D reconstruction | Estimating 3D structure from multiple views. | A longer-term direction for surface-aware defect reasoning. | Planned |
| Knowledge graph / root-cause reasoning | Structured links between defects, patterns, causes, and inspection evidence. | Mentioned as long-term reasoning support rather than current ML infrastructure. | Planned |

## Practical Takeaway
TyreVisionX already uses standard vision-learning concepts for binary classification and evaluation. The more ambitious concepts in the repo, especially graph reasoning, anomaly detection, localization, segmentation, and 3D reasoning, should still be treated as research directions rather than present capabilities.
