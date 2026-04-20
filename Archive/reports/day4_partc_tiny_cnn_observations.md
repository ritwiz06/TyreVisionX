# Day 4 - Part C (TyreVisionX Phase 1 Tiny CNN)

## Objective
- Implement and run the small baseline model for fast experimentation and reduced overfitting risk.
- Keep architecture intentionally simple before scaling model capacity.

## Model Definition
- Input: `224 x 224 x 3`
- Block 1: `Conv(16, 3x3)` -> `ReLU` -> `MaxPool`
- Block 2: `Conv(32, 3x3)` -> `ReLU` -> `MaxPool`
- Head: `Flatten` -> `Linear(1)` -> `Sigmoid` (binary output)

## What This CNN Learns (Realistically)
### Early Layer (Conv16)
- Tire edges and sidewall boundaries
- Local contrast changes between rubber and grooves

### Mid Layer (Conv32)
- Groove geometry and repeated tread patterns
- Local texture regularity across the tire surface

### Deeper Decision Stage (Flatten + Linear)
- Pattern disruptions, small discontinuities, and crack-like breaks
- Combines learned features into a final defect probability

## Important Concept: Feature Maps
- Each convolution layer produces multiple output channels.
- Each channel is a different learned detector response.
- With `Conv(16, ...)`, output has `16` feature maps.
- Think of them as `16` different learned views of the same input image.

## Run Configuration
- Model type: `simple_cnn`
- Epochs: `1`
- Batch size: `16`
- Output dir: `artifacts/day3_baseline/simple_cnn_phase1_sigmoid_v1`

## Metrics (Test Split, manual eval)
- Accuracy: `0.6863`
- Precision (macro): `0.7995`
- Recall (macro): `0.6802`
- F1 (macro): `0.6491`
- Precision (defect): `0.6202`
- Recall (defect): `0.9923`
- F1 (defect): `0.7633`
- AUROC: `0.8824`
- AUPRC: `0.8691`

## Confusion Matrix Notes
- Matrix: `[[46, 79], [1, 129]]`
- False negatives: `1`
- False positives: `79`

## Observation
- The tiny model quickly achieves very high defect recall with minimal complexity.
- Tradeoff is high false positives, which is expected for this early recall-first baseline.

## Next Day-4 Actions
1. Add decision-threshold tuning to reduce false positives while maintaining recall.
2. Run 3-5 epochs and compare stability against 1-epoch behavior.
3. Evaluate stronger augmentation to reduce sensitivity to lighting/background.
