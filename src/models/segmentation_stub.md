# Segmentation Phase Stub (Mask R-CNN / U-Net)

- Goal: pixel-level defect localization (cracks, bulges, surface anomalies).
- Data: polygon masks or binary masks per defect; consider weak labels + pseudo labels from detection stage.
- Model options:
  - Mask R-CNN (torchvision/Detectron2) for instance-level segmentation.
  - U-Net / DeepLabV3+ for semantic segmentation of defect regions.
- Pipeline sketch:
  1. Convert manifests to COCO-style JSON with segmentation polygons.
  2. Build albumentations segmentation transforms (RandomResizedCrop, color jitter, elastic).
  3. Train with Dice/Focal/BCE losses; emphasize recall.
  4. Export ONNX/TorchScript; add overlay visualization in Streamlit.
