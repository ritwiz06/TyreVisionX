# Detection Phase Stub (YOLOv8)

- Goal: localize defects on tyre surfaces using bounding boxes.
- Data: annotate defect regions per image; consider synthetic overlays for minority classes.
- Model: YOLOv8n/s with transfer learning; anchor-free head.
- Steps:
  1. Convert manifests to YOLO txt format (one txt per image).
  2. Create `data/yolo.yaml` describing train/val/test paths and class names [good, defect].
  3. Use `ultralytics` package for training and export to ONNX/TensorRT.
  4. Integrate predictions into FastAPI/Streamlit as additional endpoint/tab with Grad-CAM-style visualization.
