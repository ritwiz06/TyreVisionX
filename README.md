# TyreVisionX

AI-powered tyre defect detection baseline for factory inspection cells. Phase 1 delivers a binary classifier (good = 0, defect = 1) trained on public datasets, with hooks for future CNN→GNN fusion and detection/segmentation tracks.

## Repository Layout
```
TyreVisionX/
  configs/           # YAML configs for data, aug, training
  data/              # Place datasets here; manifests live in data/manifests
  src/               # Library + training, eval, export, services, apps
  scripts/           # Dataset prep, manifests, demo inference
  tests/             # Pytest-based sanity checks
  artifacts/         # Outputs, checkpoints, reports, registry
  .github/workflows/ # CI
```

## Goals
- **Phase 1 (this repo)**: Binary classification of tyre defects using TyreNet, Kaggle tire crack, and Kaggle tyre quality datasets. Metrics focus on F1/recall for the defect class, AUROC, and cross-dataset robustness (e.g., train on D1, test on D2/D3).
- **Future phases**: CNN→GNN hybrid for spatial reasoning, 3D/multi-modal fusion, detection (YOLOv8), and segmentation (Mask R-CNN/U-Net). Stubs and config hooks are provided.

## KPIs (Phase 1)
- F1 on defect ≥ 0.90
- Recall on defect ≥ 0.95 (false negatives are critical)
- AUROC ≥ 0.95 on main dataset
- Report cross-dataset metrics (train on D1, evaluate on D2/D3, etc.)

## Quickstart
1. **Setup env**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   make setup
   ```
2. **Datasets**
   - D1 TyreNet (public; may require manual download)
   - D2 Kaggle tire crack: https://www.kaggle.com/datasets/prajwalkatke/tire-crack-detection
   - D3 Kaggle tyre quality: https://www.kaggle.com/datasets/ritzg42/tyre-quality-image-dataset
   Place extracted folders under `data/` or set `DATA_ROOT` in `.env`.
3. **Prep manifests & splits**
   ```bash
   make manifests
   make folds
   ```
4. **Train** (ResNet-18 example)
   ```bash
   make train
   ```
5. **Evaluate**
   ```bash
   make eval
   ```
6. **Export** (TorchScript + ONNX)
   ```bash
   make export
   ```
7. **Serve FastAPI**
   ```bash
   make serve
   # POST /classify with an image file
   ```
8. **Streamlit app**
   ```bash
   make app
   ```

## Configuration
- `configs/data.yaml` defines dataset roots/manifests, label mapping (`good: 0`, `defect: 1`), and split policy (ratios or precomputed splits).
- `configs/aug_*.yaml` hold albumentations pipelines (light vs strong). Validation/test use deterministic resize + normalize.
- `configs/train_resnet18.yaml` and `configs/train_resnet34.yaml` configure model, data, training, metrics, and logging paths. Enable GNN via `model.gnn.enabled: true` (requires torch-geometric).

## Datasets & Manifests
- Use `scripts/download_datasets.py` for download guidance (Kaggle CLI notes; TyreNet may need manual link).
- Use `scripts/prepare_manifests.py` to scan dataset folders into CSV manifests under `data/manifests/`.
- Use `scripts/prepare_folds.py` for stratified train/val/test columns (70/15/15 by default).

## Training & Evaluation
- See `src/train.py` (AdamW + cosine scheduler, early stopping, class weights from training split only, ImageNet normalization everywhere).
- Metrics emphasize defect-positive class: accuracy, precision, recall, f1_macro, `f1_defect`, AUROC, AUPRC.
- Reports, confusion matrix, ROC/PR curves saved under `artifacts/reports/<exp_name>/`.

## Models
- ResNet-18/34 classifiers with configurable pretrained weights.
- Optional CNN→GNN head (`src/models/cnn_gnn.py`) using patchified ResNet features and GAT/SAGE layers (if torch-geometric installed). Falls back gracefully if missing.
- Future stubs for YOLOv8 detection and Mask R-CNN/U-Net segmentation.

## Inference
- FastAPI service in `src/service_fastapi.py` exposes `/classify` (multipart or base64). Grad-CAM optional via `?gradcam=true`. Uses registry to load latest model by default.
- Streamlit QA app `src/app_streamlit.py` for interactive uploads, Grad-CAM overlays, and batch folder evaluation.
- CLI demo `scripts/demo_infer.py` for batch classification + CSV/heatmap outputs.

## Tooling
- Python 3.10+
- PyTorch, torchvision, albumentations, scikit-learn, torchmetrics
- FastAPI, uvicorn, pydantic
- Streamlit
- Optional: torch-geometric (install separately; not a hard dependency)
- Formatting/linting: black + ruff (configured in `pyproject.toml`).

## Registry & Artifacts
- Registry stored under `artifacts/registry/` (JSON metadata). `best.pt`, configs, transforms, and metrics saved per experiment under `artifacts/experiments/<exp_name>/`.

## Progress Tracking
- See `docs/progress_log.csv` (Excel-friendly) for a running log of changes, methods, and checkpoints. Update as you iterate.

## Notes
- Label mapping is fixed: `good = 0`, `defect = 1`.
- Validation/test use deterministic resize + normalize (no random aug).
- Class weights derived from **training split only**.
- When enabling GNN mode, ensure torch-geometric and its dependencies are installed for your CUDA stack.

## License
MIT License. See `LICENSE`.
