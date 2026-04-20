"""Streamlit QA app for TyreVisionX."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import yaml
from PIL import Image

from src.data.datasets import TyreManifestDataset
from src.data.transforms import get_eval_transforms
from src.models.cnn_gnn import CNNGNNClassifier, HAS_PYG  # type: ignore
from src.models.resnet_classifier import build_resnet
from src.utils.gradcam import generate_gradcam
from src.utils.metrics import classification_metrics
from src.utils.registry import list_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "good", 1: "defect"}
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AUG = BASE_DIR / "configs/aug/light.yaml"


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_checkpoint(checkpoint: Path):
    config_path = checkpoint.parent / "config.yaml"
    config = _load_yaml(config_path) if config_path.exists() else None
    model_cfg = config["model"] if config else {"name": "resnet18", "num_classes": 2, "gnn": {"enabled": False}}
    gnn_cfg = model_cfg.get("gnn", {})
    if gnn_cfg.get("enabled"):
        if not HAS_PYG:
            raise RuntimeError("torch_geometric missing for GNN model")
        model = CNNGNNClassifier(
            model_name=model_cfg.get("name", "resnet18"),
            num_classes=model_cfg.get("num_classes", 2),
            gnn_type=gnn_cfg.get("type", "gat"),
            patch_grid=tuple(gnn_cfg.get("patch_grid", [7, 7])),
            pretrained=False,
        )
    else:
        model = build_resnet(model_name=model_cfg.get("name", "resnet18"), num_classes=model_cfg.get("num_classes", 2), pretrained=False)
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()

    aug_eval = config["data"].get("aug_eval", str(DEFAULT_AUG)) if config else str(DEFAULT_AUG)
    tfms = get_eval_transforms(aug_eval)
    version = config["logging"].get("exp_name", "unknown") if config else "unknown"
    return model, tfms, version


def load_model(selected_option: str, custom_path: str) -> Tuple[torch.nn.Module, any, str]:
    registry = list_models()
    checkpoint = None
    if selected_option in registry:
        latest = registry[selected_option][-1]
        checkpoint = Path(latest["model_dir"]) / "best.pt"
    elif custom_path:
        checkpoint = Path(custom_path)
    else:
        # fall back to any known entry
        for entries in registry.values():
            if entries:
                checkpoint = Path(entries[-1]["model_dir"]) / "best.pt"
                break

    if checkpoint and checkpoint.exists():
        return _load_checkpoint(checkpoint)
    return DummyModel().to(DEVICE), get_eval_transforms(DEFAULT_AUG), "dummy"


def run_inference(model, tfms, image: Image.Image):
    image_rgb = np.array(image.convert("RGB"))
    augmented = tfms(image=image_rgb)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    label_idx = int(torch.argmax(probs).item())
    return label_idx, probs.cpu(), tensor[0].cpu()


def render_prediction(image: Image.Image, label_idx: int, probs: torch.Tensor, gradcam_overlay: np.ndarray | None):
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input", use_column_width=True)
    with col2:
        st.metric("Prediction", LABELS.get(label_idx, str(label_idx)))
        st.write({"prob_good": float(probs[0]), "prob_defect": float(probs[1])})
        if gradcam_overlay is not None:
            st.image(gradcam_overlay, caption="Grad-CAM", use_column_width=True)


def main():
    st.set_page_config(page_title="TyreVisionX", layout="wide")
    st.title("TyreVisionX QA App")

    registry = list_models()
    options = list(registry.keys()) + ["custom"]
    selected = st.sidebar.selectbox("Model", options, index=0 if options else 0)
    custom_path = ""
    if selected == "custom":
        custom_path = st.sidebar.text_input("Checkpoint path", value="")
    model, tfms, version = load_model(selected, custom_path)
    st.sidebar.write(f"Loaded version: {version}")

    gradcam_enabled = st.sidebar.checkbox("Show Grad-CAM", value=True)

    tab1, tab2 = st.tabs(["Single/Batch Upload", "Manifest Evaluation"])

    with tab1:
        uploads = st.file_uploader("Upload tyre images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        results = []
        if uploads:
            for uploaded in uploads:
                image = Image.open(uploaded)
                label_idx, probs, tensor = run_inference(model, tfms, image)
                overlay = None
                if gradcam_enabled:
                    try:
                        overlay = generate_gradcam(tensor, model, class_idx=label_idx)
                    except Exception:
                        overlay = None
                render_prediction(image, label_idx, probs, overlay)
                results.append({"file": uploaded.name, "label": LABELS.get(label_idx, str(label_idx)), "prob_good": float(probs[0]), "prob_defect": float(probs[1])})
        if results:
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", data=csv, file_name="predictions.csv")

    with tab2:
        manifest_path = st.text_input("Manifest CSV for evaluation (with labels)", value="")
        split = st.selectbox("Split", ["train", "val", "test"])
        if manifest_path:
            try:
                ds = TyreManifestDataset(manifest_path, split=split, transforms=tfms)
                loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
                all_logits, all_targets = [], []
                with torch.no_grad():
                    for images, labels, _ in loader:
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE)
                        logits = model(images)
                        all_logits.append(logits.cpu())
                        all_targets.append(labels.cpu())
                logits_tensor = torch.cat(all_logits)
                targets_tensor = torch.cat(all_targets)
                metrics = classification_metrics(logits_tensor, targets_tensor)
                st.write(metrics)
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
