"""Batch inference demo for TyreVisionX."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.transforms import get_eval_transforms
from src.models.cnn_gnn import CNNGNNClassifier, HAS_PYG  # type: ignore
from src.models.resnet_classifier import build_resnet
from src.utils.gradcam import generate_gradcam

LABELS = {0: "good", 1: "defect"}


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(checkpoint: Path, device: torch.device):
    cfg_path = checkpoint.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml next to checkpoint {checkpoint}")
    config = load_yaml(cfg_path)
    model_cfg = config["model"]
    gnn_cfg = model_cfg.get("gnn", {})
    if gnn_cfg.get("enabled"):
        if not HAS_PYG:
            raise ImportError("torch_geometric missing; disable gnn.enabled")
        model = CNNGNNClassifier(
            model_name=model_cfg.get("name", "resnet18"),
            num_classes=model_cfg.get("num_classes", 2),
            gnn_type=gnn_cfg.get("type", "gat"),
            patch_grid=tuple(gnn_cfg.get("patch_grid", [7, 7])),
            pretrained=False,
        )
    else:
        model = build_resnet(
            model_name=model_cfg.get("name", "resnet18"),
            num_classes=model_cfg.get("num_classes", 2),
            pretrained=False,
        )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tfms = get_eval_transforms(config["data"].get("aug_eval", "configs/aug_light.yaml"))
    return model, tfms


def run_inference(model, tfms, image_path: Path, device: torch.device):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug = tfms(image=img)
    tensor = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    label_idx = int(torch.argmax(probs).item())
    return label_idx, probs.cpu(), tensor[0].cpu()


def main(checkpoint: str, input_dir: str, output_csv: str, heatmap_dir: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfms = build_model(Path(checkpoint), device)
    images = [p for p in Path(input_dir).glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    results: List[Dict] = []
    heatmap_dir_path = Path(heatmap_dir) if heatmap_dir else None
    if heatmap_dir_path:
        heatmap_dir_path.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        label_idx, probs, tensor = run_inference(model, tfms, img_path, device)
        row = {
            "image": str(img_path),
            "label": LABELS.get(label_idx, str(label_idx)),
            "prob_good": float(probs[0]),
            "prob_defect": float(probs[1]),
        }
        results.append(row)

        if heatmap_dir_path:
            try:
                overlay = generate_gradcam(tensor, model, class_idx=label_idx)
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(heatmap_dir_path / f"{img_path.stem}_gradcam.png"), overlay_bgr)
            except Exception:
                pass

    df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} results to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Checkpoint path")
    parser.add_argument("--input_dir", required=True, help="Input directory of images")
    parser.add_argument("--output_csv", default="artifacts/demo_results.csv", help="Output CSV path")
    parser.add_argument("--heatmap_dir", default=None, help="Directory to save Grad-CAM overlays")
    args = parser.parse_args()
    main(args.model, args.input_dir, args.output_csv, args.heatmap_dir)
