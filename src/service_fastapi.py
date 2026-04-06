"""FastAPI inference service for TyreVisionX."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.data.transforms import get_eval_transforms
from src.models.cnn_gnn import CNNGNNClassifier, HAS_PYG  # type: ignore
from src.models.resnet_classifier import build_resnet
from src.utils.gradcam import generate_gradcam
from src.utils.registry import get_latest_model

app = FastAPI(title="TyreVisionX API", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AUG = BASE_DIR / "configs/aug/light.yaml"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "good", 1: "defect"}


class ImagePayload(BaseModel):
    image_base64: Optional[str] = None


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


MODEL = DummyModel().to(DEVICE)
EVAL_TFMS = get_eval_transforms(DEFAULT_AUG)
MODEL_VERSION = "dummy"


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_checkpoint(checkpoint: Path):
    config_path = checkpoint.parent / "config.yaml"
    config = _load_yaml(config_path) if config_path.exists() else None

    model_cfg = config["model"] if config else {"name": "resnet18", "num_classes": 2, "pretrained": False, "gnn": {"enabled": False}}
    gnn_cfg = model_cfg.get("gnn", {})
    if gnn_cfg.get("enabled"):
        if not HAS_PYG:
            raise HTTPException(status_code=500, detail="torch_geometric missing for GNN model")
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


def _bootstrap_model():
    global MODEL, EVAL_TFMS, MODEL_VERSION
    latest = get_latest_model("resnet18_tyrenet_v1") or get_latest_model("resnet34_tyrenet_v1")
    checkpoint_env = BASE_DIR / "artifacts/experiments"
    checkpoint_path = None
    if latest:
        checkpoint_path = Path(latest) / "best.pt"
    else:
        default_ckpt = checkpoint_env / "resnet18_tyrenet_v1" / "best.pt"
        if default_ckpt.exists():
            checkpoint_path = default_ckpt

    if checkpoint_path and checkpoint_path.exists():
        try:
            MODEL, EVAL_TFMS, MODEL_VERSION = _load_checkpoint(checkpoint_path)
            return
        except Exception:
            pass
    MODEL = DummyModel().to(DEVICE)
    MODEL_VERSION = "dummy"
    EVAL_TFMS = get_eval_transforms(DEFAULT_AUG)


def read_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


@app.on_event("startup")
def startup_event():
    _bootstrap_model()


@app.post("/classify")
async def classify_image(file: UploadFile | None = File(None), payload: ImagePayload | None = Body(None), gradcam: bool = False):
    if file:
        data = await file.read()
    elif payload and payload.image_base64:
        data = base64.b64decode(payload.image_base64)
    else:
        raise HTTPException(status_code=400, detail="Provide an image file or base64 payload")

    try:
        image = read_image(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    augmented = EVAL_TFMS(image=image)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    label_idx = int(torch.argmax(probs).item())
    response = {
        "label": LABELS.get(label_idx, str(label_idx)),
        "prob_defect": float(probs[1].item()),
        "prob_good": float(probs[0].item()),
        "confidence": float(probs.max().item()),
        "model_version": MODEL_VERSION,
    }

    if gradcam:
        try:
            overlay = generate_gradcam(tensor[0].cpu(), MODEL, class_idx=label_idx)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".png", overlay_bgr)
            response["gradcam"] = base64.b64encode(buffer).decode("utf-8")
        except Exception:
            response["gradcam"] = None

    return response


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_version": MODEL_VERSION}
