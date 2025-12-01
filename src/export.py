"""Export trained models to TorchScript and ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml

from src.models.cnn_gnn import CNNGNNClassifier, HAS_PYG  # type: ignore
from src.models.resnet_classifier import build_resnet


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: Dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    gnn_cfg = model_cfg.get("gnn", {})
    if gnn_cfg.get("enabled"):
        if not HAS_PYG:
            raise ImportError("torch_geometric not installed; disable gnn.enabled")
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
    return model


def main(checkpoint: str, outdir: str | None = None) -> None:
    ckpt_path = Path(checkpoint)
    cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml next to checkpoint {ckpt_path}")
    config = load_yaml(cfg_path)

    exp_name = config.get("logging", {}).get("exp_name", "export")
    output_dir = Path(outdir) if outdir else Path("artifacts/exports") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = build_model(config, ckpt_path, device)

    size = config.get("data", {}).get("input_size", 384)
    if isinstance(size, (list, tuple)):
        height, width = size
    else:
        height = width = size

    dummy = torch.randn(1, 3, height, width, device=device)

    ts_path = output_dir / "model.ts"
    traced = torch.jit.trace(model, dummy)
    traced.save(ts_path)

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
    )

    print(f"Saved TorchScript to {ts_path}")
    print(f"Saved ONNX to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--outdir", default=None, help="Output directory")
    args = parser.parse_args()
    main(args.checkpoint, outdir=args.outdir)
