"""Evaluation script for TyreVisionX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import load_dataset_from_runtime_config
from src.data.transforms import get_eval_transforms
from src.models.cnn_gnn import CNNGNNClassifier, HAS_PYG  # type: ignore
from src.models.resnet_classifier import build_resnet
from src.utils.confusion import compute_confusion, plot_confusion
from src.utils.logging import get_logger
from src.utils.metrics import classification_metrics

logger = get_logger("eval")


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


def plot_curves(targets, probs, save_dir: Path) -> Dict[str, str]:
    save_dir.mkdir(parents=True, exist_ok=True)
    curves = {}

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = roc_auc_score(targets, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    roc_path = save_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    curves["roc_curve"] = str(roc_path)

    precision, recall, _ = precision_recall_curve(targets, probs)
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    pr_path = save_dir / "pr_curve.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()
    curves["pr_curve"] = str(pr_path)
    return curves


def evaluate(model, loader, device) -> Dict:
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    logits_tensor = torch.cat(all_logits)
    targets_tensor = torch.cat(all_targets)
    metrics = classification_metrics(logits_tensor, targets_tensor)
    probs = torch.softmax(logits_tensor, dim=1)[:, 1].numpy()
    preds = torch.argmax(logits_tensor, dim=1).numpy()
    metrics["confusion"] = compute_confusion(targets_tensor.numpy(), preds).tolist()
    return metrics, probs, targets_tensor.numpy(), preds


def main(checkpoint: str, split: str = "test", report_path: str | None = None) -> None:
    ckpt_path = Path(checkpoint)
    cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml next to checkpoint {ckpt_path}")
    config = load_yaml(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_tfms = get_eval_transforms(config["data"].get("aug_eval", "configs/aug/light.yaml"))
    dataset, manifests = load_dataset_from_runtime_config(config["data"], split=split, transforms=eval_tfms)
    loader = DataLoader(dataset, batch_size=config["data"].get("batch_size", 16), shuffle=False, num_workers=config["data"].get("num_workers", 4))

    model = build_model(config, ckpt_path, device)
    metrics, probs, targets, preds = evaluate(model, loader, device)

    exp_name = config["logging"].get("exp_name", "experiment")
    reports_dir = Path(report_path).parent if report_path else Path("artifacts/reports") / exp_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        report_path = reports_dir / "report.json"
    else:
        report_path = Path(report_path)

    curves = plot_curves(targets, probs, reports_dir)
    cm_array = np.array(metrics["confusion"])
    cm_path = plot_confusion(cm_array, labels=["good", "defect"], save_path=reports_dir / "confusion_matrix.png")

    report = {
        "checkpoint": str(ckpt_path),
        "split": split,
        "metrics": {k: float(v) if not isinstance(v, list) else v for k, v in metrics.items() if k != "confusion"},
        "confusion": metrics["confusion"],
        "plots": {**curves, "confusion": str(cm_path)},
        "manifests": manifests,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved report to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--report", default=None, help="Output report path")
    args = parser.parse_args()
    main(args.checkpoint, split=args.split, report_path=args.report)
