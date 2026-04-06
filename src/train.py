"""Training script for TyreVisionX."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.balance import compute_class_weights, focal_loss
from src.data.datasets import load_combined_datasets
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.dataset import TyreManifestDataset
from src.models.cnn_gnn import CNNGNNClassifier, HAS_PYG  # type: ignore
from src.models.resnet_classifier import build_resnet
from src.utils.logging import configure_logging, get_logger
from src.utils.metrics import classification_metrics
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


logger = get_logger("train")


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_datasets(config: Dict) -> Tuple:
    train_tfms = get_train_transforms(config["data"]["aug_train"])
    eval_tfms = get_eval_transforms(config["data"]["aug_eval"])

    manifest_csv = config["data"].get("manifest_csv")
    if manifest_csv:
        train_ds = TyreManifestDataset(manifest_csv=manifest_csv, split="train", transforms=train_tfms)
        val_ds = TyreManifestDataset(manifest_csv=manifest_csv, split="val", transforms=eval_tfms)
        return train_ds, val_ds, eval_tfms, [manifest_csv]

    data_cfg = load_yaml(Path(config["data"]["config_file"]))
    selected = config["data"].get("use_datasets", data_cfg.get("use_datasets", []))
    manifests = []
    roots = {}
    for ds in selected:
        ds_cfg = data_cfg["paths"].get(ds)
        if not ds_cfg:
            raise ValueError(f"Dataset {ds} not found in data config")
        manifests.append(ds_cfg["manifest"])
        roots[ds] = Path(ds_cfg["root"])

    train_ds = load_combined_datasets(manifests, split="train", transforms=train_tfms, roots=roots)
    val_ds = load_combined_datasets(manifests, split="val", transforms=eval_tfms, roots=roots)
    return train_ds, val_ds, eval_tfms, manifests


def compute_weights_from_manifests(manifests) -> torch.Tensor:
    from src.data.balance import compute_class_weights
    import pandas as pd

    labels = []
    for manifest in manifests:
        df = pd.read_csv(manifest)
        if "split" in df.columns:
            df = df[df["split"] == "train"]
        labels.extend(df["label"].tolist())
    return compute_class_weights(labels)


def train_epoch(model, loader, device, criterion, optimizer, scaler, config) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    grad_clip = config["training"].get("grad_clip", None)
    use_amp = bool(config["training"].get("mixed_precision", False))

    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits_tensor = torch.cat(all_logits)
    targets_tensor = torch.cat(all_targets)
    metrics = classification_metrics(logits_tensor, targets_tensor)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def eval_epoch(model, loader, device, criterion) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    logits_tensor = torch.cat(all_logits)
    targets_tensor = torch.cat(all_targets)
    metrics = classification_metrics(logits_tensor, targets_tensor)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def build_model(config: Dict) -> nn.Module:
    model_cfg = config["model"]
    gnn_cfg = model_cfg.get("gnn", {})
    if gnn_cfg.get("enabled"):
        if not HAS_PYG:
            raise ImportError("torch_geometric not available; disable gnn.enabled or install it")
        model = CNNGNNClassifier(
            model_name=model_cfg.get("name", "resnet18"),
            num_classes=model_cfg.get("num_classes", 2),
            gnn_type=gnn_cfg.get("type", "gat"),
            patch_grid=tuple(gnn_cfg.get("patch_grid", [7, 7])),
            pretrained=model_cfg.get("pretrained", True),
        )
    else:
        model = build_resnet(
            model_name=model_cfg.get("name", "resnet18"),
            num_classes=model_cfg.get("num_classes", 2),
            pretrained=model_cfg.get("pretrained", True),
        )
    return model


def save_artifacts(model: nn.Module, save_dir: Path, config_path: Path, metrics: Dict) -> None:
    ensure_dir(save_dir)
    torch.save(model.state_dict(), save_dir / "best.pt")
    shutil.copy(config_path, save_dir / "config.yaml")
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(save_dir / "class_index.json", "w", encoding="utf-8") as f:
        json.dump({"good": 0, "defect": 1}, f, indent=2)


def main(config_path: str) -> None:
    cfg_path = Path(config_path)
    config = load_yaml(cfg_path)
    set_seed(42)
    log_dir = Path(config["logging"].get("save_dir", "artifacts/experiments/default"))
    configure_logging(log_dir)
    logger.info(f"Loading config from {cfg_path}")

    train_ds, val_ds, eval_tfms, manifests = build_datasets(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"].get("batch_size", 16),
        shuffle=config["data"].get("shuffle", True),
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"].get("batch_size", 16),
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    class_weights = compute_weights_from_manifests(manifests)
    class_weights = class_weights.to(device)

    model = build_model(config).to(device)
    lr = config["training"].get("lr", 3e-4)
    weight_decay = config["training"].get("weight_decay", 1e-4)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"].get("epochs", 20))

    use_focal = bool(config["training"].get("focal_loss", False))
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    if use_focal:
        criterion = lambda logits, labels: focal_loss(logits, labels)
    else:
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        def criterion(logits, labels):
            return ce_loss(logits, labels)
    scaler = GradScaler(enabled=bool(config["training"].get("mixed_precision", False)))

    best_metric = -1.0
    best_state = None
    patience = config["training"].get("early_stop_patience", 5)
    patience_counter = 0
    history = {"train": [], "val": []}

    epochs = config["training"].get("epochs", 20)
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        train_metrics = train_epoch(model, train_loader, device, criterion, optimizer, scaler, config)
        val_metrics = eval_epoch(model, val_loader, device, criterion)
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        val_primary = val_metrics.get(config["metrics"].get("primary", "f1_defect"), 0.0)
        logger.info(f"Train loss {train_metrics['loss']:.4f} | Val {config['metrics'].get('primary','f1_defect')} {val_primary:.4f}")

        if val_primary > best_metric:
            best_metric = val_primary
            best_state = model.state_dict()
            patience_counter = 0
            logger.info("New best model found")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics_summary = {"best_val_metric": best_metric, "history": history}
    save_artifacts(model, Path(config["logging"]["save_dir"]), cfg_path, metrics_summary)

    try:
        register_model(config["logging"]["exp_name"], Path(config["logging"]["save_dir"]), metadata={"metric": best_metric})
    except Exception as exc:  # pragma: no cover - registry optional
        logger.warning(f"Registry update failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_resnet18.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
