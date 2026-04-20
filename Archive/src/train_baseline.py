"""Legacy baseline training script kept for historical reproducibility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.balance import compute_class_weights
from src.dataset import TyreManifestDataset
from src.models.feature_extractor import FrozenFeatureExtractorClassifier
from src.models.simple_cnn import SimpleCNN
from src.transforms import get_eval_transforms, get_train_transforms
from src.utils.logging import configure_logging, get_logger
from src.utils.metrics import classification_metrics
from src.utils.seed import set_seed

logger = get_logger("train_baseline")


def train_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    binary_mode: bool = False,
    binary_logits: bool = False,
    class_weights: torch.Tensor | None = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if binary_mode:
            scores = outputs.reshape(-1)
            labels_float = labels.float()
            if class_weights is not None:
                sample_weights = torch.where(labels == 1, class_weights[1], class_weights[0]).float()
                if binary_logits:
                    loss = F.binary_cross_entropy_with_logits(scores, labels_float, weight=sample_weights)
                else:
                    loss = F.binary_cross_entropy(scores, labels_float, weight=sample_weights)
            else:
                if binary_logits:
                    loss = F.binary_cross_entropy_with_logits(scores, labels_float)
                else:
                    loss = criterion(scores, labels_float)
            metric_outputs = torch.sigmoid(scores) if binary_logits else scores
        else:
            loss = criterion(outputs, labels)
            metric_outputs = outputs

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_outputs.append(metric_outputs.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits_tensor = torch.cat(all_outputs)
    targets_tensor = torch.cat(all_targets)
    metrics = classification_metrics(logits_tensor, targets_tensor)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def eval_epoch(
    model,
    loader,
    device,
    criterion,
    binary_mode: bool = False,
    binary_logits: bool = False,
    class_weights: torch.Tensor | None = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if binary_mode:
                scores = outputs.reshape(-1)
                labels_float = labels.float()
                if class_weights is not None:
                    sample_weights = torch.where(labels == 1, class_weights[1], class_weights[0]).float()
                    if binary_logits:
                        loss = F.binary_cross_entropy_with_logits(scores, labels_float, weight=sample_weights)
                    else:
                        loss = F.binary_cross_entropy(scores, labels_float, weight=sample_weights)
                else:
                    if binary_logits:
                        loss = F.binary_cross_entropy_with_logits(scores, labels_float)
                    else:
                        loss = criterion(scores, labels_float)
                metric_outputs = torch.sigmoid(scores) if binary_logits else scores
            else:
                loss = criterion(outputs, labels)
                metric_outputs = outputs

            total_loss += loss.item() * labels.size(0)
            all_outputs.append(metric_outputs.cpu())
            all_targets.append(labels.cpu())

    logits_tensor = torch.cat(all_outputs)
    targets_tensor = torch.cat(all_targets)
    metrics = classification_metrics(logits_tensor, targets_tensor)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def build_model(
    model_type: str,
    pretrained: bool,
    unfreeze_last_block: bool,
    use_batchnorm: bool = False,
    dropout: float = 0.0,
    output_logits: bool = False,
) -> nn.Module:
    if model_type == "simple_cnn":
        return SimpleCNN(
            in_channels=3,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            output_logits=output_logits,
        )
    if model_type == "frozen_resnet18":
        return FrozenFeatureExtractorClassifier(
            backbone_name="resnet18",
            num_classes=2,
            pretrained=pretrained,
            freeze_backbone=True,
            unfreeze_last_block=unfreeze_last_block,
        )
    if model_type == "frozen_resnet34":
        return FrozenFeatureExtractorClassifier(
            backbone_name="resnet34",
            num_classes=2,
            pretrained=pretrained,
            freeze_backbone=True,
            unfreeze_last_block=unfreeze_last_block,
        )
    if model_type == "frozen_resnet50":
        return FrozenFeatureExtractorClassifier(
            backbone_name="resnet50",
            num_classes=2,
            pretrained=pretrained,
            freeze_backbone=True,
            unfreeze_last_block=unfreeze_last_block,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline classifier")
    parser.add_argument("--manifest", default="data/manifests/D1_tyrenet_manifest.csv", help="Manifest CSV path")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument(
        "--preset",
        default="day5",
        choices=["none", "light", "day5", "strong"],
        help="Training augmentation preset",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop if monitored metric does not improve for N epochs (0 disables).",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement required to reset early-stopping counter.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", default="artifacts/day3_baseline/simple_cnn_v1", help="Output dir")
    parser.add_argument(
        "--model_type",
        default="simple_cnn",
        choices=["simple_cnn", "frozen_resnet18", "frozen_resnet34", "frozen_resnet50"],
        help="Baseline architecture",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained backbone weights when applicable",
    )
    parser.add_argument(
        "--unfreeze_last_block",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Unfreeze ResNet layer4 while keeping earlier backbone layers frozen",
    )
    parser.add_argument(
        "--use_batchnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable BatchNorm in SimpleCNN blocks",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability before final linear layer in SimpleCNN",
    )
    parser.add_argument(
        "--output_logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit logits for SimpleCNN (recommended with BCEWithLogitsLoss)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(f"--dropout must be in [0, 1). Got: {args.dropout}")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    configure_logging(out_dir)
    logger.info(f"Manifest: {args.manifest}")

    train_tfms = get_train_transforms(img_size=args.img_size, preset=args.preset)
    eval_tfms = get_eval_transforms(img_size=args.img_size)

    train_ds = TyreManifestDataset(manifest_csv=args.manifest, split="train", transforms=train_tfms)
    val_ds = TyreManifestDataset(manifest_csv=args.manifest, split="val", transforms=eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model_type,
        pretrained=args.pretrained,
        unfreeze_last_block=args.unfreeze_last_block,
        use_batchnorm=args.use_batchnorm,
        dropout=args.dropout,
        output_logits=args.output_logits,
    ).to(device)

    binary_mode = args.model_type == "simple_cnn"
    binary_logits = binary_mode and args.output_logits
    class_weights = compute_class_weights(train_ds.df["label"].tolist(), num_classes=2).to(device)
    if binary_mode:
        criterion = nn.BCEWithLogitsLoss() if binary_logits else nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_state = None
    best_epoch = 0
    epochs_ran = 0
    no_improve_epochs = 0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        epochs_ran = epoch
        logger.info(f"Epoch {epoch}/{args.epochs}")
        train_metrics = train_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            binary_mode=binary_mode,
            binary_logits=binary_logits,
            class_weights=class_weights if binary_mode else None,
        )
        val_metrics = eval_epoch(
            model,
            val_loader,
            device,
            criterion,
            binary_mode=binary_mode,
            binary_logits=binary_logits,
            class_weights=class_weights if binary_mode else None,
        )
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        val_f1 = val_metrics.get("f1_defect", 0.0)
        val_recall_defect = val_metrics.get("recall_defect", 0.0)
        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        gap = train_loss - val_loss
        logger.info(
            "Train loss %.4f | Val loss %.4f | Val recall_defect %.4f | Val f1_defect %.4f | Gap(train-val) %.4f"
            % (train_loss, val_loss, val_recall_defect, val_f1, gap)
        )

        if val_f1 > (best_metric + args.early_stopping_min_delta):
            best_metric = val_f1
            best_state = model.state_dict()
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d (best epoch %d, best val_f1_defect %.4f)"
                % (epoch, best_epoch, best_metric)
            )
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "last.pt")
    if best_state:
        torch.save(best_state, out_dir / "best.pt")

    summary = {
        "best_val_f1_defect": best_metric,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "history": history,
        "args": vars(args),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": args.model_type,
                "pretrained": args.pretrained,
                "unfreeze_last_block": args.unfreeze_last_block,
                "use_batchnorm": args.use_batchnorm,
                "dropout": args.dropout,
                "output_logits": args.output_logits,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
