"""Baseline evaluation script for CNN baselines and frozen feature extractors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TyreManifestDataset
from src.models.feature_extractor import FrozenFeatureExtractorClassifier
from src.models.simple_cnn import SimpleCNN
from src.transforms import get_eval_transforms
from src.utils.confusion import compute_confusion, plot_confusion
from src.utils.metrics import classification_metrics


def _meta_value(meta_batch: Any, key: str, index: int) -> str:
    if isinstance(meta_batch, dict):
        value = meta_batch.get(key, "")
        if isinstance(value, (list, tuple)):
            return str(value[index])
        return str(value)
    return ""


def _resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def evaluate(model, loader, device) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    all_outputs = []
    all_targets = []
    all_preds = []
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for images, labels, meta in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1):
                pos_scores = outputs.reshape(-1).float().cpu()
                if torch.any((pos_scores < 0.0) | (pos_scores > 1.0)):
                    pos_scores = torch.sigmoid(pos_scores)
                probs = torch.stack([1.0 - pos_scores, pos_scores], dim=1)
                preds = (pos_scores >= 0.5).long()
                metric_outputs = pos_scores
            else:
                probs = torch.softmax(outputs, dim=1).cpu()
                preds = torch.argmax(probs, dim=1)
                metric_outputs = outputs.cpu()

            all_outputs.append(metric_outputs)
            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())

            for i in range(labels.size(0)):
                target_i = int(labels[i].item())
                pred_i = int(preds[i].item())
                record = {
                    "image_path": _meta_value(meta, "image_path", i),
                    "dataset_id": _meta_value(meta, "dataset_id", i),
                    "split": _meta_value(meta, "split", i),
                    "target": target_i,
                    "pred": pred_i,
                    "target_label": "defect" if target_i == 1 else "good",
                    "pred_label": "defect" if pred_i == 1 else "good",
                    "prob_good": float(probs[i, 0].item()),
                    "prob_defect": float(probs[i, 1].item()),
                }
                record["is_misclassified"] = int(target_i != pred_i)
                record["is_false_negative"] = int(target_i == 1 and pred_i == 0)
                records.append(record)

    logits_tensor = torch.cat(all_outputs)
    targets_tensor = torch.cat(all_targets)
    preds_tensor = torch.cat(all_preds)
    metrics = classification_metrics(logits_tensor, targets_tensor)

    cm = compute_confusion(targets_tensor.numpy(), preds_tensor.numpy())
    metrics["confusion"] = cm.tolist()
    return metrics, records


def _plot_record_grid(records: List[Dict[str, Any]], save_path: Path, title: str, max_images: int) -> Path | None:
    selected = records[: max(0, max_images)]
    if not selected:
        return None

    n = len(selected)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i, rec in enumerate(selected):
        ax = axes[i]
        img_path = _resolve_image_path(rec["image_path"])
        image = cv2.imread(str(img_path))
        if image is None:
            ax.text(0.5, 0.5, f"Missing\n{img_path.name}", ha="center", va="center")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(
            f"T:{rec['target_label']} P:{rec['pred_label']}\n"
            f"p_def:{rec['prob_defect']:.2f} {img_path.name}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def export_misclassification_artifacts(
    records: List[Dict[str, Any]],
    out_dir: Path,
    max_visuals: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    records_df = pd.DataFrame(records)
    mis_df = records_df[records_df["is_misclassified"] == 1].copy()
    fn_df = records_df[records_df["is_false_negative"] == 1].copy()
    fp_df = records_df[(records_df["target"] == 0) & (records_df["pred"] == 1)].copy()

    mis_csv = out_dir / "misclassified.csv"
    fn_csv = out_dir / "false_negatives.csv"
    fp_csv = out_dir / "false_positives.csv"
    mis_df.to_csv(mis_csv, index=False)
    fn_df.to_csv(fn_csv, index=False)
    fp_df.to_csv(fp_csv, index=False)

    mis_grid = _plot_record_grid(
        mis_df.to_dict(orient="records"),
        out_dir / "misclassified_grid.png",
        "Misclassified Samples",
        max_visuals,
    )
    fn_grid = _plot_record_grid(
        fn_df.to_dict(orient="records"),
        out_dir / "false_negative_grid.png",
        "False Negative Samples (Critical)",
        max_visuals,
    )
    fp_grid = _plot_record_grid(
        fp_df.to_dict(orient="records"),
        out_dir / "false_positive_grid.png",
        "False Positive Samples",
        max_visuals,
    )

    out["num_misclassified"] = int(len(mis_df))
    out["num_false_negatives"] = int(len(fn_df))
    out["num_false_positives"] = int(len(fp_df))
    out["misclassified_csv"] = str(mis_csv)
    out["false_negatives_csv"] = str(fn_csv)
    out["false_positives_csv"] = str(fp_csv)
    out["misclassified_grid"] = str(mis_grid) if mis_grid else None
    out["false_negative_grid"] = str(fn_grid) if fn_grid else None
    out["false_positive_grid"] = str(fp_grid) if fp_grid else None
    return out


def build_model(
    model_type: str,
    pretrained: bool,
    unfreeze_last_block: bool,
    use_batchnorm: bool = False,
    dropout: float = 0.0,
    output_logits: bool = False,
) -> torch.nn.Module:
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


def infer_model_type(checkpoint_path: Path) -> tuple[str, bool, bool, bool, float, bool]:
    model_info_path = checkpoint_path.parent / "model_info.json"
    if model_info_path.exists():
        info = json.loads(model_info_path.read_text(encoding="utf-8"))
        return (
            info.get("model_type", "simple_cnn"),
            bool(info.get("pretrained", False)),
            bool(info.get("unfreeze_last_block", False)),
            bool(info.get("use_batchnorm", False)),
            float(info.get("dropout", 0.0)),
            bool(info.get("output_logits", False)),
        )
    return "simple_cnn", False, False, False, 0.0, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline classifier")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", default="data/processed/D1_manifest.csv", help="Manifest CSV path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--out_dir", default=None, help="Output dir for report")
    parser.add_argument(
        "--model_type",
        default=None,
        choices=["simple_cnn", "frozen_resnet18", "frozen_resnet34", "frozen_resnet50"],
        help="Model type; if omitted, inferred from model_info.json",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Backbone pretrained flag; if omitted, inferred from model_info.json",
    )
    parser.add_argument(
        "--unfreeze_last_block",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether layer4 was unfrozen during training; if omitted, inferred from model_info.json",
    )
    parser.add_argument(
        "--use_batchnorm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="SimpleCNN BatchNorm flag; if omitted, inferred from model_info.json",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="SimpleCNN dropout probability; if omitted, inferred from model_info.json",
    )
    parser.add_argument(
        "--output_logits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="SimpleCNN output logits flag; if omitted, inferred from model_info.json",
    )
    parser.add_argument(
        "--save_misclassified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save misclassification CSV and image grids",
    )
    parser.add_argument("--max_visuals", type=int, default=20, help="Max images for each misclassification grid")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    (
        inferred_model_type,
        inferred_pretrained,
        inferred_unfreeze,
        inferred_batchnorm,
        inferred_dropout,
        inferred_output_logits,
    ) = infer_model_type(checkpoint_path)
    model_type = args.model_type if args.model_type is not None else inferred_model_type
    pretrained = args.pretrained if args.pretrained is not None else inferred_pretrained
    unfreeze_last_block = args.unfreeze_last_block if args.unfreeze_last_block is not None else inferred_unfreeze
    use_batchnorm = args.use_batchnorm if args.use_batchnorm is not None else inferred_batchnorm
    dropout = args.dropout if args.dropout is not None else inferred_dropout
    output_logits = args.output_logits if args.output_logits is not None else inferred_output_logits

    model = build_model(
        model_type=model_type,
        pretrained=pretrained,
        unfreeze_last_block=unfreeze_last_block,
        use_batchnorm=use_batchnorm,
        dropout=dropout,
        output_logits=output_logits,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    eval_tfms = get_eval_transforms(img_size=args.img_size)
    dataset = TyreManifestDataset(manifest_csv=args.manifest, split=args.split, transforms=eval_tfms)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    metrics, records = evaluate(model, loader, device)

    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_path = plot_confusion(
        np.array(metrics["confusion"]),
        labels=["good", "defect"],
        save_path=out_dir / "confusion_matrix.png",
    )

    misclassification_artifacts = None
    if args.save_misclassified:
        misclassification_artifacts = export_misclassification_artifacts(records, out_dir, args.max_visuals)

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "model_type": model_type,
        "pretrained": pretrained,
        "unfreeze_last_block": unfreeze_last_block,
        "use_batchnorm": use_batchnorm,
        "dropout": dropout,
        "output_logits": output_logits,
        "metrics": {k: float(v) if not isinstance(v, list) else v for k, v in metrics.items()},
        "confusion_matrix": str(cm_path),
        "misclassification_artifacts": misclassification_artifacts,
    }

    report_path = out_dir / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
