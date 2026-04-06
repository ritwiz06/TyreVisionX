"""Run Day-5 long training + threshold/PR analysis across seeds and model variants."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import TyreManifestDataset
from src.models.simple_cnn import SimpleCNN
from src.transforms import get_eval_transforms


@dataclass
class ModelVariant:
    key: str
    label: str
    lr: float
    weight_decay: float
    preset: str
    use_batchnorm: bool
    dropout: float
    output_logits: bool

    def train_args(self) -> List[str]:
        args = [
            "--model_type",
            "simple_cnn",
            "--img_size",
            "224",
            "--preset",
            self.preset,
            "--lr",
            str(self.lr),
            "--weight_decay",
            str(self.weight_decay),
            "--dropout",
            str(self.dropout),
        ]
        args.append("--use_batchnorm" if self.use_batchnorm else "--no-use_batchnorm")
        args.append("--output_logits" if self.output_logits else "--no-output_logits")
        return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day-5 long-run seed sweep")
    parser.add_argument("--manifest", default="data/processed/D1_manifest.csv")
    parser.add_argument("--out_root", default="artifacts/day5/longrun_seed_sweep")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 123, 999])
    parser.add_argument("--max_threshold", type=float, default=0.9)
    parser.add_argument("--min_threshold", type=float, default=0.1)
    parser.add_argument("--threshold_step", type=float, default=0.01)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_training(
    variant: ModelVariant,
    seed: int,
    manifest: str,
    out_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "src/train_baseline.py",
        "--manifest",
        manifest,
        "--epochs",
        str(epochs),
        "--early_stopping_patience",
        str(patience),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--out_dir",
        str(out_dir),
        *variant.train_args(),
    ]

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str((ROOT / "artifacts" / ".mplconfig").resolve()))
    env.setdefault("XDG_CACHE_HOME", str((ROOT / "artifacts" / ".cache").resolve()))

    log_path = out_dir / "train.log"
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed for {variant.key} seed={seed}. See {log_path}")


def _load_model(run_dir: Path) -> tuple[SimpleCNN, bool]:
    info = json.loads((run_dir / "model_info.json").read_text(encoding="utf-8"))
    model = SimpleCNN(
        in_channels=3,
        use_batchnorm=bool(info.get("use_batchnorm", False)),
        dropout=float(info.get("dropout", 0.0)),
        output_logits=bool(info.get("output_logits", False)),
    )
    state = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, bool(info.get("output_logits", False))


def _collect_probs(run_dir: Path, manifest: str, split: str) -> pd.DataFrame:
    model, output_logits = _load_model(run_dir)
    ds = TyreManifestDataset(manifest_csv=manifest, split=split, transforms=get_eval_transforms(img_size=224))
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for images, labels, meta in dl:
            scores = model(images).reshape(-1)
            probs = torch.sigmoid(scores) if output_logits else scores
            probs_np = probs.detach().cpu().numpy()
            y_np = labels.numpy().astype(int)
            paths = meta["image_path"] if isinstance(meta, dict) else [""] * len(y_np)
            for p, y, pr in zip(paths, y_np, probs_np):
                rows.append({"image_path": str(p), "target": int(y), "prob_defect": float(pr)})
    return pd.DataFrame(rows)


def _metrics_at_threshold(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    y_true = df["target"].to_numpy(dtype=int)
    y_prob = df["prob_defect"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _choose_threshold(sweep_df: pd.DataFrame, min_recall: float = 0.99) -> float:
    valid = sweep_df[sweep_df["recall"] >= min_recall].copy()
    if not valid.empty:
        valid = valid.sort_values(["precision", "recall", "threshold"], ascending=[False, False, False])
        return float(valid.iloc[0]["threshold"])
    fallback = sweep_df.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False])
    return float(fallback.iloc[0]["threshold"])


def _plot_threshold_curve(sweep_df: pd.DataFrame, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sweep_df["recall"], sweep_df["precision"], lw=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def evaluate_and_export(run_dir: Path, manifest: str, thresholds: np.ndarray) -> Dict[str, float]:
    val_df = _collect_probs(run_dir, manifest, split="val")
    test_df = _collect_probs(run_dir, manifest, split="test")

    val_sweep_rows = [_metrics_at_threshold(val_df, float(t)) for t in thresholds]
    val_sweep = pd.DataFrame(val_sweep_rows)
    chosen_threshold = _choose_threshold(val_sweep, min_recall=0.99)

    val_sweep.to_csv(run_dir / "threshold_sweep_val.csv", index=False)
    _plot_threshold_curve(val_sweep, run_dir / "threshold_recall_precision_val.png", "Validation Precision-Recall vs Threshold")

    test_metrics = _metrics_at_threshold(test_df, chosen_threshold)
    y_true = test_df["target"].to_numpy(dtype=int)
    y_prob = test_df["prob_defect"].to_numpy(dtype=float)

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.0
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = 0.0

    p_arr, r_arr, thr_arr = precision_recall_curve(y_true, y_prob)
    pr_df = pd.DataFrame({
        "precision": p_arr,
        "recall": r_arr,
        "threshold": np.append(thr_arr, np.nan),
    })
    pr_df.to_csv(run_dir / "pr_curve_test.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r_arr, p_arr, lw=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Test PR Curve (AUPRC={auprc:.4f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "pr_curve_test.png", dpi=180)
    plt.close(fig)

    test_df = test_df.copy()
    test_df["pred"] = (test_df["prob_defect"] >= chosen_threshold).astype(int)
    test_df["prob_good"] = 1.0 - test_df["prob_defect"]
    test_df["is_misclassified"] = (test_df["pred"] != test_df["target"]).astype(int)
    test_df["is_false_negative"] = ((test_df["target"] == 1) & (test_df["pred"] == 0)).astype(int)
    test_df["is_false_positive"] = ((test_df["target"] == 0) & (test_df["pred"] == 1)).astype(int)
    test_df.to_csv(run_dir / "predictions_test.csv", index=False)
    test_df[test_df["is_misclassified"] == 1].to_csv(run_dir / "misclassified.csv", index=False)
    test_df[test_df["is_false_negative"] == 1].to_csv(run_dir / "false_negatives.csv", index=False)
    test_df[test_df["is_false_positive"] == 1].to_csv(run_dir / "false_positives.csv", index=False)

    report = {
        "chosen_threshold": chosen_threshold,
        "val_best": _metrics_at_threshold(val_df, chosen_threshold),
        "test": {
            **test_metrics,
            "auroc": auroc,
            "auprc": auprc,
        },
    }
    with open(run_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {
        "chosen_threshold": chosen_threshold,
        "test_recall": test_metrics["recall"],
        "test_precision": test_metrics["precision"],
        "test_f1": test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_fn": int(test_metrics["fn"]),
        "test_fp": int(test_metrics["fp"]),
        "test_auroc": auroc,
        "test_auprc": auprc,
    }


def plot_model_pr_curves(out_root: Path, model_key: str, model_label: str, seeds: List[int]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for seed in seeds:
        run_dir = out_root / model_key / f"seed_{seed}"
        pr_path = run_dir / "pr_curve_test.csv"
        if not pr_path.exists():
            continue
        pr_df = pd.read_csv(pr_path)
        ax.plot(pr_df["recall"], pr_df["precision"], lw=1.5, label=f"seed {seed}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Test PR Curves - {model_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_root / f"pr_curves_{model_key}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    variants = [
        ModelVariant(
            key="baseline",
            label="Baseline",
            lr=1e-3,
            weight_decay=0.0,
            preset="none",
            use_batchnorm=False,
            dropout=0.0,
            output_logits=False,
        ),
        ModelVariant(
            key="augmentation",
            label="+ Augmentation",
            lr=1e-3,
            weight_decay=1e-4,
            preset="day5",
            use_batchnorm=False,
            dropout=0.0,
            output_logits=False,
        ),
        ModelVariant(
            key="batchnorm",
            label="+ BatchNorm",
            lr=3e-4,
            weight_decay=1e-4,
            preset="day5",
            use_batchnorm=True,
            dropout=0.0,
            output_logits=True,
        ),
        ModelVariant(
            key="dropout",
            label="+ Dropout",
            lr=3e-4,
            weight_decay=1e-4,
            preset="day5",
            use_batchnorm=True,
            dropout=0.3,
            output_logits=True,
        ),
    ]

    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(args.min_threshold, args.max_threshold + 1e-9, args.threshold_step)

    rows: List[Dict[str, object]] = []
    for variant in variants:
        for seed in args.seeds:
            run_dir = out_root / variant.key / f"seed_{seed}"
            done_marker = run_dir / "eval_report.json"
            if args.skip_existing and done_marker.exists():
                print(f"[skip] {variant.key} seed={seed} (existing)")
            else:
                print(f"[train] {variant.key} seed={seed}")
                run_training(
                    variant=variant,
                    seed=seed,
                    manifest=args.manifest,
                    out_dir=run_dir,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                print(f"[eval] {variant.key} seed={seed}")
                evaluate_and_export(run_dir, args.manifest, thresholds)

            report = json.loads((run_dir / "eval_report.json").read_text(encoding="utf-8"))
            metrics = report["test"]
            rows.append(
                {
                    "model_key": variant.key,
                    "model": variant.label,
                    "seed": seed,
                    "threshold": report["chosen_threshold"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "fn": metrics["fn"],
                    "fp": metrics["fp"],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "run_dir": str(run_dir),
                }
            )

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(out_root / "summary_runs.csv", index=False)

    agg_df = (
        runs_df.groupby(["model_key", "model"], as_index=False)
        .agg(
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            fn_mean=("fn", "mean"),
            fn_std=("fn", "std"),
            fp_mean=("fp", "mean"),
            fp_std=("fp", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
        )
        .sort_values("recall_mean", ascending=False)
    )
    agg_df.to_csv(out_root / "summary_by_model.csv", index=False)

    for variant in variants:
        plot_model_pr_curves(out_root, variant.key, variant.label, args.seeds)

    print("\n=== Per-run summary ===")
    print(runs_df.to_string(index=False))
    print("\n=== Aggregated by model (mean/std across seeds) ===")
    print(agg_df.to_string(index=False))
    print(f"\nSaved outputs to: {out_root}")


if __name__ == "__main__":
    main()
