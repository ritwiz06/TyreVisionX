"""Evaluation helpers for anomaly scores."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

from src.anomaly.thresholds import metrics_at_threshold


def safe_auroc(targets: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(targets, scores))
    except ValueError:
        return None


def safe_auprc(targets: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        return float(average_precision_score(targets, scores))
    except ValueError:
        return None


def build_predictions(records: list[dict[str, Any]], targets: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    df = pd.DataFrame(records).copy()
    df["target"] = targets.astype(int)
    df["anomaly_score"] = scores.astype(float)
    df["threshold"] = float(threshold)
    df["pred"] = (df["anomaly_score"] >= threshold).astype(int)
    df["is_false_negative"] = ((df["target"] == 1) & (df["pred"] == 0)).astype(int)
    df["is_false_positive"] = ((df["target"] == 0) & (df["pred"] == 1)).astype(int)
    return df


def evaluate_scores(targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    threshold_metrics = metrics_at_threshold(targets, scores, threshold)
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    return {
        "threshold_metrics": threshold_metrics,
        "confusion_matrix": cm.tolist(),
        "auroc": safe_auroc(targets, scores),
        "auprc": safe_auprc(targets, scores),
    }


def plot_score_distributions(df: pd.DataFrame, save_path: str | Path, title: str) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    normal = df[df["target"] == 0]["anomaly_score"]
    anomaly = df[df["target"] == 1]["anomaly_score"]
    ax.hist(normal, bins=30, alpha=0.65, label="normal/good")
    ax.hist(anomaly, bins=30, alpha=0.65, label="anomaly/defect")
    if "threshold" in df.columns and len(df):
        ax.axvline(float(df["threshold"].iloc[0]), color="black", linestyle="--", label="threshold")
    ax.set_title(title)
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return str(save_path)


def plot_pr_curve(targets: np.ndarray, scores: np.ndarray, save_path: str | Path) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(targets, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Anomaly recall")
    ax.set_ylabel("Anomaly precision")
    ax.set_title("Anomaly PR Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return str(save_path)


def plot_confusion_matrix(cm: list[list[int]] | np.ndarray, save_path: str | Path, title: str) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["normal", "anomaly"])
    ax.set_yticks([0, 1], labels=["normal", "anomaly"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return str(save_path)
