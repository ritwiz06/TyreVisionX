"""Analyze false negatives from the completed D1 anomaly baseline."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _filename_index(path_text: str) -> int | None:
    match = re.search(r"\((\d+)\)", Path(path_text).name)
    return int(match.group(1)) if match else None


def summarize_false_negatives(false_negatives_csv: str | Path) -> dict:
    path = Path(false_negatives_csv)
    if not path.exists():
        raise FileNotFoundError(f"Missing false negatives CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return {"count": 0, "score_min": None, "score_max": None, "score_mean": None, "near_threshold_count": 0}
    distances = (df["threshold"].astype(float) - df["anomaly_score"].astype(float)).clip(lower=0)
    return {
        "count": int(len(df)),
        "score_min": float(df["anomaly_score"].min()),
        "score_max": float(df["anomaly_score"].max()),
        "score_mean": float(df["anomaly_score"].mean()),
        "threshold": float(df["threshold"].iloc[0]),
        "mean_margin_below_threshold": float(distances.mean()),
        "near_threshold_count": int((distances <= 1.0).sum()),
        "far_below_threshold_count": int((distances > 3.0).sum()),
    }


def build_review_table(false_negatives_csv: str | Path, out_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(false_negatives_csv)
    if df.empty:
        review = df.copy()
    else:
        review = df.copy()
        review["filename"] = review["image_path"].map(lambda value: Path(str(value)).name)
        review["filename_index"] = review["image_path"].map(_filename_index)
        review["margin_below_threshold"] = review["threshold"].astype(float) - review["anomaly_score"].astype(float)
        review["review_priority"] = pd.cut(
            review["margin_below_threshold"],
            bins=[-float("inf"), 0.5, 1.5, 3.0, float("inf")],
            labels=["near_threshold", "moderately_near", "below_threshold", "far_below_threshold"],
        ).astype(str)
        review["likely_inference_from_metadata"] = (
            "Filename metadata only confirms defect class; visual inspection required to determine if cracks are small, "
            "low contrast, edge-located, or label-questionable."
        )
        review = review.sort_values(["margin_below_threshold", "filename"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(out_csv, index=False)
    return review


def make_contact_sheet(review_df: pd.DataFrame, out_png: str | Path, max_images: int = 48) -> bool:
    rows = []
    for _, row in review_df.head(max_images).iterrows():
        path = Path(str(row["image_path"]))
        if path.exists():
            rows.append((row, path))
    if not rows:
        return False

    thumb = 180
    label_h = 54
    cols = 4
    tile_w = 240
    tile_h = thumb + label_h
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, sheet_rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (row, path) in enumerate(rows):
        col = idx % cols
        r = idx // cols
        x = col * tile_w
        y = r * tile_h
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb, thumb))
        sheet.paste(img, (x + (tile_w - img.width) // 2, y + 4))
        draw.text((x + 6, y + thumb + 8), str(row.get("filename", path.name))[:34], fill="black", font=font)
        draw.text(
            (x + 6, y + thumb + 23),
            f"score={float(row['anomaly_score']):.2f} th={float(row['threshold']):.2f}",
            fill="black",
            font=font,
        )
        draw.text((x + 6, y + thumb + 38), f"priority={row.get('review_priority', '')}", fill="black", font=font)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return True


def write_report(summary: dict, review_csv: str | Path, contact_sheet: str | Path, out_md: str | Path) -> None:
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# False Negative Analysis",
        "",
        "Status: completed from the first D1 anomaly run.",
        "",
        "## Summary",
        "",
        f"- false negatives: `{summary['count']}`",
        f"- threshold: `{summary.get('threshold')}`",
        f"- score range: `{summary.get('score_min')}` to `{summary.get('score_max')}`",
        f"- mean score: `{summary.get('score_mean')}`",
        f"- near threshold count (<= 1.0 below threshold): `{summary.get('near_threshold_count')}`",
        f"- far below threshold count (> 3.0 below threshold): `{summary.get('far_below_threshold_count')}`",
        "",
        "## Outputs",
        "",
        f"- review table: `{review_csv}`",
        f"- contact sheet: `{contact_sheet}`",
        "",
        "## Interpretation",
        "",
        "The CSV metadata confirms these are defect-labeled tyres that scored below the anomaly threshold. "
        "It does not reveal the visual cause by itself. Visual review is required to determine whether missed "
        "defects are small cracks, low-contrast defects, edge-of-image defects, or possible label noise.",
        "",
        "Global pooled ResNet features can miss local tyre defects because a small crack may occupy only a small "
        "part of the image. Pooling compresses the whole image into one vector, so local evidence can be diluted "
        "by normal tread, sidewall, and lighting patterns.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze D1 anomaly false negatives.")
    parser.add_argument(
        "--false_negatives",
        default="artifacts/anomaly/d1_resnet18_mahalanobis_v1/false_negatives_test.csv",
    )
    parser.add_argument("--out_csv", default="reports/anomaly/false_negative_review_table.csv")
    parser.add_argument("--out_png", default="reports/anomaly/false_negative_contact_sheet.png")
    parser.add_argument("--out_md", default="reports/anomaly/false_negative_analysis.md")
    parser.add_argument("--out_json", default="reports/anomaly/false_negative_analysis.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_false_negatives(args.false_negatives)
    review = build_review_table(args.false_negatives, args.out_csv)
    contact_written = make_contact_sheet(review, args.out_png)
    summary["contact_sheet_written"] = contact_written
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, args.out_csv, args.out_png, args.out_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
