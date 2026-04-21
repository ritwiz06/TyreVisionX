"""Export review tables and contact sheets for anomaly errors."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def export_error_review_pack(
    run_dir: str | Path,
    output_dir: str | Path = "reports/anomaly",
    prefix: str = "high_recall",
    max_images: int = 48,
) -> dict[str, str | int]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fn = _read_error_csv(run_dir / "false_negatives_test.csv")
    fp = _read_error_csv(run_dir / "false_positives_test.csv")
    fn_table = output_dir / f"{prefix}_false_negative_review_table.csv"
    fp_table = output_dir / f"{prefix}_false_positive_review_table.csv"
    fn.to_csv(fn_table, index=False)
    fp.to_csv(fp_table, index=False)

    fn_sheet = output_dir / f"{prefix}_false_negative_contact_sheet.png"
    fp_sheet = output_dir / f"{prefix}_false_positive_contact_sheet.png"
    fn_status = write_contact_sheet(fn, fn_sheet, title=f"{prefix} false negatives", max_images=max_images)
    fp_status = write_contact_sheet(fp, fp_sheet, title=f"{prefix} false positives", max_images=max_images)

    report = output_dir / f"{prefix}_error_review.md"
    report.write_text(
        "\n".join(
            [
                "# High-Recall Error Review",
                "",
                f"Source run: `{run_dir}`",
                "",
                "| Error Type | Count | Review Table | Contact Sheet |",
                "|---|---:|---|---|",
                f"| False negatives | {len(fn)} | `{fn_table}` | {fn_status} |",
                f"| False positives | {len(fp)} | `{fp_table}` | {fp_status} |",
                "",
                "## Interpretation Notes",
                "- False negatives are defective tyres still scored below the high-recall threshold.",
                "- False positives are good tyres scored above the high-recall threshold.",
                "- The tables and contact sheets are review aids only; visual failure causes require human inspection.",
                "- Likely patterns to check: subtle cracks, edge defects, glare, low contrast, unusual tread texture, or possible label ambiguity.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "report": str(report),
        "false_negatives": len(fn),
        "false_positives": len(fp),
        "false_negative_table": str(fn_table),
        "false_positive_table": str(fp_table),
        "false_negative_contact_sheet": str(fn_sheet) if fn_sheet.exists() else fn_status,
        "false_positive_contact_sheet": str(fp_sheet) if fp_sheet.exists() else fp_status,
    }


def _read_error_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing error CSV: {path}")
    df = pd.read_csv(path)
    if "image_path" not in df.columns:
        raise ValueError(f"Expected image_path column in {path}")
    return df


def write_contact_sheet(df: pd.DataFrame, out_path: str | Path, title: str, max_images: int = 48) -> str:
    out_path = Path(out_path)
    rows = df.head(max_images).to_dict(orient="records")
    if not rows:
        return "No rows to display."

    tiles = []
    for row in rows:
        path = Path(str(row["image_path"]))
        if not path.exists():
            continue
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((180, 180))
                tile = Image.new("RGB", (230, 245), "white")
                tile.paste(img, ((230 - img.width) // 2, 8))
                draw = ImageDraw.Draw(tile)
                score = float(row.get("anomaly_score", 0.0))
                threshold = float(row.get("threshold", 0.0))
                draw.text((8, 194), path.name[:32], fill=(0, 0, 0))
                draw.text((8, 212), f"score={score:.3f} th={threshold:.3f}", fill=(0, 0, 0))
                tiles.append(tile)
        except Exception:
            continue

    if not tiles:
        return "Local image files were unavailable or unreadable."

    cols = 4
    rows_count = (len(tiles) + cols - 1) // cols
    header_h = 32
    sheet = Image.new("RGB", (cols * 230, header_h + rows_count * 245), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 230, header_h + (idx // cols) * 245))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return f"`{out_path}`"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export review pack for anomaly false negatives/positives.")
    parser.add_argument("--run_dir", default="artifacts/anomaly/local_features/resnet50_knn_threshold_sweep")
    parser.add_argument("--output_dir", default="reports/anomaly")
    parser.add_argument("--prefix", default="high_recall")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(export_error_review_pack(args.run_dir, args.output_dir, args.prefix))


if __name__ == "__main__":
    main()
