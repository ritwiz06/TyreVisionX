"""Compare false-negative sets between two anomaly runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def load_false_negatives(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"False-negative CSV not found: {path}")
    df = pd.read_csv(path)
    if "image_path" not in df.columns:
        raise ValueError(f"False-negative CSV must include image_path: {path}")
    return df


def compare_false_negative_sets(
    reference_csv: str | Path,
    candidate_csv: str | Path,
    out_dir: str | Path,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> dict[str, str | int]:
    ref = load_false_negatives(reference_csv)
    cand = load_false_negatives(candidate_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_paths = set(ref["image_path"].astype(str))
    cand_paths = set(cand["image_path"].astype(str))
    fixed_paths = sorted(ref_paths - cand_paths)
    new_miss_paths = sorted(cand_paths - ref_paths)
    still_missed_paths = sorted(ref_paths & cand_paths)

    fixed = ref[ref["image_path"].astype(str).isin(fixed_paths)].copy()
    new_misses = cand[cand["image_path"].astype(str).isin(new_miss_paths)].copy()
    still = ref[ref["image_path"].astype(str).isin(still_missed_paths)].copy()

    fixed_csv = out_dir / "false_negatives_fixed_by_candidate.csv"
    new_csv = out_dir / "false_negatives_new_in_candidate.csv"
    still_csv = out_dir / "false_negatives_still_missed_by_both.csv"
    fixed.to_csv(fixed_csv, index=False)
    new_misses.to_csv(new_csv, index=False)
    still.to_csv(still_csv, index=False)

    contact_sheet = out_dir / "false_negative_overlap_contact_sheet.png"
    contact_sheet_status = _write_contact_sheet(fixed, still, new_misses, contact_sheet)

    report_path = out_dir / "false_negative_overlap_analysis.md"
    report_path.write_text(
        "\n".join(
            [
                "# False Negative Overlap Analysis",
                "",
                f"Reference run: `{reference_name}`",
                f"Candidate run: `{candidate_name}`",
                "",
                "| Group | Count | Output |",
                "|---|---:|---|",
                f"| Fixed by candidate | {len(fixed)} | `{fixed_csv}` |",
                f"| Still missed by both | {len(still)} | `{still_csv}` |",
                f"| New candidate misses | {len(new_misses)} | `{new_csv}` |",
                "",
                f"Contact sheet status: {contact_sheet_status}",
                "",
                "## Interpretation Notes",
                "- Fixed cases are defects the reference missed but the candidate caught.",
                "- Still-missed cases are the highest priority for visual review.",
                "- New candidate misses show any regression introduced by the local-feature method.",
                "- This analysis compares decisions only; it does not prove the visual cause without human inspection.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "report": str(report_path),
        "fixed_csv": str(fixed_csv),
        "still_missed_csv": str(still_csv),
        "new_misses_csv": str(new_csv),
        "contact_sheet": str(contact_sheet) if contact_sheet.exists() else contact_sheet_status,
        "fixed_count": len(fixed),
        "still_missed_count": len(still),
        "new_miss_count": len(new_misses),
    }


def _write_contact_sheet(fixed: pd.DataFrame, still: pd.DataFrame, new_misses: pd.DataFrame, out_path: Path) -> str:
    rows = []
    for label, df in [("fixed", fixed), ("still_missed", still), ("new_miss", new_misses)]:
        for _, row in df.head(12).iterrows():
            rows.append((label, str(row["image_path"])))
    if not rows:
        return "No false-negative rows available for contact sheet."

    thumbs = []
    for label, path_str in rows:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((180, 180))
                tile = Image.new("RGB", (220, 230), "white")
                tile.paste(img, ((220 - img.width) // 2, 8))
                draw = ImageDraw.Draw(tile)
                draw.text((8, 192), label, fill=(0, 0, 0))
                draw.text((8, 210), path.name[:34], fill=(0, 0, 0))
                thumbs.append(tile)
        except Exception:
            continue

    if not thumbs:
        return "No local image files were readable for contact sheet."

    cols = 4
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 220, rows_count * 230), "white")
    for idx, tile in enumerate(thumbs):
        sheet.paste(tile, ((idx % cols) * 220, (idx // cols) * 230))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return f"Wrote `{out_path}`."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare anomaly false-negative CSV files.")
    parser.add_argument("--reference_csv", required=True)
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--out_dir", default="reports/anomaly/false_negative_overlap")
    parser.add_argument("--reference_name", default="resnet50_knn_reference")
    parser.add_argument("--candidate_name", default="local_feature_candidate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_false_negative_sets(
        reference_csv=args.reference_csv,
        candidate_csv=args.candidate_csv,
        out_dir=args.out_dir,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
    )
    print(result)


if __name__ == "__main__":
    main()
