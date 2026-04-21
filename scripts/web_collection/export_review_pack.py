"""Export a visual review pack for manual or AI-assisted candidate review."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.pilot import copy_review_pack_images


def _thumbnail(path: Path, size: int = 220) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _write_contact_sheet(queue: pd.DataFrame, out_path: Path, cols: int = 3) -> bool:
    rows = []
    for _, row in queue.iterrows():
        image_path = Path(str(row.get("review_pack_image") or row.get("local_raw_path") or ""))
        if image_path.exists():
            rows.append((row, image_path))
    if not rows:
        return False
    tile_w, tile_h = 300, 290
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, sheet_rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (row, image_path) in enumerate(rows):
        col = idx % cols
        r = idx // cols
        x, y = col * tile_w, r * tile_h
        sheet.paste(_thumbnail(image_path), (x + 40, y + 10))
        draw.text((x + 8, y + 238), str(row["candidate_id"])[:32], fill="black", font=font)
        draw.text((x + 8, y + 254), f"quality={row.get('quality_status', '')}", fill="black", font=font)
        draw.text((x + 8, y + 270), f"triage={row.get('anomaly_triage_bucket', '')}", fill="black", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


def export_review_pack(queue_csv: str | Path, out_dir: str | Path) -> dict[str, str | int | bool]:
    queue = pd.read_csv(queue_csv).fillna("")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_queue = copy_review_pack_images(queue, out_dir)
    pack_queue_path = out_dir / "review_pack_queue.csv"
    pack_queue.to_csv(pack_queue_path, index=False)
    contact_sheet_path = out_dir / "contact_sheet_001.png"
    contact_sheet_written = _write_contact_sheet(pack_queue, contact_sheet_path)

    gallery = out_dir / "review_gallery.md"
    lines = [
        "# Pilot Review Gallery",
        "",
        "Use this gallery and contact sheet for human or AI-assisted review. Do not treat suggestions as final labels.",
        "",
    ]
    for _, row in pack_queue.iterrows():
        lines.extend(
            [
                f"## {row['candidate_id']}",
                "",
                f"- local path: `{row.get('local_raw_path', '')}`",
                f"- review-pack image: `{row.get('review_pack_image', '')}`",
                f"- source URL: {row.get('source_url', '')}",
                f"- page URL: {row.get('page_url', '')}",
                f"- query: `{row.get('query_text', '')}`",
                f"- quality status: `{row.get('quality_status', '')}`",
                f"- anomaly triage: `{row.get('anomaly_triage_bucket', '')}`",
                "",
            ]
        )
        if row.get("review_pack_image"):
            lines.extend([f"![{row['candidate_id']}]({Path(row['review_pack_image']).name})", ""])
    gallery.write_text("\n".join(lines), encoding="utf-8")

    report = Path("reports/web_collection/pilot_review_pack.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Pilot Review Pack\n\n"
        f"Queue CSV: `{queue_csv}`\n\n"
        f"Review pack directory: `{out_dir}`\n\n"
        f"Rows: `{len(pack_queue)}`\n\n"
        f"Contact sheet created: `{contact_sheet_written}`\n\n"
        "Upload `contact_sheet_001.png` or the copied images plus `review_pack_queue.csv` to ChatGPT for assisted review. "
        "Then enter final human decisions in `reports/web_collection/pilot_review_decisions_template.csv`.\n",
        encoding="utf-8",
    )
    return {
        "rows": int(len(pack_queue)),
        "out_dir": str(out_dir),
        "queue_csv": str(pack_queue_path),
        "gallery": str(gallery),
        "contact_sheet": str(contact_sheet_path),
        "contact_sheet_written": contact_sheet_written,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a pilot review pack.")
    parser.add_argument("--queue", default="data/interim/web_candidates/pilot_01/review_queue.csv")
    parser.add_argument("--out_dir", default="data/interim/web_candidates/pilot_01/review_pack")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(export_review_pack(args.queue, args.out_dir))


if __name__ == "__main__":
    main()
