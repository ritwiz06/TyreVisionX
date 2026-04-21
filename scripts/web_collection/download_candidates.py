"""Download or copy candidate images listed in metadata.

Network downloads are optional and should only be used with approved providers
and source URLs. Local paths and file:// URLs are supported for smoke testing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.io import load_yaml, read_candidates_csv, write_candidates_csv


def _extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"


def _copy_or_download(source_url: str, out_path: Path, timeout: int) -> None:
    parsed = urlparse(source_url)
    if parsed.scheme in {"", "file"}:
        source_path = Path(parsed.path if parsed.scheme == "file" else source_url)
        if not source_path.exists():
            raise FileNotFoundError(f"Local source image does not exist: {source_path}")
        shutil.copy2(source_path, out_path)
        return
    if parsed.scheme in {"http", "https"}:
        request = urllib.request.Request(source_url, headers={"User-Agent": "TyreVisionX-research-curation/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            out_path.write_bytes(response.read())
        return
    raise ValueError(f"Unsupported source URL scheme for download: {parsed.scheme}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/copy candidate images from metadata.")
    parser.add_argument("--config", default="configs/web_collection/web_collection.yaml")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    metadata_path = args.metadata or config["storage"]["metadata_csv_path"]
    output_path = args.out or metadata_path
    raw_dir = Path(config["storage"]["raw_image_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    timeout = int(config.get("download", {}).get("timeout_seconds", 20))

    df = read_candidates_csv(metadata_path)
    for idx, row in df.iterrows():
        candidate_id = str(row["candidate_id"])
        source_url = str(row["source_url"] or "")
        local_source_path = str(row.get("local_source_path") or "")
        if not source_url and local_source_path:
            source_url = Path(local_source_path).expanduser().resolve().as_uri()
        out_path = raw_dir / f"{candidate_id}{_extension_from_url(source_url)}"
        try:
            _copy_or_download(source_url, out_path, timeout=timeout)
            df.at[idx, "local_raw_path"] = str(out_path)
            df.at[idx, "download_status"] = "downloaded"
        except Exception as exc:
            df.at[idx, "download_status"] = "failed"
            note = str(row.get("review_notes") or "")
            df.at[idx, "review_notes"] = ";".join(part for part in [note, f"download_failed:{type(exc).__name__}:{exc}"] if part)

    write_candidates_csv(output_path, df)
    print(f"Updated download metadata: {output_path}")


if __name__ == "__main__":
    main()
