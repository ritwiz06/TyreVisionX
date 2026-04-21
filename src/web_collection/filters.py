"""Deduplication and quality filters for web-image candidates."""
from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class FilterResult:
    kept: pd.DataFrame
    rejected: pd.DataFrame
    review_needed: pd.DataFrame
    summary: dict[str, Any]


def compute_content_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_average_hash(path: str | Path, hash_size: int = 8) -> str:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS))
    avg = float(arr.mean())
    bits = (arr >= avg).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance_hex(left: str, right: str) -> int:
    if not left or not right:
        return 10**9
    return int(bin(int(left, 16) ^ int(right, 16)).count("1"))


def blur_score_laplacian(path: str | Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def inspect_image(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Local image path does not exist: {path}")
    with Image.open(path) as img:
        width, height = img.size
        img.verify()
    return {
        "content_hash": compute_content_hash(path),
        "perceptual_hash": compute_average_hash(path),
        "width": int(width),
        "height": int(height),
        "blur_score": blur_score_laplacian(path),
        "file_size_bytes": int(path.stat().st_size),
        "mime_type": mimetypes.guess_type(path.name)[0] or "",
    }


def _quality_decision(row: pd.Series, cfg: dict[str, Any]) -> tuple[str, str]:
    hard_reasons: list[str] = []
    soft_reasons: list[str] = []
    min_width = int(cfg.get("min_width", 224))
    min_height = int(cfg.get("min_height", 224))
    max_aspect = float(cfg.get("max_aspect_ratio", 4.0))
    min_blur = float(cfg.get("blur_laplacian_threshold", 30.0))
    max_file_size = cfg.get("max_file_size_bytes")

    if row.get("download_status") != "downloaded":
        hard_reasons.append("not_downloaded")
    if int(row.get("width") or 0) < min_width or int(row.get("height") or 0) < min_height:
        hard_reasons.append("below_min_resolution")
    width = float(row.get("width") or 0)
    height = float(row.get("height") or 0)
    if width <= 0 or height <= 0:
        hard_reasons.append("invalid_dimensions")
    elif max(width / height, height / width) > max_aspect:
        soft_reasons.append("unusual_aspect_ratio")
    if max_file_size and int(row.get("file_size_bytes") or 0) > int(max_file_size):
        soft_reasons.append("large_file")
    if float(row.get("blur_score") or 0.0) < min_blur:
        soft_reasons.append("possibly_blurry")

    if hard_reasons:
        return "rejected", ";".join(hard_reasons + soft_reasons)
    if soft_reasons:
        return "review_needed", ";".join(soft_reasons)
    return "kept", ""


def apply_quality_and_dedupe_filters(df: pd.DataFrame, config: dict[str, Any]) -> FilterResult:
    """Inspect local files and split candidates into kept/rejected/review-needed."""

    df = df.copy()
    quality_cfg = config.get("quality_filtering", {})
    dedupe_cfg = config.get("deduplication", {})
    exact_seen: dict[str, str] = {}
    phash_seen: list[tuple[str, str]] = []
    phash_threshold = int(dedupe_cfg.get("hash_distance_threshold", 6))

    statuses: list[str] = []
    notes: list[str] = []
    dedupe_groups: list[str] = []

    for idx, row in df.iterrows():
        local_path = str(row.get("local_raw_path") or "").strip()
        status = "pending"
        note = ""
        dedupe_group = ""
        try:
            if not local_path:
                raise FileNotFoundError("local_raw_path is empty")
            info = inspect_image(local_path)
            for key, value in info.items():
                df.at[idx, key] = value
            df.at[idx, "download_status"] = row.get("download_status") or "downloaded"

            content_hash = str(info["content_hash"])
            phash = str(info["perceptual_hash"])
            if dedupe_cfg.get("exact_file_hash", True) and content_hash in exact_seen:
                status, note = "rejected", "duplicate_exact_hash"
                dedupe_group = exact_seen[content_hash]
            elif dedupe_cfg.get("perceptual_hash", True):
                for seen_hash, seen_id in phash_seen:
                    if hamming_distance_hex(phash, seen_hash) <= phash_threshold:
                        status, note = "review_needed", f"possible_duplicate_phash:{seen_id}"
                        dedupe_group = seen_id
                        break

            candidate_id = str(row.get("candidate_id") or idx)
            exact_seen.setdefault(content_hash, candidate_id)
            phash_seen.append((phash, candidate_id))
            if not status or status == "pending":
                status, note = _quality_decision(df.loc[idx], quality_cfg)
        except Exception as exc:
            status = "rejected"
            note = f"image_validation_failed:{type(exc).__name__}:{exc}"
            df.at[idx, "download_status"] = row.get("download_status") or "failed"

        statuses.append(status)
        notes.append(note)
        dedupe_groups.append(dedupe_group)

    df["quality_status"] = statuses
    df["review_notes"] = [
        ";".join(part for part in [str(existing or ""), note] if part)
        for existing, note in zip(df.get("review_notes", ""), notes)
    ]
    df["dedupe_group"] = dedupe_groups

    kept = df[df["quality_status"] == "kept"].copy()
    rejected = df[df["quality_status"] == "rejected"].copy()
    review_needed = df[df["quality_status"] == "review_needed"].copy()
    summary = {
        "total": int(len(df)),
        "kept": int(len(kept)),
        "rejected": int(len(rejected)),
        "review_needed": int(len(review_needed)),
    }
    return FilterResult(kept=kept, rejected=rejected, review_needed=review_needed, summary=summary)
