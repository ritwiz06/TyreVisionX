"""I/O helpers for web-candidate metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from src.web_collection.schemas import CANDIDATE_COLUMNS, CandidateRecord, QuerySpec


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def load_query_catalog(path: str | Path) -> list[QuerySpec]:
    payload = load_yaml(path)
    queries = payload.get("queries", [])
    return [QuerySpec.from_dict(item) for item in queries]


def records_to_dataframe(records: Iterable[CandidateRecord]) -> pd.DataFrame:
    rows = [record.to_dict() for record in records]
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def write_candidates_csv(path: str | Path, records: Iterable[CandidateRecord] | pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = records if isinstance(records, pd.DataFrame) else records_to_dataframe(records)
    for column in CANDIDATE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df[CANDIDATE_COLUMNS].to_csv(path, index=False)


def read_candidates_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in CANDIDATE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[CANDIDATE_COLUMNS]


def read_manual_url_file(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manual URL file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).fillna("").to_dict(orient="records")
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("candidates", [])
        if not isinstance(payload, list):
            raise ValueError("Manual JSON input must be a list or a dict with a 'candidates' list.")
        return payload
    raise ValueError(f"Unsupported manual URL file format: {path.suffix}. Use CSV or JSON.")
