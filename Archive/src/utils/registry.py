"""Simple on-disk model registry."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.utils.paths import get_registry_root, ensure_dir


REGISTRY_DIR = ensure_dir(get_registry_root() / "registry")
REGISTRY_FILE = REGISTRY_DIR / "registry.json"


def _load_registry() -> Dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_registry(registry: Dict) -> None:
    ensure_dir(REGISTRY_DIR)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_model(exp_name: str, model_dir: Path, metadata: Optional[Dict] = None) -> Dict:
    registry = _load_registry()
    entries = registry.get(exp_name, [])
    version_id = len(entries) + 1
    entry = {
        "exp_name": exp_name,
        "version": version_id,
        "model_dir": str(model_dir.resolve()),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata or {},
    }
    entries.append(entry)
    registry[exp_name] = entries
    _save_registry(registry)
    return entry


def get_latest_model(exp_name: str) -> Optional[Path]:
    registry = _load_registry()
    entries = registry.get(exp_name, [])
    if not entries:
        return None
    latest = entries[-1]
    return Path(latest["model_dir"])


def list_models() -> Dict:
    return _load_registry()
