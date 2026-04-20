"""Path helpers respecting environment variables."""
from __future__ import annotations

import os
from pathlib import Path

DATA_ENV = "DATA_ROOT"
REGISTRY_ENV = "MODEL_REGISTRY"


def get_data_root() -> Path:
    return Path(os.getenv(DATA_ENV, "./data")).resolve()


def get_registry_root() -> Path:
    return Path(os.getenv(REGISTRY_ENV, "./artifacts")).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
