"""Logging utilities using loguru."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


_LOG_CONFIGURED = False


def configure_logging(save_dir: Optional[Path] = None) -> None:
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_path = save_dir / "run.log"
        logger.add(log_path, level="INFO", rotation="5 MB")

    _LOG_CONFIGURED = True


def get_logger(name: str = "TyreVisionX"):
    if not _LOG_CONFIGURED:
        configure_logging()
    return logger.bind(context=name)
