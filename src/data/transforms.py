"""Albumentations pipelines for training and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import albumentations as A
import yaml
from albumentations.pytorch import ToTensorV2

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

TRANSFORM_MAP = {
    "HorizontalFlip": A.HorizontalFlip,
    "Rotate": A.Rotate,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "GaussianBlur": A.GaussianBlur,
    "RandomResizedCrop": A.RandomResizedCrop,
    "Perspective": A.Perspective,
    "CoarseDropout": A.CoarseDropout,
    "ColorJitter": A.ColorJitter,
    "Resize": A.Resize,
}


def _load_config(config_path_or_dict: Any) -> Dict:
    if isinstance(config_path_or_dict, (str, Path)):
        with open(config_path_or_dict, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if isinstance(config_path_or_dict, dict):
        return config_path_or_dict
    raise ValueError("Invalid transform config; expected path or dict")


def _build_ops(ops_cfg: List[Dict]) -> List:
    ops = []
    for op in ops_cfg:
        name = op.get("name")
        if not name or name not in TRANSFORM_MAP:
            continue
        params = {k: v for k, v in op.items() if k != "name"}
        ops.append(TRANSFORM_MAP[name](**params))
    return ops


def get_train_transforms(config_path_or_dict: Any) -> A.Compose:
    cfg = _load_config(config_path_or_dict)
    size = cfg.get("size", [384, 384])
    ops_cfg = cfg.get("train", cfg.get("augmentations", []))
    ops = _build_ops(ops_cfg)

    if not any(isinstance(op, (A.Resize, A.RandomResizedCrop)) for op in ops):
        ops.append(A.Resize(height=size[0], width=size[1]))

    mean = cfg.get("normalize", {}).get("mean", DEFAULT_MEAN)
    std = cfg.get("normalize", {}).get("std", DEFAULT_STD)

    ops.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(ops)


def get_eval_transforms(config_path_or_dict: Any) -> A.Compose:
    cfg = _load_config(config_path_or_dict)
    size = cfg.get("size", [384, 384])
    mean = cfg.get("normalize", {}).get("mean", DEFAULT_MEAN)
    std = cfg.get("normalize", {}).get("std", DEFAULT_STD)

    ops = [A.Resize(height=size[0], width=size[1]), A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(ops)
