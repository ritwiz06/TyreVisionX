"""Albumentations pipelines for training/evaluation."""
from __future__ import annotations

from typing import Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _random_resized_crop(img_size: int, scale: tuple[float, float], ratio: tuple[float, float], p: float) -> A.BasicTransform:
    """Create RandomResizedCrop across albumentations 1.x/2.x APIs."""
    try:
        return A.RandomResizedCrop(size=(img_size, img_size), scale=scale, ratio=ratio, p=p)
    except Exception:
        return A.RandomResizedCrop(height=img_size, width=img_size, scale=scale, ratio=ratio, p=p)


def _coarse_dropout(p: float) -> A.BasicTransform:
    """Create CoarseDropout across albumentations 1.x/2.x APIs."""
    try:
        return A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.04, 0.12),
            hole_width_range=(0.04, 0.12),
            fill=0,
            p=p,
        )
    except Exception:
        return A.CoarseDropout(
            max_holes=4,
            max_height=48,
            max_width=48,
            min_holes=1,
            min_height=16,
            min_width=16,
            p=p,
        )


def get_train_transforms(
    img_size: int = 384,
    preset: Literal["none", "light", "day5", "strong"] = "day5",
) -> A.Compose:
    if preset == "none":
        ops = [
            A.Resize(height=img_size, width=img_size),
        ]
    elif preset == "light":
        ops = [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.15),
        ]
    elif preset == "day5":
        # Day-5 recipe: crop + mild geometric/lighting augmentation for stability.
        ops = [
            _random_resized_crop(img_size=img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
        ]
    elif preset == "strong":
        ops = [
            _random_resized_crop(img_size=img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.Perspective(scale=0.05, p=0.3),
            _coarse_dropout(p=0.4),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        ]
    else:
        raise ValueError(f"Unknown preset: {preset}")

    ops.extend([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    return A.Compose(ops)


def get_eval_transforms(img_size: int = 384) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
