"""Multi-crop utilities for local anomaly scoring."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def make_multicrop_batch(images: torch.Tensor, crop_fraction: float = 0.82) -> torch.Tensor:
    """Return full image plus center and corner crops resized to input size."""

    _, _, h, w = images.shape
    crop_h = max(1, int(h * crop_fraction))
    crop_w = max(1, int(w * crop_fraction))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    positions = [
        (0, 0),
        (0, w - crop_w),
        (h - crop_h, 0),
        (h - crop_h, w - crop_w),
        (top, left),
    ]
    crops = [images]
    for y, x in positions:
        crop = images[:, :, y : y + crop_h, x : x + crop_w]
        crops.append(F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False))
    return torch.stack(crops, dim=1)


def make_fine_patch_grid_batch(images: torch.Tensor, grid_size: int = 3, include_full: bool = True) -> torch.Tensor:
    """Return a fine grid of resized local patches, optionally with the full image."""

    _, _, h, w = images.shape
    crops = [images] if include_full else []
    for gy in range(grid_size):
        y0 = int(round(gy * h / grid_size))
        y1 = int(round((gy + 1) * h / grid_size))
        for gx in range(grid_size):
            x0 = int(round(gx * w / grid_size))
            x1 = int(round((gx + 1) * w / grid_size))
            crop = images[:, :, y0:y1, x0:x1]
            crops.append(F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False))
    return torch.stack(crops, dim=1)
