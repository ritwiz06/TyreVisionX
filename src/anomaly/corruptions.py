"""Realistic image corruptions for anomaly robustness evaluation."""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CorruptionSpec:
    """One deterministic corruption setting."""

    name: str
    family: str
    level: str
    params: dict[str, Any]


def default_corruptions() -> list[CorruptionSpec]:
    """Moderate nuisance corruptions relevant to tyre image inspection."""

    return [
        CorruptionSpec("gaussian_noise_low", "gaussian_noise", "low", {"sigma": 6.0}),
        CorruptionSpec("gaussian_noise_medium", "gaussian_noise", "medium", {"sigma": 12.0}),
        CorruptionSpec("gaussian_blur_low", "gaussian_blur", "low", {"kernel": 3, "sigma": 0.6}),
        CorruptionSpec("gaussian_blur_medium", "gaussian_blur", "medium", {"kernel": 5, "sigma": 1.0}),
        CorruptionSpec("jpeg_compression_mild", "jpeg_compression", "mild", {"quality": 85}),
        CorruptionSpec("brightness_darker", "brightness_shift", "darker", {"beta": -25}),
        CorruptionSpec("brightness_brighter", "brightness_shift", "brighter", {"beta": 25}),
        CorruptionSpec("contrast_lower", "contrast_shift", "lower", {"alpha": 0.82}),
        CorruptionSpec("contrast_higher", "contrast_shift", "higher", {"alpha": 1.18}),
    ]


def spec_from_dict(payload: dict[str, Any]) -> CorruptionSpec:
    return CorruptionSpec(
        name=str(payload["name"]),
        family=str(payload["family"]),
        level=str(payload.get("level", "")),
        params=dict(payload.get("params", {})),
    )


def apply_corruption(image_rgb: np.ndarray, spec: CorruptionSpec | None, seed: int = 0) -> np.ndarray:
    """Apply a moderate corruption to an RGB uint8 image.

    ``spec=None`` or ``spec.name == "clean"`` returns a copy of the original.
    """

    image = image_rgb.copy()
    if spec is None or spec.name == "clean":
        return image
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if spec.family == "gaussian_noise":
        rng = np.random.default_rng(seed)
        sigma = float(spec.params.get("sigma", 6.0))
        noise = rng.normal(0.0, sigma, size=image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if spec.family == "gaussian_blur":
        kernel = int(spec.params.get("kernel", 3))
        if kernel % 2 == 0:
            kernel += 1
        sigma = float(spec.params.get("sigma", 0.6))
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma)

    if spec.family == "jpeg_compression":
        quality = int(spec.params.get("quality", 85))
        pil = Image.fromarray(image)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return np.asarray(Image.open(buffer).convert("RGB"))

    if spec.family == "brightness_shift":
        beta = float(spec.params.get("beta", 0.0))
        return np.clip(image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    if spec.family == "contrast_shift":
        alpha = float(spec.params.get("alpha", 1.0))
        mean = np.array([127.5], dtype=np.float32)
        return np.clip((image.astype(np.float32) - mean) * alpha + mean, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported corruption family: {spec.family}")
