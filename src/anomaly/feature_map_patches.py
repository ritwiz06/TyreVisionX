"""Feature-map patch descriptors for patch-aware anomaly detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.torchvision_compat import load_torchvision_models


@dataclass
class PatchFeatureBatch:
    """Patch descriptors for a dataset split."""

    patch_embeddings: np.ndarray
    image_indices: np.ndarray
    targets: np.ndarray
    records: list[dict[str, Any]]
    patches_per_image: int


class ResNetFeatureMapExtractor(nn.Module):
    """Frozen ResNet feature-map extractor for patch descriptors."""

    def __init__(
        self,
        backbone: str = "resnet50",
        layer: str = "layer3",
        weights_path: str | Path | None = None,
        pretrained: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError("Patch-aware benchmark currently supports resnet50 only.")
        if layer not in {"layer2", "layer3", "layer4", "layer2_layer3"}:
            raise ValueError("layer must be one of: layer2, layer3, layer4, layer2_layer3")

        models = load_torchvision_models()
        use_torchvision_weights = pretrained and not weights_path
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_torchvision_weights else None
        model = models.resnet50(weights=weights)
        self.weight_source = "torchvision_imagenet" if use_torchvision_weights else "none"

        if weights_path:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Configured patch feature weights_path does not exist: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            cleaned = _clean_resnet_state_dict(state_dict)
            load_result = model.load_state_dict(cleaned, strict=False)
            loaded_keys = set(cleaned) - set(load_result.unexpected_keys)
            loaded_conv_keys = [key for key in loaded_keys if key.startswith("conv1") or key.startswith("layer")]
            if not loaded_conv_keys:
                raise RuntimeError("weights_path did not load compatible ResNet convolution keys.")
            self.weight_source = f"local_weights_path:{weights_path}"

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
        self.layer2_module = model.layer2
        self.layer3_module = model.layer3
        self.layer4_module = model.layer4
        self.backbone = backbone
        self.layer = layer
        self.normalize = normalize
        self.embedding_dim = _layer_dim(layer)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.stem(images)
            layer2 = self.layer2_module(x)
            if self.layer == "layer2":
                return _maybe_normalize(layer2, self.normalize)

            layer3 = self.layer3_module(layer2)
            if self.layer == "layer3":
                return _maybe_normalize(layer3, self.normalize)
            if self.layer == "layer2_layer3":
                layer2_norm = _maybe_normalize(layer2, self.normalize)
                layer3_norm = _maybe_normalize(layer3, self.normalize)
                layer3_up = F.interpolate(layer3_norm, size=layer2_norm.shape[-2:], mode="bilinear", align_corners=False)
                return _maybe_normalize(torch.cat([layer2_norm, layer3_up], dim=1), self.normalize)

            layer4 = self.layer4_module(layer3)
            return _maybe_normalize(layer4, self.normalize)


def extract_patch_features(
    extractor: ResNetFeatureMapExtractor,
    loader: DataLoader,
    device: torch.device,
    max_patches_per_image: int | None = None,
) -> PatchFeatureBatch:
    """Extract flattened patch descriptors from a feature map."""

    extractor.to(device)
    extractor.eval()
    patch_batches: list[np.ndarray] = []
    image_index_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    patches_per_image_seen: int | None = None
    offset = 0

    with torch.no_grad():
        for images, targets, meta in loader:
            images = images.to(device)
            features = extractor(images)
            b, c, h, w = features.shape
            patches = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
            if max_patches_per_image and max_patches_per_image < h * w:
                idx = torch.linspace(0, h * w - 1, max_patches_per_image, device=patches.device).round().long()
                patches = patches[:, idx, :]
            pnum = patches.shape[1]
            patches_per_image_seen = pnum
            patch_batches.append(patches.reshape(b * pnum, c).detach().cpu().numpy())
            image_index_batches.append(np.repeat(np.arange(offset, offset + b), pnum))
            targets_np = targets.detach().cpu().numpy().astype(int)
            target_batches.append(targets_np)
            for i in range(b):
                records.append(
                    {
                        "image_path": str(_meta_value(meta, "image_path", i)),
                        "target": int(targets_np[i]),
                        "label_str": str(_meta_value(meta, "label_str", i)),
                        "split": str(_meta_value(meta, "split", i)),
                        "source_dataset": str(_meta_value(meta, "source_dataset", i)),
                        "product_type": str(_meta_value(meta, "product_type", i)),
                    }
                )
            offset += b

    if not patch_batches or patches_per_image_seen is None:
        raise ValueError("No patch features extracted; dataset/loader is empty.")

    return PatchFeatureBatch(
        patch_embeddings=np.concatenate(patch_batches, axis=0),
        image_indices=np.concatenate(image_index_batches, axis=0).astype(int),
        targets=np.concatenate(target_batches, axis=0),
        records=records,
        patches_per_image=int(patches_per_image_seen),
    )


def _layer_dim(layer: str) -> int:
    return {"layer2": 512, "layer3": 1024, "layer4": 2048, "layer2_layer3": 1536}[layer]


def _maybe_normalize(features: torch.Tensor, normalize: bool) -> torch.Tensor:
    return F.normalize(features, p=2, dim=1) if normalize else features


def _meta_value(meta: Any, key: str, index: int) -> Any:
    if isinstance(meta, dict):
        value = meta.get(key, "")
        if isinstance(value, (list, tuple)):
            return value[index]
        if torch.is_tensor(value):
            return value[index].item()
        return value
    return ""


def _clean_resnet_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("model.", "module.", "backbone.", "encoder.", "feature_extractor.", "net.")
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        if new_key.startswith("resnet."):
            new_key = new_key[len("resnet.") :]
        cleaned[new_key] = value
    return cleaned
