"""Frozen CNN feature extraction for anomaly baselines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.torchvision_compat import load_torchvision_models


@dataclass
class FeatureBatch:
    embeddings: np.ndarray
    targets: np.ndarray
    records: list[dict[str, Any]]


class ResNetEmbeddingExtractor(nn.Module):
    """Frozen ResNet pooled embedding extractor."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        normalize: bool = True,
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        models = load_torchvision_models()
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.normalize = normalize
        self.weights_path = str(weights_path) if weights_path else None
        self.weight_source = "torchvision_imagenet" if pretrained and not weights_path else "none"
        use_torchvision_weights = pretrained and not weights_path

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_torchvision_weights else None
            model = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_torchvision_weights else None
            model = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_torchvision_weights else None
            model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported anomaly backbone: {backbone}")

        if weights_path:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Configured feature_extractor.weights_path does not exist: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            cleaned = _clean_resnet_state_dict(state_dict)
            load_result = model.load_state_dict(cleaned, strict=False)
            loaded_keys = set(cleaned) - set(load_result.unexpected_keys)
            loaded_conv_keys = [key for key in loaded_keys if key.startswith("conv1") or key.startswith("layer")]
            if not loaded_conv_keys:
                raise RuntimeError(
                    "Configured weights_path did not load any ResNet convolution/backbone keys. "
                    "Use a torchvision ResNet state dict or a compatible TyreVisionX ResNet checkpoint."
                )
            self.weight_source = f"local_weights_path:{weights_path}"

        self.embedding_dim = int(model.fc.in_features)
        self.encoder = nn.Sequential(*(list(model.children())[:-1]))
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(images).flatten(1)
            if self.normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features


def _meta_value(meta: Any, key: str, index: int) -> Any:
    if isinstance(meta, dict):
        value = meta.get(key, "")
        if isinstance(value, (list, tuple)):
            return value[index]
        if torch.is_tensor(value):
            return value[index].item()
        return value
    return ""


def extract_embeddings(
    model: ResNetEmbeddingExtractor,
    loader: DataLoader,
    device: torch.device,
) -> FeatureBatch:
    model.to(device)
    model.eval()

    embedding_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, targets, meta in loader:
            images = images.to(device)
            embeddings = model(images).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy().astype(int)
            embedding_batches.append(embeddings)
            target_batches.append(targets_np)

            for i in range(len(targets_np)):
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

    if not embedding_batches:
        raise ValueError("No embeddings extracted; dataset/loader is empty.")

    return FeatureBatch(
        embeddings=np.concatenate(embedding_batches, axis=0),
        targets=np.concatenate(target_batches, axis=0),
        records=records,
    )


def save_embeddings(path: str | Path, embeddings: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)


def _clean_resnet_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize common checkpoint key prefixes to torchvision ResNet keys."""

    cleaned: dict[str, Any] = {}
    prefixes = (
        "model.",
        "module.",
        "backbone.",
        "encoder.",
        "feature_extractor.",
        "net.",
    )
    for key, value in state_dict.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        # Feature-extractor checkpoints may wrap ResNet as model.backbone.<key>.
        if new_key.startswith("resnet."):
            new_key = new_key[len("resnet.") :]
        cleaned[new_key] = value
    return cleaned
