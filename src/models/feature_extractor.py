"""Frozen pretrained feature extractor with a trainable linear head."""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from src.utils.torchvision_compat import load_torchvision_models

models = load_torchvision_models()


class FrozenFeatureExtractorClassifier(nn.Module):
    """Pretrained ResNet backbone + linear classifier head."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_block: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_block = unfreeze_last_block

        self.backbone, feature_dim = self._build_backbone(backbone_name, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if self.unfreeze_last_block:
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True

    @staticmethod
    def _build_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
        if backbone_name not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if backbone_name == "resnet18":
            weights_enum = models.ResNet18_Weights.IMAGENET1K_V1
            constructor = models.resnet18
        elif backbone_name == "resnet34":
            weights_enum = models.ResNet34_Weights.IMAGENET1K_V1
            constructor = models.resnet34
        else:
            weights_enum = models.ResNet50_Weights.IMAGENET1K_V2
            constructor = models.resnet50

        weights = None
        if pretrained:
            try:
                weights = weights_enum
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"Could not use pretrained weights for {backbone_name}: {exc}. Falling back to random init."
                )
                weights = None

        try:
            backbone = constructor(weights=weights)
        except Exception as exc:  # pragma: no cover
            if pretrained:
                warnings.warn(
                    f"Failed to load pretrained {backbone_name} ({exc}). Falling back to random init."
                )
                backbone = constructor(weights=None)
            else:
                raise

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
            if self.unfreeze_last_block:
                self.backbone.layer4.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.freeze_backbone and not self.unfreeze_last_block:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features)
