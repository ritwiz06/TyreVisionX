"""ResNet classifier builders."""
from __future__ import annotations

from typing import Literal

import torch.nn as nn

from src.utils.torchvision_compat import load_torchvision_models

models = load_torchvision_models()


def build_resnet(model_name: Literal["resnet18", "resnet34"], num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
