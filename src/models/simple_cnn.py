"""Small CNN baseline for tyre defect classification (binary sigmoid output)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Phase-1 tiny baseline:

    Input(224x224x3) -> Conv16 -> (BatchNorm) -> ReLU -> MaxPool ->
    Conv32 -> (BatchNorm) -> ReLU -> MaxPool ->
    Flatten -> (Dropout) -> Linear -> (optional Sigmoid)
    """

    def __init__(
        self,
        in_channels: int = 3,
        use_batchnorm: bool = False,
        dropout: float = 0.0,
        output_logits: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.LazyLinear(1)
        self.output_logits = output_logits
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        if self.output_logits:
            return logits
        return self.sigmoid(logits)
