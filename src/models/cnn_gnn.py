"""CNN->GNN hybrid classifier (optional torch-geometric)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from torch_geometric.nn import GATConv, SAGEConv, global_max_pool, global_mean_pool

    HAS_PYG = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PYG = False


def _grid_edge_index(h: int, w: int, device: torch.device) -> torch.Tensor:
    edges = []
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        n_idx = ny * w + nx
                        edges.append((idx, n_idx))
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index


class CNNGNNClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 2,
        gnn_type: str = "gat",
        patch_grid: Tuple[int, int] = (7, 7),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for CNNGNNClassifier. Install it or disable gnn.enabled.")

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported model {model_name}")

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        hidden_dim = backbone.fc.in_features
        self.patch_grid = patch_grid
        self.pool = nn.AdaptiveAvgPool2d(patch_grid)

        if gnn_type == "gat":
            self.conv1 = GATConv(hidden_dim, hidden_dim // 2, heads=2, concat=False)
            self.conv2 = GATConv(hidden_dim // 2, hidden_dim // 2, heads=2, concat=False)
        else:
            self.conv1 = SAGEConv(hidden_dim, hidden_dim // 2)
            self.conv2 = SAGEConv(hidden_dim // 2, hidden_dim // 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        feats = self.feature_extractor(x)
        feats = self.pool(feats)
        b, c, h, w = feats.shape
        n = h * w
        node_feats = feats.view(b, c, n).permute(0, 2, 1)  # [B, N, C]

        base_edge = _grid_edge_index(h, w, device=x.device)
        all_edges = []
        offset = 0
        for _ in range(b):
            all_edges.append(base_edge + offset)
            offset += n
        edge_index = torch.cat(all_edges, dim=1)

        batch_vec = torch.arange(b, device=x.device).repeat_interleave(n)
        node_feats = node_feats.reshape(b * n, c)

        h1 = F.relu(self.conv1(node_feats, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))

        pooled = torch.cat(
            [
                global_mean_pool(h2, batch_vec),
                global_max_pool(h2, batch_vec),
            ],
            dim=1,
        )
        logits = self.classifier(pooled)
        return logits
