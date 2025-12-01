"""Model builders for TyreVisionX."""
from .resnet_classifier import build_resnet
from .cnn_gnn import CNNGNNClassifier

__all__ = ["build_resnet", "CNNGNNClassifier"]
