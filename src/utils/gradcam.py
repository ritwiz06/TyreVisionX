"""Grad-CAM utilities for ResNet backbones."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_full_backward_hook(backward_hook)
        self.target_layer.register_forward_hook(forward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1)[0])
        score = logits[:, class_idx].sum()
        score.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()
        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze(0).squeeze(0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam_np = cam.cpu().numpy()
        cam_np = cv2.resize(cam_np, (x.shape[3], x.shape[2]))
        return cam_np


def overlay_heatmap(image_tensor: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay heatmap on a single image tensor (CHW normalized)."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * IMAGENET_STD + IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def generate_gradcam(image_tensor: torch.Tensor, model: nn.Module, class_idx: int | None = None) -> np.ndarray:
    """Generate Grad-CAM overlay for a single-image batch."""
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    # try to locate last conv layer of ResNet
    target_layer = None
    for name in ["layer4", "features", "conv5"]:
        module = getattr(model, name, None)
        if module is not None:
            target_layer = list(module.modules())[-1] if isinstance(module, nn.Sequential) else module
            break
    if target_layer is None:
        raise ValueError("Could not find target layer for Grad-CAM")

    cam_generator = GradCAM(model, target_layer)
    heatmap = cam_generator(image_tensor, class_idx=class_idx)
    overlay = overlay_heatmap(image_tensor[0], heatmap)
    return overlay
