import numpy as np
import torch

from src.data.transforms import get_eval_transforms, get_train_transforms


def test_train_transforms_shape():
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    tfm = get_train_transforms("configs/aug_light.yaml")
    out = tfm(image=img)
    tensor = out["image"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 3


def test_eval_transforms_deterministic():
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    tfm = get_eval_transforms("configs/aug_light.yaml")
    out1 = tfm(image=img)["image"]
    out2 = tfm(image=img)["image"]
    assert torch.allclose(out1, out2)
