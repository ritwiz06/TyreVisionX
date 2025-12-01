import cv2
import numpy as np
import pandas as pd
import torch

from src.data.datasets import TyreManifestDataset


def test_manifest_dataset(tmp_path):
    img_path = tmp_path / "img.png"
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(img_path), img)

    manifest = tmp_path / "manifest.csv"
    df = pd.DataFrame(
        [
            {
                "image_path": str(img_path),
                "label_str": "defect",
                "label": 1,
                "dataset_id": "DTEST",
                "split": "train",
            }
        ]
    )
    df.to_csv(manifest, index=False)

    ds = TyreManifestDataset(manifest_path=manifest, split="train", transforms=None)
    assert len(ds) == 1
    image, label, meta = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3
    assert label == 1
    assert meta["dataset_id"] == "DTEST"
