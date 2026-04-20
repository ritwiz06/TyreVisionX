import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.datasets import TyreManifestDataset, load_dataset_from_runtime_config


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


def test_runtime_data_config_resolves_dataset_root(tmp_path):
    dataset_root = tmp_path / "D1_tyrenet"
    defect_dir = dataset_root / "defect"
    defect_dir.mkdir(parents=True)
    img_path = defect_dir / "sample.png"
    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(img_path), img)

    manifest = tmp_path / "D1_manifest.csv"
    pd.DataFrame(
        [
            {
                "image_path": "defect/sample.png",
                "label_str": "defect",
                "label": 1,
                "dataset_id": "D1",
                "split": "train",
            }
        ]
    ).to_csv(manifest, index=False)

    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "use_datasets": ["D1"],
                "paths": {
                    "D1": {
                        "root": str(dataset_root),
                        "manifest": str(manifest),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    dataset, manifests = load_dataset_from_runtime_config(
        {"config_file": str(config_path)},
        split="train",
        transforms=None,
    )
    image, label, meta = dataset[0]
    assert manifests == [str(manifest)]
    assert isinstance(image, torch.Tensor)
    assert label == 1
    assert meta["image_path"] == str(img_path)
