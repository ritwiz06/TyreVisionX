import cv2
import numpy as np
import pandas as pd

from scripts.data.create_anomaly_manifests import create_anomaly_manifests


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    cv2.imwrite(str(path), image)


def test_create_anomaly_manifests_smoke(tmp_path):
    root = tmp_path / "raw"
    for rel in ["good/a.png", "good/b.png", "defect/c.png", "defect/d.png", "good/e.png", "defect/f.png"]:
        _write_image(root / rel)

    source = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {"image_path": "good/a.png", "label": 0, "label_str": "good", "split": "train", "dataset_id": "DTEST"},
            {"image_path": "good/b.png", "label": 0, "label_str": "good", "split": "train", "dataset_id": "DTEST"},
            {"image_path": "defect/c.png", "label": 1, "label_str": "defect", "split": "train", "dataset_id": "DTEST"},
            {"image_path": "good/e.png", "label": 0, "label_str": "good", "split": "val", "dataset_id": "DTEST"},
            {"image_path": "defect/d.png", "label": 1, "label_str": "defect", "split": "val", "dataset_id": "DTEST"},
            {"image_path": "defect/f.png", "label": 1, "label_str": "defect", "split": "test", "dataset_id": "DTEST"},
            {"image_path": "good/a.png", "label": 0, "label_str": "good", "split": "test", "dataset_id": "DTEST"},
        ]
    ).to_csv(source, index=False)

    outputs = create_anomaly_manifests(
        source_manifest=source,
        output_dir=tmp_path / "out",
        dataset_id="DTEST",
        product_type="tyre",
        image_roots=[root],
        report_path=tmp_path / "report.md",
    )

    train = pd.read_csv(outputs.normal_train)
    val = pd.read_csv(outputs.val_mixed)
    test = pd.read_csv(outputs.test_mixed)

    assert set(train["target"]) == {0}
    assert set(val["target"]) == {0, 1}
    assert set(test["target"]) == {0, 1}
    assert {"image_path", "target", "is_normal", "source_dataset", "product_type"}.issubset(train.columns)

