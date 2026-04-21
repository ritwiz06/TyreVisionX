from pathlib import Path

import pandas as pd

from scripts.data.import_roboflow_export import (
    get_dataset_entry,
    load_registry,
    normalize_label,
    prepare_import_manifest,
)


def test_external_dataset_registry_parses():
    registry = load_registry("configs/data/external_dataset_registry.yaml")
    ids = {entry["id"] for entry in registry["datasets"]}
    assert "roboflow_good_tire_bad_tire" in ids
    assert "roboflow_tires_defects_omar" in ids
    assert "roboflow_tire_quality_tirescanner" in ids


def test_registry_license_blocks_pending_items():
    registry = load_registry("configs/data/external_dataset_registry.yaml")
    tire_quality = get_dataset_entry(registry, "roboflow_tire_quality_tirescanner")
    college = get_dataset_entry(registry, "roboflow_tire_college_segmentation")
    assert "pending" in tire_quality["license_status"]
    assert "pending" in college["license_status"]


def test_label_normalization_case_insensitive():
    mapping = {"Good": "normal", "Defected": "anomaly"}
    assert normalize_label("good", mapping) == "normal"
    assert normalize_label("Defected", mapping) == "anomaly"
    assert normalize_label("unknown", mapping) == "unmapped"


def test_import_scaffold_classification_manifest(tmp_path: Path):
    export_dir = tmp_path / "export"
    image_dir = export_dir / "train" / "good"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.jpg"
    image_path.write_bytes(b"not-real-image-but-path-test")
    out_csv = tmp_path / "review.csv"

    prepare_import_manifest(
        export_dir=export_dir,
        dataset_id="roboflow_good_tire_bad_tire",
        registry_path="configs/data/external_dataset_registry.yaml",
        out_csv=out_csv,
    )
    df = pd.read_csv(out_csv)
    assert len(df) == 1
    assert df.loc[0, "normalized_label"] == "normal"
    assert int(df.loc[0, "target"]) == 0
    assert df.loc[0, "import_status"] == "review_required"
