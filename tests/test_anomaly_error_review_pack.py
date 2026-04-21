import pandas as pd
from PIL import Image

from scripts.anomaly.export_error_review_pack import export_error_review_pack
from scripts.anomaly.compare_patch_false_negative_sets import compare_false_negative_sets


def test_error_review_pack_exports_tables_and_sheets(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), "white").save(image_path)
    row = {
        "image_path": str(image_path),
        "target": 1,
        "anomaly_score": 0.2,
        "threshold": 0.5,
        "pred": 0,
    }
    pd.DataFrame([row]).to_csv(run_dir / "false_negatives_test.csv", index=False)
    pd.DataFrame([{**row, "target": 0, "pred": 1}]).to_csv(run_dir / "false_positives_test.csv", index=False)

    result = export_error_review_pack(run_dir, tmp_path / "reports", "demo")

    assert result["false_negatives"] == 1
    assert (tmp_path / "reports" / "demo_error_review.md").exists()
    assert (tmp_path / "reports" / "demo_false_negative_contact_sheet.png").exists()


def test_patch_false_negative_compare_wrapper_imports(tmp_path):
    ref = tmp_path / "ref.csv"
    cand = tmp_path / "cand.csv"
    pd.DataFrame({"image_path": ["a.jpg"], "target": [1]}).to_csv(ref, index=False)
    pd.DataFrame({"image_path": ["b.jpg"], "target": [1]}).to_csv(cand, index=False)
    result = compare_false_negative_sets(ref, cand, tmp_path / "out")
    assert result["fixed_count"] == 1
    assert result["new_miss_count"] == 1
