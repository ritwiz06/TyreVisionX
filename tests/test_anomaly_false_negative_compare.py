import pandas as pd

from scripts.anomaly.compare_false_negative_sets import compare_false_negative_sets


def test_false_negative_overlap_reporting(tmp_path):
    ref_csv = tmp_path / "ref.csv"
    cand_csv = tmp_path / "cand.csv"
    pd.DataFrame({"image_path": ["a.jpg", "b.jpg", "c.jpg"], "target": [1, 1, 1]}).to_csv(ref_csv, index=False)
    pd.DataFrame({"image_path": ["b.jpg", "d.jpg"], "target": [1, 1]}).to_csv(cand_csv, index=False)

    result = compare_false_negative_sets(ref_csv, cand_csv, tmp_path / "out")

    assert result["fixed_count"] == 2
    assert result["still_missed_count"] == 1
    assert result["new_miss_count"] == 1
    assert (tmp_path / "out" / "false_negative_overlap_analysis.md").exists()
