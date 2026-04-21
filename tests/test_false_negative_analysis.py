from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.anomaly.analyze_false_negatives import build_review_table, summarize_false_negatives


def test_false_negative_analysis_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "false_negatives.csv"
    csv_path.write_text(
        "image_path,target,label_str,split,source_dataset,product_type,anomaly_score,threshold,pred,is_false_negative,is_false_positive\n"
        "/tmp/Defective (1).jpg,1,defect,test,D1,tyre,9.5,10.0,0,1,0\n"
        "/tmp/Defective (2).jpg,1,defect,test,D1,tyre,7.0,10.0,0,1,0\n",
        encoding="utf-8",
    )
    summary = summarize_false_negatives(csv_path)
    assert summary["count"] == 2
    assert summary["near_threshold_count"] == 1
    review = build_review_table(csv_path, tmp_path / "review.csv")
    assert "margin_below_threshold" in review.columns
    assert review.iloc[0]["filename"] == "Defective (1).jpg"
    assert pd.read_csv(tmp_path / "review.csv").shape[0] == 2
