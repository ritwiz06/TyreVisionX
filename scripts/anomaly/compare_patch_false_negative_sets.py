"""Compare high-recall reference false negatives against a patch-aware run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anomaly.compare_false_negative_sets import compare_false_negative_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare patch-aware false-negative sets.")
    parser.add_argument(
        "--reference_csv",
        default="artifacts/anomaly/local_features/resnet50_knn_threshold_sweep/false_negatives_test.csv",
    )
    parser.add_argument(
        "--candidate_csv",
        default="artifacts/anomaly/patch_aware/resnet50_featuremap_patch_knn_threshold_sweep/false_negatives_test.csv",
    )
    parser.add_argument("--out_dir", default="reports/anomaly/patch_false_negative_overlap")
    parser.add_argument("--reference_name", default="resnet50_knn_threshold_sweep_reference")
    parser.add_argument("--candidate_name", default="resnet50_featuremap_patch_knn_threshold_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_false_negative_sets(
        reference_csv=args.reference_csv,
        candidate_csv=args.candidate_csv,
        out_dir=args.out_dir,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
    )
    print(result)


if __name__ == "__main__":
    main()
