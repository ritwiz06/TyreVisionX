"""Run lower/mid-level patch-aware anomaly benchmark configs and merge results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.patch_benchmark import load_patch_benchmark_config, run_patch_aware_benchmark, write_patch_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layer3 and layer2+layer3 patch-aware benchmarks.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/anomaly/anomaly_patch_aware_layer3.yaml",
            "configs/anomaly/anomaly_patch_aware_layer23.yaml",
        ],
    )
    parser.add_argument("--out_csv", default="reports/anomaly/patch_layer_benchmark.csv")
    parser.add_argument("--out_report", default="reports/anomaly/patch_layer_benchmark.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for config_path in args.configs:
        cfg = load_patch_benchmark_config(config_path)
        result = run_patch_aware_benchmark(cfg, config_path=config_path)
        frames.append(pd.read_csv(result["comparison_csv"]))

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["variant"], keep="first")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    write_patch_comparison_report(merged, args.out_report)
    print(f"Patch layer benchmark complete: {out_csv}")
    for _, row in merged.iterrows():
        print(
            f"{row['variant']}: recall={row.get('test_recall')} "
            f"fn={row.get('test_false_negatives')} fp={row.get('test_false_positives')} "
            f"artifact={row.get('artifact_dir')}"
        )


if __name__ == "__main__":
    main()
