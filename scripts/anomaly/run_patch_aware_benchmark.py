"""Run feature-map patch-aware anomaly benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.patch_benchmark import load_patch_benchmark_config, run_patch_aware_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run patch-aware anomaly benchmark.")
    parser.add_argument("--config", default="configs/anomaly/anomaly_patch_aware_benchmark.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_patch_aware_benchmark(load_patch_benchmark_config(args.config), config_path=args.config)
    print(f"Patch-aware benchmark complete: {result['comparison_csv']}")
    for row in result["variants"]:
        print(
            f"{row['variant']}: recall={row.get('test_recall')} "
            f"fn={row.get('test_false_negatives')} fp={row.get('test_false_positives')} "
            f"artifact={row.get('artifact_dir')}"
        )


if __name__ == "__main__":
    main()
