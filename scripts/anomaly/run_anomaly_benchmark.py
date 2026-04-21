"""Run the controlled TyreVisionX anomaly benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.benchmark import load_benchmark_config, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anomaly benchmark variants.")
    parser.add_argument("--config", default="configs/anomaly/anomaly_benchmark.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_benchmark_config(config_path)
    result = run_benchmark(config, config_path=config_path)
    print(f"Benchmark complete. Comparison CSV: {result['comparison_csv']}")
    for row in result["variants"]:
        print(
            f"{row['variant']}: {row['run_status']} recall={row.get('test_recall')} "
            f"fn={row.get('test_false_negatives')} artifact={row.get('artifact_dir')}"
        )


if __name__ == "__main__":
    main()
