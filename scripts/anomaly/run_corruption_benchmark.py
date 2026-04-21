"""Run corruption robustness evaluation for anomaly variants."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.io import load_yaml
from src.anomaly.robustness import run_corruption_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anomaly corruption benchmark.")
    parser.add_argument("--config", default="configs/anomaly/anomaly_corruption_benchmark.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_corruption_benchmark(load_yaml(args.config), config_path=args.config)
    print(f"Corruption benchmark complete: {result['csv']}")


if __name__ == "__main__":
    main()
