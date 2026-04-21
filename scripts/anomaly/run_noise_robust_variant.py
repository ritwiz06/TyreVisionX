"""Run a mild noise-robust ResNet50 kNN anomaly variant."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.io import load_yaml
from src.anomaly.robustness import run_noise_robust_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mild noise-robust anomaly variant.")
    parser.add_argument("--config", default="configs/anomaly/anomaly_resnet50_knn_noise_robust.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_noise_robust_variant(load_yaml(args.config), config_path=args.config)
    metrics = metadata["test"]["threshold_metrics"]
    print(f"Noise-robust variant complete: {metadata['outputs']['run_dir']}")
    print(f"Test recall={metrics['recall']:.4f} fn={metrics['fn']} fp={metrics['fp']}")


if __name__ == "__main__":
    main()
