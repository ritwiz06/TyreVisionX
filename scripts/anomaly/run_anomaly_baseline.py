"""Run the first TyreVisionX good-only anomaly baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pformat

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.io import load_yaml, write_json
from src.anomaly.pipeline import run_anomaly_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit/calibrate/evaluate the anomaly baseline.")
    parser.add_argument("--config", default="configs/anomaly/anomaly_baseline.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all"],
        help="Currently only 'all' is supported: fit normal, calibrate validation threshold, evaluate test once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing anomaly config: {config_path}")

    config = load_yaml(config_path)
    try:
        metadata = run_anomaly_baseline(config, config_path=config_path)
    except Exception as exc:
        failure_path = Path(config.get("outputs", {}).get("run_dir", "artifacts/anomaly/failed")) / "failure.json"
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "help": (
                "Check that anomaly manifests exist under data/manifests/ and that pretrained "
                "torchvision weights are available when feature_extractor.pretrained=true. "
                "For offline runs, set feature_extractor.weights_path or cache the expected torchvision .pth file."
            ),
        }
        write_json(failure_path, payload)
        print("Anomaly baseline failed with a clear prerequisite error:")
        print(pformat(payload))
        print(f"Failure details written to: {failure_path}")
        raise SystemExit(1) from exc

    print(f"Anomaly baseline complete: {metadata['outputs']['run_dir']}")
    print(f"Chosen threshold: {metadata['threshold']['value']:.6f}")
    print(f"Test metrics: {metadata['test']['threshold_metrics']}")


if __name__ == "__main__":
    main()
