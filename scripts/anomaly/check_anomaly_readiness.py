"""Check whether the TyreVisionX anomaly baseline can run offline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly.io import load_yaml, write_json


EXPECTED_TORCHVISION_WEIGHTS = {
    "resnet18": "resnet18-f37072fd.pth",
    "resnet34": "resnet34-b627a593.pth",
    "resnet50": "resnet50-11ad3fa6.pth",
}


def candidate_cache_dirs() -> list[Path]:
    dirs = [
        ROOT / "artifacts/.torch/hub/checkpoints",
        Path.home() / ".cache/torch/hub/checkpoints",
        Path(os.environ.get("TORCH_HOME", "")) / "hub/checkpoints" if os.environ.get("TORCH_HOME") else None,
    ]
    unique: list[Path] = []
    for path in dirs:
        if path and path not in unique:
            unique.append(path)
    return unique


def find_weight_file(filename: str) -> list[Path]:
    found: list[Path] = []
    for directory in candidate_cache_dirs():
        path = directory / filename
        if path.exists():
            found.append(path)
    return found


def find_project_checkpoints() -> list[Path]:
    roots = [ROOT / "artifacts"]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.pth", "*.pt", "*.ckpt"):
            found.extend(root.rglob(pattern))
    return sorted(found)


def inspect_readiness(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    feature_cfg = config.get("feature_extractor", {})
    backbone = str(feature_cfg.get("backbone", "resnet18"))
    explicit_path = feature_cfg.get("weights_path")
    expected_filename = EXPECTED_TORCHVISION_WEIGHTS.get(backbone)

    candidates: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None

    if explicit_path:
        path = Path(explicit_path)
        item = {
            "source_type": "configured_weights_path",
            "path": str(path),
            "exists": path.exists(),
            "compatible": path.exists(),
            "backbone": backbone,
            "notes": "Explicit config path has highest priority.",
        }
        candidates.append(item)
        if item["compatible"]:
            chosen = item

    if expected_filename:
        for path in find_weight_file(expected_filename):
            item = {
                "source_type": "torchvision_cache",
                "path": str(path),
                "exists": True,
                "compatible": True,
                "backbone": backbone,
                "notes": "Matches configured torchvision backbone filename.",
            }
            candidates.append(item)
            if chosen is None:
                chosen = item

    # Supported fallback: project-local ResNet50 ImageNet cache, already used by historical runs.
    if backbone != "resnet50":
        for path in find_weight_file(EXPECTED_TORCHVISION_WEIGHTS["resnet50"]):
            candidates.append(
                {
                    "source_type": "fallback_torchvision_cache",
                    "path": str(path),
                    "exists": True,
                    "compatible": True,
                    "backbone": "resnet50",
                    "notes": "Supported fallback only if config is changed to backbone=resnet50.",
                }
            )

    project_checkpoints = find_project_checkpoints()
    for path in project_checkpoints[:100]:
        candidates.append(
            {
                "source_type": "project_checkpoint",
                "path": str(path),
                "exists": True,
                "compatible": False,
                "backbone": "unknown",
                "notes": "Found locally, but not selected automatically unless proven ResNet-backbone compatible.",
            }
        )

    manifest_checks = {
        key: Path(config["data"][key]).exists()
        for key in ["normal_train_manifest", "validation_manifest", "test_manifest"]
        if key in config.get("data", {})
    }
    can_run = chosen is not None and all(manifest_checks.values())
    blocker = None
    if not can_run:
        if chosen is None:
            if expected_filename:
                blocker = (
                    f"Missing compatible pretrained weights for {backbone}. "
                    f"Place {expected_filename} under ~/.cache/torch/hub/checkpoints/ or set feature_extractor.weights_path."
                )
            else:
                blocker = f"Unsupported backbone for readiness checker: {backbone}"
        else:
            missing = [key for key, ok in manifest_checks.items() if not ok]
            blocker = f"Missing anomaly manifests: {missing}"

    return {
        "config_path": str(config_path),
        "configured_backbone": backbone,
        "expected_torchvision_weight": expected_filename,
        "candidate_weight_sources": candidates,
        "chosen_weight_source": chosen,
        "manifest_checks": manifest_checks,
        "can_run_real_anomaly_baseline": can_run,
        "blocker": blocker,
    }


def write_markdown_report(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = payload.get("chosen_weight_source")
    lines = [
        "# Anomaly Readiness Report",
        "",
        f"Config: `{payload['config_path']}`",
        f"Configured backbone: `{payload['configured_backbone']}`",
        f"Expected torchvision weight: `{payload['expected_torchvision_weight']}`",
        f"Can run real anomaly baseline now: `{payload['can_run_real_anomaly_baseline']}`",
        "",
        "## Chosen Weight Source",
        "",
    ]
    if chosen:
        lines.extend(
            [
                f"- source type: `{chosen['source_type']}`",
                f"- backbone: `{chosen['backbone']}`",
                f"- path: `{chosen['path']}`",
                f"- notes: {chosen['notes']}",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Manifest Checks", "", "| Manifest | Exists |", "|---|---:|"])
    lines.extend(f"| {key} | `{value}` |" for key, value in payload["manifest_checks"].items())
    lines.extend(["", "## Candidate Weight Sources", "", "| Type | Backbone | Compatible | Path | Notes |", "|---|---|---:|---|---|"])
    for item in payload["candidate_weight_sources"]:
        lines.append(
            f"| {item['source_type']} | {item['backbone']} | `{item['compatible']}` | `{item['path']}` | {item['notes']} |"
        )
    lines.extend(["", "## Blocker", "", payload["blocker"] or "No blocker found. A real run is feasible."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check offline readiness for the anomaly baseline.")
    parser.add_argument("--config", default="configs/anomaly/anomaly_baseline.yaml")
    parser.add_argument("--json", default="reports/anomaly/anomaly_readiness_report.json")
    parser.add_argument("--report", default="reports/anomaly/anomaly_readiness_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = inspect_readiness(args.config)
    write_json(args.json, payload)
    write_markdown_report(payload, args.report)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
