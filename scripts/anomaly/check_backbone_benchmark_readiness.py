"""Check local backbone readiness for controlled anomaly benchmarks."""
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

from src.anomaly.io import write_json


BACKBONES = {
    "resnet18": {
        "expected_files": ["resnet18-f37072fd.pth"],
        "code_support": True,
        "notes": "Reference backbone; already used for the first D1 anomaly run.",
    },
    "resnet50": {
        "expected_files": ["resnet50-11ad3fa6.pth"],
        "code_support": True,
        "notes": "Larger ResNet; may improve semantic features but still uses global pooling unless patch mode is enabled.",
    },
    "efficientnet_b0": {
        "expected_files": ["efficientnet_b0_rwightman-7f5810bc.pth"],
        "code_support": False,
        "notes": "Potential compact CNN backbone; pending local weights and extractor support.",
    },
    "convnext_tiny": {
        "expected_files": ["convnext_tiny-983f1562.pth"],
        "code_support": False,
        "notes": "Modern CNN family; pending local weights and extractor support.",
    },
    "vit_b_16": {
        "expected_files": ["vit_b_16-c867db91.pth"],
        "code_support": False,
        "notes": "Transformer backbone; pending local weights and extractor support.",
    },
}


def cache_dirs() -> list[Path]:
    candidates = [
        ROOT / "artifacts/.torch/hub/checkpoints",
        Path.home() / ".cache/torch/hub/checkpoints",
    ]
    if os.environ.get("TORCH_HOME"):
        candidates.append(Path(os.environ["TORCH_HOME"]) / "hub/checkpoints")
    seen: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.append(path)
    return seen


def find_weight(expected_files: list[str]) -> Path | None:
    for directory in cache_dirs():
        for filename in expected_files:
            path = directory / filename
            if path.exists():
                return path
    return None


def inspect_backbones() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for backbone, spec in BACKBONES.items():
        path = find_weight(spec["expected_files"])
        rows.append(
            {
                "backbone": backbone,
                "source_type": "torchvision_cache" if path else "missing_local_weights",
                "local_path": str(path) if path else None,
                "runnable_now": bool(path and spec["code_support"]),
                "code_support_exists": bool(spec["code_support"]),
                "expected_files": spec["expected_files"],
                "notes": spec["notes"] if path or spec["code_support"] else f"{spec['notes']} Local weights not found.",
            }
        )
    cached = []
    for directory in cache_dirs():
        if directory.exists():
            cached.extend(str(path) for path in sorted(directory.glob("*.pth")))
    return {"cache_dirs": [str(path) for path in cache_dirs()], "backbones": rows, "cached_weight_files": cached}


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Backbone Readiness Report",
        "",
        "This report checks local pretrained weights only. It does not download models.",
        "",
        "| Backbone | Runnable Now | Code Support | Source Type | Local Path | Notes |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in payload["backbones"]:
        lines.append(
            f"| `{row['backbone']}` | `{row['runnable_now']}` | `{row['code_support_exists']}` | "
            f"{row['source_type']} | `{row['local_path'] or ''}` | {row['notes']} |"
        )
    lines.extend(["", "## Cached Weight Files", ""])
    if payload["cached_weight_files"]:
        lines.extend(f"- `{path}`" for path in payload["cached_weight_files"])
    else:
        lines.append("- none found")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check anomaly benchmark backbone readiness.")
    parser.add_argument("--json", default="reports/anomaly/backbone_readiness_report.json")
    parser.add_argument("--report", default="reports/anomaly/backbone_readiness_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = inspect_backbones()
    write_json(args.json, payload)
    write_markdown(payload, args.report)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
