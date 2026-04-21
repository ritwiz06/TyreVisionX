"""Create good-only and mixed manifests for anomaly detection.

The canonical anomaly manifests are derived from a supervised manifest with
``good = 0`` and ``defect = 1`` labels. The output convention is generic:
``target = 0`` means normal and ``target = 1`` means anomaly.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_COLUMNS = {"image_path", "label", "split"}


@dataclass
class ManifestOutputs:
    normal_train: Path
    val_mixed: Path
    test_mixed: Path
    report: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_repo_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _candidate_paths(path_str: str, repo_root: Path, roots: Iterable[Path]) -> list[Path]:
    path = Path(path_str)
    if path.is_absolute():
        return [path]
    candidates = [repo_root / path]
    for root in roots:
        candidates.append(root / path)
    return candidates


def _resolve_image_path(path_str: str, repo_root: Path, roots: Iterable[Path]) -> tuple[str, bool]:
    for candidate in _candidate_paths(path_str, repo_root, roots):
        if candidate.exists():
            return _as_repo_relative(candidate, repo_root), True
    return path_str, False


def _normalize_source_manifest(
    df: pd.DataFrame,
    repo_root: Path,
    image_roots: list[Path],
    product_type: str,
) -> tuple[pd.DataFrame, int]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Source manifest missing required columns: {sorted(missing)}")

    out = df.copy()
    if "dataset_id" not in out.columns:
        out["dataset_id"] = "unknown"
    if "label_str" not in out.columns:
        out["label_str"] = out["label"].map({0: "good", 1: "defect"}).fillna("unknown")

    resolved_paths = []
    missing_count = 0
    for path_str in out["image_path"].astype(str):
        resolved, exists = _resolve_image_path(path_str, repo_root, image_roots)
        resolved_paths.append(resolved)
        if not exists:
            missing_count += 1

    out["image_path"] = resolved_paths
    out["target"] = out["label"].astype(int).map({0: 0, 1: 1})
    if out["target"].isna().any():
        bad = sorted(out.loc[out["target"].isna(), "label"].unique().tolist())
        raise ValueError(f"Unexpected labels in source manifest: {bad}")

    out["target"] = out["target"].astype(int)
    out["is_normal"] = out["target"] == 0
    out["source_dataset"] = out["dataset_id"].astype(str)
    out["product_type"] = product_type
    out["original_split"] = out["split"].astype(str)

    columns = [
        "image_path",
        "target",
        "label",
        "label_str",
        "split",
        "is_normal",
        "source_dataset",
        "product_type",
        "dataset_id",
        "original_split",
    ]
    return out[columns], missing_count


def create_anomaly_manifests(
    source_manifest: Path,
    output_dir: Path,
    dataset_id: str = "D1",
    product_type: str = "tyre",
    image_roots: list[Path] | None = None,
    dry_run: bool = False,
    report_path: Path | None = None,
) -> ManifestOutputs:
    repo_root = _repo_root()
    output_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir
    source_manifest = source_manifest if source_manifest.is_absolute() else repo_root / source_manifest
    image_roots = image_roots or []
    resolved_roots = [root if root.is_absolute() else repo_root / root for root in image_roots]

    if not source_manifest.exists():
        raise FileNotFoundError(
            f"Source manifest not found: {source_manifest}. "
            "Create the canonical supervised manifest under data/manifests/ first."
        )

    df = pd.read_csv(source_manifest)
    normalized, missing_count = _normalize_source_manifest(
        df=df,
        repo_root=repo_root,
        image_roots=resolved_roots,
        product_type=product_type,
    )

    normal_train = normalized[(normalized["split"] == "train") & (normalized["target"] == 0)].copy()
    val_mixed = normalized[normalized["split"] == "val"].copy()
    test_mixed = normalized[normalized["split"] == "test"].copy()

    for name, frame in {
        "normal_train": normal_train,
        "val_mixed": val_mixed,
        "test_mixed": test_mixed,
    }.items():
        if frame.empty:
            raise ValueError(f"Derived anomaly manifest '{name}' is empty. Check source split/label columns.")

    outputs = ManifestOutputs(
        normal_train=output_dir / f"{dataset_id}_anomaly_train_normal.csv",
        val_mixed=output_dir / f"{dataset_id}_anomaly_val_mixed.csv",
        test_mixed=output_dir / f"{dataset_id}_anomaly_test_mixed.csv",
        report=(report_path if report_path else repo_root / "reports/anomaly/anomaly_manifest_report.md"),
    )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs.report.parent.mkdir(parents=True, exist_ok=True)
        normal_train.to_csv(outputs.normal_train, index=False)
        val_mixed.to_csv(outputs.val_mixed, index=False)
        test_mixed.to_csv(outputs.test_mixed, index=False)
        outputs.report.write_text(
            build_report(
                source_manifest=source_manifest,
                outputs=outputs,
                frames={
                    "normal_train": normal_train,
                    "val_mixed": val_mixed,
                    "test_mixed": test_mixed,
                },
                missing_count=missing_count,
                dry_run=False,
            ),
            encoding="utf-8",
        )
    else:
        print(
            build_report(
                source_manifest=source_manifest,
                outputs=outputs,
                frames={
                    "normal_train": normal_train,
                    "val_mixed": val_mixed,
                    "test_mixed": test_mixed,
                },
                missing_count=missing_count,
                dry_run=True,
            )
        )

    return outputs


def _count_table(frame: pd.DataFrame) -> str:
    counts = frame["target"].map({0: "normal", 1: "anomaly"}).value_counts().reindex(["normal", "anomaly"], fill_value=0)
    return f"| normal | {int(counts['normal'])} |\n| anomaly | {int(counts['anomaly'])} |"


def build_report(
    source_manifest: Path,
    outputs: ManifestOutputs,
    frames: dict[str, pd.DataFrame],
    missing_count: int,
    dry_run: bool,
) -> str:
    lines = [
        "# Anomaly Manifest Report",
        "",
        f"Status: {'dry run only' if dry_run else 'generated'}",
        "",
        f"Source manifest: `{source_manifest}`",
        "",
        "Canonical decision: anomaly manifests are written under `data/manifests/`.",
        "`data/processed/` remains legacy compatibility only.",
        "",
        f"Image paths unresolved during generation: `{missing_count}`",
        "",
    ]
    for name, frame in frames.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"Rows: `{len(frame)}`",
                "",
                "| Target | Count |",
                "|---|---:|",
                _count_table(frame),
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            f"- normal train: `{outputs.normal_train}`",
            f"- validation mixed: `{outputs.val_mixed}`",
            f"- test mixed: `{outputs.test_mixed}`",
            "",
            "## Data Contract",
            "",
            "Required columns include `image_path`, `target`, `split`, `is_normal`, `source_dataset`, and `product_type`.",
            "`target = 0` means normal/good and `target = 1` means anomaly/defect.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create anomaly manifests from a supervised manifest.")
    parser.add_argument("--source_manifest", default="data/manifests/D1_tyrenet_manifest.csv")
    parser.add_argument("--output_dir", default="data/manifests")
    parser.add_argument("--dataset_id", default="D1")
    parser.add_argument("--product_type", default="tyre")
    parser.add_argument(
        "--image_root",
        action="append",
        default=["data/raw", "data/D1_tyrenet"],
        help="Candidate root for dataset-relative image paths. May be passed multiple times.",
    )
    parser.add_argument("--report", default="reports/anomaly/anomaly_manifest_report.md")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = create_anomaly_manifests(
        source_manifest=Path(args.source_manifest),
        output_dir=Path(args.output_dir),
        dataset_id=args.dataset_id,
        product_type=args.product_type,
        image_roots=[Path(p) for p in args.image_root],
        dry_run=args.dry_run,
        report_path=Path(args.report),
    )
    if not args.dry_run:
        print(f"Wrote anomaly manifests to {outputs.normal_train.parent}")
        print(f"Wrote report to {outputs.report}")


if __name__ == "__main__":
    main()
