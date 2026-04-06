"""Create a dataset manifest CSV with stratified splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {"good": 0, "defect": 1}


def parse_extensions(ext_str: str) -> Tuple[str, ...]:
    exts = []
    for item in ext_str.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        exts.append(item.lower())
    return tuple(sorted(set(exts)))


def ensure_split_ratios(split_str: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in split_str.split(",")]
    if len(parts) != 3:
        raise ValueError("Split must have three comma-separated values, e.g. 0.7,0.15,0.15")
    ratios = tuple(float(p) for p in parts)
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    return ratios  # type: ignore[return-value]


def collect_images(
    base_dir: Path,
    class_dir: Path,
    label_str: str,
    extensions: Tuple[str, ...],
    recursive: bool,
) -> List[Dict]:
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class directory: {class_dir}")
    pattern = "**/*" if recursive else "*"
    rows: List[Dict] = []
    for path in class_dir.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        rows.append(
            {
                "abs_path": path.resolve(),
                "label_str": label_str,
                "label": LABEL_MAP[label_str],
                "dataset_id": "D1",
            }
        )
    return rows


def to_rel_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def assign_splits(
    df: pd.DataFrame, split: Tuple[float, float, float], seed: int
) -> pd.DataFrame:
    train_ratio, val_ratio, test_ratio = split
    if len(df) < 3:
        raise ValueError("Not enough samples to create train/val/test splits.")
    labels = df["label"].values

    # First split train vs temp.
    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=val_ratio + test_ratio,
            stratify=labels,
            random_state=seed,
        )
        # Then split temp into val/test.
        temp_labels = temp_df["label"].values
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_labels,
            random_state=seed,
        )
    except ValueError as exc:
        print(f"[WARN] Stratified split failed: {exc}", file=sys.stderr)
        print("[WARN] Falling back to non-stratified split.", file=sys.stderr)
        train_df, temp_df = train_test_split(
            df, test_size=val_ratio + test_ratio, random_state=seed, shuffle=True
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed, shuffle=True
        )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    class_counts = df["label_str"].value_counts()
    split_counts = df.groupby(["split", "label_str"]).size().unstack(fill_value=0)

    print(f"Total images: {total}")
    print("Class counts:")
    for label, count in class_counts.items():
        print(f"  {label}: {count}")
    print("Split counts per class:")
    print(split_counts.to_string())


def build_manifest(
    dataset_root: Path,
    dataset_id: str,
    good_dir: str,
    defect_dir: str,
    extensions: Tuple[str, ...],
    recursive: bool,
    split: Tuple[float, float, float],
    seed: int,
    repo_root: Path,
) -> pd.DataFrame:
    good_path = dataset_root / good_dir
    defect_path = dataset_root / defect_dir

    rows = []
    rows.extend(collect_images(dataset_root, good_path, "good", extensions, recursive))
    rows.extend(collect_images(dataset_root, defect_path, "defect", extensions, recursive))

    if not rows:
        raise ValueError("No images found. Check dataset_root and class directories.")

    df = pd.DataFrame(rows)
    df["dataset_id"] = dataset_id
    df["image_path"] = df["abs_path"].apply(lambda p: to_rel_path(Path(p), repo_root))
    df = df.drop(columns=["abs_path"])

    df = assign_splits(df, split=split, seed=seed)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset manifest CSV with splits.")
    parser.add_argument("--dataset_root", default="data/raw", help="Dataset root")
    parser.add_argument("--dataset_id", default="D1", help="Dataset identifier")
    parser.add_argument("--good_dir", default="good", help="Good class folder name")
    parser.add_argument("--defect_dir", default="defect", help="Defect class folder name")
    parser.add_argument("--out_csv", default="data/processed/D1_manifest.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", default="0.7,0.15,0.15", help="Split ratios train,val,test")
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Comma-separated list of extensions",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan class folders",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root
    output_path = Path(args.out_csv)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    extensions = parse_extensions(args.extensions)
    split = ensure_split_ratios(args.split)

    try:
        df = build_manifest(
            dataset_root=dataset_root,
            dataset_id=args.dataset_id,
            good_dir=args.good_dir,
            defect_dir=args.defect_dir,
            extensions=extensions,
            recursive=args.recursive,
            split=split,
            seed=args.seed,
            repo_root=repo_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved manifest to {output_path}")
    print_summary(df)


if __name__ == "__main__":
    main()
