"""Helper script with download guidance for TyreVisionX datasets."""
from __future__ import annotations

import os
from pathlib import Path


def main():
    data_root = Path(os.getenv("DATA_ROOT", "./data")).resolve()
    print(f"Data root: {data_root}")
    print("\nD1: TyreNet (public)")
    print("- Download manually from the official source (search 'TyreNet dataset').")
    print("- Place extracted images under data/D1_tyrenet.")

    print("\nD2: Kaggle tire crack dataset")
    print("- Requires Kaggle CLI and API token (~/.kaggle/kaggle.json).")
    print("- Command: kaggle datasets download -d prajwalkatke/tire-crack-detection -p data/D2_tire_crack")
    print("- Unzip the archive inside that folder.")

    print("\nD3: Kaggle tyre quality dataset")
    print("- Command: kaggle datasets download -d ritzg42/tyre-quality-image-dataset -p data/D3_tyre_quality")
    print("- Unzip the archive inside that folder.")

    print("\nAfter placing datasets, run `make manifests` then `make folds`.")


if __name__ == "__main__":
    main()
