"""Simple CLI entry point for TyreVisionX."""
from __future__ import annotations

import argparse

from src import evaluate, export, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TyreVisionX CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Run training")
    train_p.add_argument("--config", default="configs/train/train_resnet18.yaml", help="Config YAML")

    eval_p = subparsers.add_parser("eval", help="Run evaluation")
    eval_p.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    eval_p.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    eval_p.add_argument("--report", default=None, help="Optional output JSON path")

    export_p = subparsers.add_parser("export", help="Export TorchScript/ONNX")
    export_p.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    export_p.add_argument("--outdir", default=None, help="Output directory")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train.main(args.config)
    elif args.command == "eval":
        evaluate.main(args.checkpoint, split=args.split, report_path=args.report)
    elif args.command == "export":
        export.main(args.checkpoint, outdir=args.outdir)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
