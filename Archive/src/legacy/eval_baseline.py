"""Legacy baseline evaluation namespace wrapper."""

from src.eval_baseline import (
    build_model,
    evaluate,
    export_misclassification_artifacts,
    infer_model_type,
    main,
)

__all__ = [
    "build_model",
    "evaluate",
    "export_misclassification_artifacts",
    "infer_model_type",
    "main",
]


if __name__ == "__main__":
    main()
