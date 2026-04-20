"""Legacy baseline training namespace wrapper."""

from src.train_baseline import build_model, eval_epoch, main, train_epoch

__all__ = ["build_model", "eval_epoch", "main", "train_epoch"]


if __name__ == "__main__":
    main()
