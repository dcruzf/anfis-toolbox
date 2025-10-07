"""Minimal reproduction of Example 1 from Jang (1993) with quick model checks.

The original ANFIS paper models the two-input nonlinear function::

    z(x, y) = sin(x) / x * sin(y) / y

We generate the same 11x11 grid of samples across [-10, 10]², train an
ANFIS model with four Gaussian membership functions per input (16 rules),
and run a short verification routine that mirrors the checklist discussed
in the documentation:

1. Hold out a validation split.
2. Evaluate with the built-in metrics helpers.
3. Inspect raw predictions and residual statistics.

Run the script with ``python examples/anfis_paper_example.py``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anfis_toolbox import quick_evaluate
from anfis_toolbox.metrics import MetricReport, compute_metrics
from anfis_toolbox.regressor import ANFISRegressor


@dataclass
class DatasetSplit:
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray


def sinc(x: np.ndarray) -> np.ndarray:
    """Vectorised sinc equal to sin(x)/x with the limit value 1 at x=0."""

    out = np.empty_like(x, dtype=float)
    mask = np.abs(x) > np.finfo(float).eps
    out[mask] = np.sin(x[mask]) / x[mask]
    out[~mask] = 1.0
    return out


def generate_anfis_example_data(
    grid_min: float = -10.0,
    grid_max: float = 10.0,
    points_per_axis: int = 11,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    """Build the grid data used in Example 1 and split into train/validation."""

    axis = np.linspace(grid_min, grid_max, points_per_axis)
    x1, x2 = np.meshgrid(axis, axis)
    X = np.column_stack([x1.ravel(), x2.ravel()])
    y = (sinc(x1) * sinc(x2)).ravel()

    rng = np.random.default_rng(seed)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    split_idx = int(np.floor((1.0 - validation_ratio) * indices.size))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    return DatasetSplit(
        train_X=X[train_idx],
        train_y=y[train_idx],
        val_X=X[val_idx],
        val_y=y[val_idx],
    )

def train_regressor(
    train_X: np.ndarray,
    train_y: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    random_state: int = 42,
    verbose: bool = False,
    margin: float = 0.0,
    n_mfs: int = 4,
    mf_type: str = "gaussian",
    shuffle: bool = False,
) -> tuple[ANFISRegressor, list[float]]:
    """Train an ANFIS regressor and return the fitted model plus MSE history."""

    regressor = ANFISRegressor(
        n_mfs=n_mfs,
        mf_type=mf_type,
        init="grid",
        overlap=0.5,
        margin=margin,
        learning_rate=learning_rate,
        epochs=epochs,
        shuffle=shuffle,
        verbose=verbose,
        random_state=random_state,
    )
    regressor.fit(train_X, train_y)
    history = regressor.training_history_ or {"train": []}
    train_losses = list(history.get("train", []))
    return regressor, train_losses


def train_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[ANFISRegressor, list[float]]:
    """Convenience wrapper used by the quick evaluation path."""

    return train_regressor(
        train_X,
        train_y,
        learning_rate=0.01,
        epochs=60,
        random_state=random_state,
        verbose=False,
    )


def run_quick_checks(split: DatasetSplit, model: ANFISRegressor) -> MetricReport:
    """Perform the quick evaluation workflow for regression."""

    print("\n=== Step 1: quick evaluation on validation split ===")
    quick_metrics = quick_evaluate(
        model, split.val_X, split.val_y, print_results=True, task="regression"
    )
    print(
        f"Quick check summary -> RMSE: {quick_metrics['rmse']:.6f}, "
        f"MAE: {quick_metrics['mae']:.6f}, R²: {quick_metrics['r2']:.4f}"
    )

    print("\n=== Step 2: compute_metrics for richer report ===")
    preds = model.predict(split.val_X).ravel()
    report = compute_metrics(split.val_y, y_pred=preds, task="regression")
    for key, value in report.to_dict().items():
        if isinstance(value, float):
            print(f"{key:>28s}: {value: .6f}")
    print("Top-level task:", report.task)

    print("\n=== Step 3: prediction samples & residuals ===")
    sample_rows = np.column_stack([split.val_X[:5], split.val_y[:5], preds[:5]])
    print(" x1    x2    target    prediction    residual")
    for row in sample_rows:
        residual = row[-1] - row[-2]
        print(
            f"{row[0]: .3f} {row[1]: .3f} {row[2]: .6f} {row[3]: .6f} {residual: .6f}"
        )
    residuals = preds - split.val_y
    print(
        f"Residual summary -> mean: {residuals.mean(): .6f}, "
        f"std: {residuals.std(): .6f}, max abs: {np.max(np.abs(residuals)): .6f}"
    )

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an ANFIS model on the Jang (1993) Example 1 dataset. "
            "Run in quick mode or sweep several learning rates/epochs to "
            "approximate the article's extended training curves."
        )
    )
    parser.add_argument(
        "--mode",
        choices={"quick", "sweep"},
        default="quick",
        help="Quick runs a single training pass; sweep runs multiple configurations.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of epochs for quick mode (default: 60).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for quick mode (default: 0.01).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Margin factor for the Gaussian membership function initialisation.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle data between epochs during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used for initialisation and sweep repetitions.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training progress from ANFISRegressor.",
    )
    parser.add_argument(
        "--sweep-learning-rates",
        type=float,
        nargs="+",
        default=[0.01, 0.005, 0.001],
        help="Learning rates to evaluate when running in sweep mode.",
    )
    parser.add_argument(
        "--sweep-epochs",
        type=int,
        default=150,
        help="Epoch count to use in sweep mode (default: 150).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="How many random seeds to try for each learning rate in sweep mode.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="After sweep, run quick evaluation on the best configuration.",
    )
    return parser.parse_args(argv)


def run_quick_mode(split: DatasetSplit, args: argparse.Namespace) -> None:
    regressor, losses = train_regressor(
        split.train_X,
        split.train_y,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        random_state=args.seed,
        margin=args.margin,
        shuffle=args.shuffle,
        verbose=args.verbose,
    )
    if losses:
        print(
            "Trained ANFISRegressor with 16 rules. Initial/Final train loss: "
            f"{losses[0]:.6f} -> {losses[-1]:.6f}"
        )
    else:
        print("Trained ANFISRegressor (no training loss history recorded).")
    run_quick_checks(split, regressor)


def run_sweep_mode(split: DatasetSplit, args: argparse.Namespace) -> None:
    print(
        "Running article-style sweep -> learning rates: "
        f"{args.sweep_learning_rates}, epochs: {args.sweep_epochs}, "
        f"repetitions: {args.repetitions}"
    )

    results: list[dict[str, float | int]] = []
    best_entry: dict[str, float | int] | None = None
    best_val_rmse = float("inf")

    for lr in args.sweep_learning_rates:
        for rep in range(args.repetitions):
            seed = args.seed + rep
            regressor, losses = train_regressor(
                split.train_X,
                split.train_y,
                learning_rate=lr,
                epochs=args.sweep_epochs,
                random_state=seed,
                margin=args.margin,
                shuffle=args.shuffle,
                verbose=args.verbose,
            )
            preds_train = regressor.predict(split.train_X).ravel()
            preds_val = regressor.predict(split.val_X).ravel()
            train_rmse = np.sqrt(np.mean((preds_train - split.train_y) ** 2))
            val_rmse = np.sqrt(np.mean((preds_val - split.val_y) ** 2))
            entry: dict[str, float | int] = {
                "learning_rate": lr,
                "seed": seed,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "final_loss": losses[-1] if losses else float("nan"),
            }
            results.append(entry)
            print(
                f"lr={lr:.4f} seed={seed} -> train RMSE: {train_rmse:.6f}, "
                f"val RMSE: {val_rmse:.6f}, final loss: {entry['final_loss']:.6f}"
            )
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_entry = entry

    if not results:
        print("No sweep results produced.")
        return

    train_rmses = np.array([item["train_rmse"] for item in results], dtype=float)
    val_rmses = np.array([item["val_rmse"] for item in results], dtype=float)

    print("\n=== Sweep summary (RMSE) ===")
    print(
        f"Train RMSE mean: {train_rmses.mean():.6f} ± {train_rmses.std():.6f}, "
        f"min/max: {train_rmses.min():.6f} / {train_rmses.max():.6f}"
    )
    print(
        f"Val   RMSE mean: {val_rmses.mean():.6f} ± {val_rmses.std():.6f}, "
        f"min/max: {val_rmses.min():.6f} / {val_rmses.max():.6f}"
    )

    if best_entry and args.report:
        print("\n=== Detailed report for best validation RMSE ===")
        print(
            f"Best config -> lr={best_entry['learning_rate']:.4f}, "
            f"seed={best_entry['seed']}, val RMSE={best_entry['val_rmse']:.6f}"
        )
        regressor, _ = train_regressor(
            split.train_X,
            split.train_y,
            learning_rate=float(best_entry["learning_rate"]),
            epochs=args.sweep_epochs,
            random_state=int(best_entry["seed"]),
            margin=args.margin,
            shuffle=args.shuffle,
            verbose=args.verbose,
        )
        run_quick_checks(split, regressor)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    split = generate_anfis_example_data()

    if args.mode == "quick":
        run_quick_mode(split, args)
    else:
        run_sweep_mode(split, args)


if __name__ == "__main__":
    main()
