"""Validation and evaluation utilities for ANFIS models."""

import logging
import time

import numpy as np

from .metrics import mean_absolute_error, mean_squared_error, r2_score
from .model import ANFIS
from .model_selection import KFold, train_test_split


class ANFISValidator:
    """Validation utilities for ANFIS models."""

    def __init__(self, model: ANFIS):
        """Initialize validator with ANFIS model."""
        self.model = model

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        epochs: int = 50,
        learning_rate: float = 0.01,
        method: str = "hybrid",
        *,
        n_mfs: int = 3,
        shuffle: bool = True,
        random_state: int | None = 42,
    ) -> dict[str, float]:
        """Perform cross-validation on the ANFIS model.

        Parameters:
            X: Input data
            y: Target values
            cv: Number of cross-validation folds
            epochs: Training epochs per fold
            learning_rate: Learning rate for training
            method: Training method ('hybrid' or 'backprop')
            n_mfs: Number of membership functions per input for fold models
            shuffle: Whether to shuffle data before splitting into folds
            random_state: Random seed for reproducibility when shuffling

        Returns:
            Dictionary with cross-validation metrics
        """
        kfold = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)

        fold_scores = {"mse": [], "mae": [], "r2": [], "training_time": []}

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logging.info("Training fold %d/%d...", fold + 1, cv)

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Reset model parameters (create new instance)
            # Import here to avoid circular imports
            from . import builders

            fold_model = builders.QuickANFIS.for_regression(X_train, n_mfs=n_mfs)

            # Train model
            start_time = time.time()
            if method.lower() == "hybrid":
                # Default fit uses HybridTrainer
                fold_model.fit(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=False)
            else:
                # Use SGDTrainer explicitly for backprop branch
                from .optim import SGDTrainer

                trainer = SGDTrainer(learning_rate=learning_rate, epochs=epochs, verbose=False)
                fold_model.fit(X_train, y_train, trainer=trainer)

            training_time = time.time() - start_time

            # Evaluate
            y_pred = fold_model.predict(X_val)

            fold_scores["mse"].append(mean_squared_error(y_val, y_pred))
            fold_scores["mae"].append(mean_absolute_error(y_val, y_pred))
            fold_scores["r2"].append(r2_score(y_val, y_pred))
            fold_scores["training_time"].append(training_time)

        # Calculate statistics
        results = {}
        for metric, values in fold_scores.items():
            results[f"{metric}_mean"] = np.mean(values)
            results[f"{metric}_std"] = np.std(values)

        return results

    def train_test_split_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        epochs: int = 50,
        learning_rate: float = 0.01,
        method: str = "hybrid",
        *,
        random_state: int | None = 42,
    ) -> dict:
        """Train model with train-test split and return detailed evaluation.

        Parameters:
            X: Input data
            y: Target values
            test_size: Proportion of data for testing
            epochs: Training epochs
            learning_rate: Learning rate
            method: Training method ('hybrid' or 'backprop')

        Returns:
            Dictionary with evaluation results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train model
        start_time = time.time()
        if method.lower() == "hybrid":
            # Default fit uses HybridTrainer
            losses = self.model.fit(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=False)
        else:
            from .optim import SGDTrainer

            trainer = SGDTrainer(learning_rate=learning_rate, epochs=epochs, verbose=False)
            losses = self.model.fit(X_train, y_train, trainer=trainer)

        training_time = time.time() - start_time

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        results = {
            "training_time": training_time,
            "final_loss": losses[-1] if losses else None,
            "train_metrics": {
                "mse": mean_squared_error(y_train, y_train_pred),
                "mae": mean_absolute_error(y_train, y_train_pred),
                "r2": r2_score(y_train, y_train_pred),
            },
            "test_metrics": {
                "mse": mean_squared_error(y_test, y_test_pred),
                "mae": mean_absolute_error(y_test, y_test_pred),
                "r2": r2_score(y_test, y_test_pred),
            },
            "data_splits": {"train_size": len(X_train), "test_size": len(X_test)},
            "losses": losses,
            "predictions": {
                "y_train_true": y_train,
                "y_train_pred": y_train_pred,
                "y_test_true": y_test,
                "y_test_pred": y_test_pred,
            },
        }

        return results

    def learning_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: list[float] = None,
        epochs: int = 50,
        learning_rate: float = 0.01,
        method: str = "hybrid",
        *,
        n_mfs: int = 3,
        random_state: int | None = 42,
    ) -> dict:
        """Generate learning curves for different training set sizes.

        Parameters:
            X: Input data
            y: Target values
            train_sizes: List of training set size fractions
            epochs: Training epochs
            learning_rate: Learning rate
            method: Training method

        Returns:
            Dictionary with learning curve data
        """
        if train_sizes is None:
            train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

        results = {"train_sizes": [], "train_scores": [], "val_scores": [], "training_times": []}

        # Use 80% of data for learning curve, 20% fixed for validation
        X_lc, X_val, y_lc, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

        for size in train_sizes:
            logging.info("Training with %.0f%% of training data...", size * 100)

            # Sample training data reproducibly
            n_samples = max(1, int(len(X_lc) * size))
            rng = np.random.RandomState(random_state)
            indices = rng.choice(len(X_lc), n_samples, replace=False)
            X_train_size = X_lc[indices]
            y_train_size = y_lc[indices]

            # Create new model instance
            # Import here to avoid circular imports
            from . import builders

            size_model = builders.QuickANFIS.for_regression(X_train_size, n_mfs=n_mfs)

            # Train
            start_time = time.time()
            if method.lower() == "hybrid":
                size_model.fit(X_train_size, y_train_size, epochs=epochs, learning_rate=learning_rate, verbose=False)
            else:
                from .optim import SGDTrainer

                trainer = SGDTrainer(learning_rate=learning_rate, epochs=epochs, verbose=False)
                size_model.fit(X_train_size, y_train_size, trainer=trainer)

            training_time = time.time() - start_time

            # Evaluate
            y_train_pred = size_model.predict(X_train_size)
            y_val_pred = size_model.predict(X_val)

            train_score = r2_score(y_train_size, y_train_pred)
            val_score = r2_score(y_val, y_val_pred)

            results["train_sizes"].append(n_samples)
            results["train_scores"].append(train_score)
            results["val_scores"].append(val_score)
            results["training_times"].append(training_time)

        return results


class ANFISMetrics:
    """Metrics calculator for ANFIS models."""

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate comprehensive regression metrics.

        Parameters:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary with metric values
        """
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": ANFISMetrics._mean_absolute_percentage_error(y_true, y_pred),
            "max_error": np.max(np.abs(y_true - y_pred)),
            "std_error": np.std(y_true - y_pred),
        }

    @staticmethod
    def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def model_complexity_metrics(model: ANFIS) -> dict[str, int]:
        """Calculate model complexity metrics.

        Parameters:
            model: ANFIS model

        Returns:
            Dictionary with complexity metrics
        """
        n_inputs = model.n_inputs
        n_rules = model.n_rules

        # Count total parameters
        n_premise_params = 0
        for _input_name, mfs in model.membership_layer.membership_functions.items():
            for mf in mfs:
                n_premise_params += len(mf.parameters)

        n_consequent_params = model.consequent_layer.parameters.size

        return {
            "n_inputs": n_inputs,
            "n_rules": n_rules,
            "n_premise_parameters": n_premise_params,
            "n_consequent_parameters": n_consequent_params,
            "total_parameters": n_premise_params + n_consequent_params,
        }


def quick_evaluate(
    model: ANFIS, X_test: np.ndarray, y_test: np.ndarray, print_results: bool = True
) -> dict[str, float]:
    """Quick evaluation function for ANFIS models.

    Parameters:
        model: Trained ANFIS model
        X_test: Test input data
        y_test: Test target values
        print_results: Whether to print results

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    metrics = ANFISMetrics.regression_metrics(y_test, y_pred)

    if print_results:
        print("=" * 50)  # noqa: T201
        print("ANFIS Model Evaluation Results")  # noqa: T201
        print("=" * 50)  # noqa: T201
        print(f"Mean Squared Error (MSE):     {metrics['mse']:.6f}")  # noqa: T201
        print(f"Root Mean Squared Error:      {metrics['rmse']:.6f}")  # noqa: T201
        print(f"Mean Absolute Error (MAE):    {metrics['mae']:.6f}")  # noqa: T201
        print(f"R-squared (R²):               {metrics['r2']:.4f}")  # noqa: T201
        print(f"Mean Abs. Percentage Error:   {metrics['mape']:.2f}%")  # noqa: T201
        print(f"Maximum Error:                {metrics['max_error']:.6f}")  # noqa: T201
        print(f"Standard Deviation of Error:  {metrics['std_error']:.6f}")  # noqa: T201
        print("=" * 50)  # noqa: T201
    else:  # pragma: no cover - covered by explicit test path without printing
        # No output requested; simply return the computed metrics
        pass

    return metrics
