"""Evaluation utilities for ANFIS models."""

import numpy as np

from .metrics import mean_absolute_error, mean_squared_error, r2_score
from .model import ANFIS


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
