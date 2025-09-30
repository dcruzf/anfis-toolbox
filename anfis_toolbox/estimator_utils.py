"""Lightweight utilities for scikit-learn style estimators without external dependencies.

This module provides a minimal subset of the scikit-learn estimator contract so that
high-level ANFIS interfaces can expose familiar methods (`fit`, `predict`,
`get_params`, `set_params`, etc.) without requiring scikit-learn as a runtime
dependency. The helpers here intentionally implement only the pieces we need
and keep them Numpy-centric for portability.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "BaseEstimatorLike",
    "RegressorMixinLike",
    "ClassifierMixinLike",
    "FittedMixin",
    "NotFittedError",
    "check_is_fitted",
    "infer_feature_names",
]


class NotFittedError(RuntimeError):
    """Exception raised when an estimator is used before fitting."""


class BaseEstimatorLike:
    """Mixin implementing scikit-learn style parameter inspection.

    Parameters are assumed to live on the instance `__dict__` and be declared in
    `__init__`. This matches the common sklearn design pattern and enables
    cloning/grid-search like workflows without relying on sklearn itself.
    """

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters following sklearn conventions."""

        def clone_param(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: clone_param(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(clone_param(v) for v in value)
            # Primitive / numpy scalars
            if isinstance(value, (str, int, float, bool, type(None), np.generic)):
                return value
            # Fallback to deepcopy for custom objects
            return deepcopy(value)

        return {key: clone_param(value) for key, value in self.__dict__.items() if not key.endswith("_")}

    def set_params(self, **params: Any) -> BaseEstimatorLike:
        """Set estimator parameters and return self."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for {type(self).__name__}.")
            setattr(self, key, value)
        return self


class FittedMixin:
    """Mixin providing utility to guard against using estimators pre-fit."""

    def _mark_fitted(self):
        self.is_fitted_ = True

    def _require_is_fitted(self, attributes: Iterable[str] | None = None):
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(f"{type(self).__name__} instance is not fitted yet.")
        if attributes:
            missing = [attr for attr in attributes if not hasattr(self, attr)]
            if missing:
                raise NotFittedError(
                    f"Estimator {type(self).__name__} is missing fitted attribute(s): {', '.join(missing)}"
                )


def check_is_fitted(estimator: FittedMixin, attributes: Iterable[str] | None = None):
    """Check if the estimator is fitted by verifying `is_fitted_` and optional attributes."""
    estimator._require_is_fitted(attributes)


class RegressorMixinLike:
    """Mixin implementing a default `score` method for regressors."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        y_true = np.asarray(y, dtype=float).reshape(-1)
        y_pred = np.asarray(self.predict(X), dtype=float).reshape(-1)
        if y_true.shape != y_pred.shape:
            raise ValueError("Predicted values have a different shape than y.")
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        y_mean = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - y_mean) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - ss_res / ss_tot


class ClassifierMixinLike:
    """Mixin implementing default `score` via simple accuracy."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given test data and labels."""
        y_true = np.asarray(y)
        y_pred = np.asarray(self.predict(X))
        if y_true.shape != y_pred.shape:
            raise ValueError("Predicted values have a different shape than y.")
        if y_true.size == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))


@dataclass
class ValidationResult:
    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]


def _ensure_2d_array(X: Any) -> tuple[np.ndarray, list[str]]:
    """Validate and convert input data to a 2D float64 numpy array."""
    if hasattr(X, "to_numpy"):
        values = X.to_numpy(dtype=float)
        names = getattr(X, "columns", None)
        feature_names = [str(col) for col in names] if names is not None else None
    else:
        values = np.asarray(X, dtype=float)
        feature_names = None

    if values.ndim != 2:
        raise ValueError("Input data must be 2-dimensional (n_samples, n_features).")
    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(values.shape[1])]

    return values, feature_names


def _ensure_vector(y: Any, *, allow_2d_column: bool = True) -> np.ndarray:
    array = np.asarray(y)
    if array.ndim == 2:
        if array.shape[1] == 1 and allow_2d_column:
            array = array.reshape(-1)
        else:
            raise ValueError("Target array must be 1-dimensional or a column vector.")
    elif array.ndim != 1:
        raise ValueError("Target array must be 1-dimensional.")
    return array


def infer_feature_names(X: Any) -> list[str]:
    """Return feature names inferred from the input data structure."""
    if hasattr(X, "columns"):
        return [str(col) for col in X.columns]
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("Expected 2D array-like input to infer feature names.")
    return [f"x{i + 1}" for i in range(X_arr.shape[1])]
