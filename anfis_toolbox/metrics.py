"""Common metrics utilities for ANFIS Toolbox.

This module provides lightweight, dependency-free metrics that are useful
for training and evaluating ANFIS models.
"""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true, y_pred) -> float:
    """Compute the mean squared error (MSE).

    Parameters:
        y_true: Array-like of true target values, shape (...,)
        y_pred: Array-like of predicted values, same shape as y_true

    Returns:
        The mean of squared differences over all elements as a float.

    Notes:
        - Inputs are coerced to NumPy arrays with dtype=float.
        - Broadcasting follows NumPy semantics. If shapes are not compatible
          for element-wise subtraction, a ValueError will be raised by NumPy.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    diff = yt - yp
    return float(np.mean(diff * diff))


def mean_absolute_error(y_true, y_pred) -> float:
    """Compute the mean absolute error (MAE).

    Parameters:
        y_true: Array-like of true target values, shape (...,)
        y_pred: Array-like of predicted values, same shape as y_true

    Returns:
        The mean of absolute differences over all elements as a float.

    Notes:
        - Inputs are coerced to NumPy arrays with dtype=float.
        - Broadcasting follows NumPy semantics. If shapes are not compatible
          for element-wise subtraction, a ValueError will be raised by NumPy.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def root_mean_squared_error(y_true, y_pred) -> float:
    """Compute the root mean squared error (RMSE).

    This is simply the square root of mean_squared_error.
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def mean_absolute_percentage_error(y_true, y_pred, epsilon: float = 1e-12) -> float:
    """Compute the mean absolute percentage error (MAPE) in percent.

    MAPE = mean( abs((y_true - y_pred) / max(abs(y_true), epsilon)) ) * 100

    Parameters:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted values, broadcastable to y_true.
        epsilon: Small constant to avoid division by zero when y_true == 0.

    Returns:
        MAPE value as a percentage (float).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(yt), float(epsilon))
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon: float = 1e-12) -> float:
    """Compute the symmetric mean absolute percentage error (SMAPE) in percent.

    SMAPE = mean( 200 * |y_true - y_pred| / (|y_true| + |y_pred|) )
    with an epsilon added to denominator to avoid division by zero.

    Parameters:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted values, broadcastable to y_true.
        epsilon: Small constant added to denominator to avoid division by zero.

    Returns:
        SMAPE value as a percentage (float).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(yt) + np.abs(yp), float(epsilon))
    return float(np.mean(200.0 * np.abs(yt - yp) / denom))


def r2_score(y_true, y_pred, epsilon: float = 1e-12) -> float:
    """Compute the coefficient of determination R^2.

    R^2 = 1 - SS_res / SS_tot, where SS_res = sum((y - y_hat)^2)
    and SS_tot = sum((y - mean(y))^2). If SS_tot is ~0 (constant target),
    returns 1.0 when predictions match the constant target (SS_res ~0),
    otherwise 0.0.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    diff = yt - yp
    ss_res = float(np.sum(diff * diff))
    yt_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    if ss_tot <= float(epsilon):
        return 1.0 if ss_res <= float(epsilon) else 0.0
    return 1.0 - ss_res / ss_tot


def pearson_correlation(y_true, y_pred, epsilon: float = 1e-12) -> float:
    """Compute the Pearson correlation coefficient r.

    Returns 0.0 when the standard deviation of either input is ~0 (undefined r).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yt_centered = yt - np.mean(yt)
    yp_centered = yp - np.mean(yp)
    num = float(np.sum(yt_centered * yp_centered))
    den = float(np.sqrt(np.sum(yt_centered * yt_centered) * np.sum(yp_centered * yp_centered)))
    if den <= float(epsilon):
        return 0.0
    return num / den


def mean_squared_logarithmic_error(y_true, y_pred) -> float:
    """Compute the mean squared logarithmic error (MSLE).

    Requires non-negative inputs. Uses log1p for numerical stability:
    MSLE = mean( (log1p(y_true) - log1p(y_pred))^2 ).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if np.any(yt < 0) or np.any(yp < 0):
        raise ValueError("mean_squared_logarithmic_error requires non-negative y_true and y_pred")
    diff = np.log1p(yt) - np.log1p(yp)
    return float(np.mean(diff * diff))
