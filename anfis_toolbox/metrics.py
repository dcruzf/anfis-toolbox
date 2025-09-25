"""Common metrics utilities for ANFIS Toolbox.

This module provides lightweight, dependency-free metrics that are useful
for training and evaluating ANFIS models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .model import ANFIS


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


# -----------------------------
# Classification metrics and helpers
# -----------------------------


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along a given axis."""
    z = np.asarray(logits, dtype=float)
    zmax = np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z - zmax)
    den = np.sum(ez, axis=axis, keepdims=True)
    return ez / den


def cross_entropy(y_true, logits: np.ndarray, epsilon: float = 1e-12) -> float:
    """Compute mean cross-entropy from integer labels or one-hot vs logits.

    Parameters:
        y_true: Array-like of shape (n_samples,) of integer class labels, or
                one-hot array of shape (n_samples, n_classes).
        logits: Array-like raw scores, shape (n_samples, n_classes).
        epsilon: Small constant for numerical stability.

    Returns:
        Mean cross-entropy (float).
    """
    logits = np.asarray(logits, dtype=float)
    n = logits.shape[0]
    if n == 0:
        return 0.0
    # Stable log-softmax
    zmax = np.max(logits, axis=1, keepdims=True)
    logsumexp = zmax + np.log(np.sum(np.exp(logits - zmax), axis=1, keepdims=True))
    log_probs = logits - logsumexp  # (n, k)

    yt = np.asarray(y_true)
    if yt.ndim == 1:
        # integer labels
        yt = yt.reshape(-1)
        if yt.shape[0] != n:
            raise ValueError("y_true length must match logits batch size")
        # pick log prob at true class
        idx = (np.arange(n), yt.astype(int))
        nll = -log_probs[idx]
    else:
        # one-hot
        if yt.shape != logits.shape:
            raise ValueError("For one-hot y_true, shape must match logits")
        nll = -np.sum(yt * log_probs, axis=1)
    return float(np.mean(nll))


def log_loss(y_true, y_prob: np.ndarray, epsilon: float = 1e-12) -> float:
    """Compute mean log loss from integer/one-hot labels and probabilities."""
    P = np.asarray(y_prob, dtype=float)
    P = np.clip(P, float(epsilon), 1.0)
    yt = np.asarray(y_true)
    n = P.shape[0]
    if yt.ndim == 1:
        idx = (np.arange(n), yt.astype(int))
        nll = -np.log(P[idx])
    else:
        if yt.shape != P.shape:
            raise ValueError("For one-hot y_true, shape must match probabilities")
        nll = -np.sum(yt * np.log(P), axis=1)
    return float(np.mean(nll))


def accuracy(y_true, y_pred) -> float:
    """Compute accuracy from integer/one-hot labels and logits/probabilities.

    y_pred can be class indices (n,), logits (n,k), or probabilities (n,k).
    y_true can be class indices (n,) or one-hot (n,k).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yp.ndim == 2:
        yp_cls = np.argmax(yp, axis=1)
    else:
        yp_cls = yp.reshape(-1).astype(int)
    if yt.ndim == 2:
        yt_cls = np.argmax(yt, axis=1)
    else:
        yt_cls = yt.reshape(-1).astype(int)
    if yt_cls.shape[0] != yp_cls.shape[0]:
        raise ValueError("y_true and y_pred must have same number of samples")
    return float(np.mean(yt_cls == yp_cls))


def partition_coefficient(U: np.ndarray) -> float:
    """Bezdek's Partition Coefficient (PC) in [1/k, 1]. Higher is crisper.

    Parameters:
        U: Membership matrix of shape (n_samples, n_clusters).

    Returns:
        PC value as float.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError("U must be a 2D membership matrix")
    n = U.shape[0]
    if n == 0:
        return 0.0
    return float(np.sum(U * U) / float(n))


def classification_entropy(U: np.ndarray, epsilon: float = 1e-12) -> float:
    """Classification Entropy (CE). Lower is better (crisper).

    Parameters:
        U: Membership matrix of shape (n_samples, n_clusters).
        epsilon: Small constant to avoid log(0).

    Returns:
        CE value as float.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError("U must be a 2D membership matrix")
    n = U.shape[0]
    if n == 0:
        return 0.0
    Uc = np.clip(U, float(epsilon), 1.0)
    return float(-np.sum(Uc * np.log(Uc)) / float(n))


def xie_beni_index(
    X: np.ndarray,
    U: np.ndarray,
    C: np.ndarray,
    m: float = 2.0,
    epsilon: float = 1e-12,
) -> float:
    """Xie–Beni index (XB). Lower is better.

    XB = sum_i sum_k u_ik^m ||x_i - v_k||^2 / (n * min_{p!=q} ||v_p - v_q||^2)

    Parameters:
        X: Data array, shape (n_samples, n_features) or (n_samples,).
        U: Membership matrix, shape (n_samples, n_clusters).
        C: Cluster centers, shape (n_clusters, n_features).
        m: Fuzzifier (>1).
        epsilon: Small constant to avoid division by zero.

    Returns:
        XB value as float (np.inf when centers < 2).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 1D or 2D array-like")
    U = np.asarray(U, dtype=float)
    C = np.asarray(C, dtype=float)
    if U.ndim != 2:
        raise ValueError("U must be a 2D membership matrix")
    if C.ndim != 2:
        raise ValueError("C must be a 2D centers matrix")
    if X.shape[0] != U.shape[0]:
        raise ValueError("X and U must have the same number of samples")
    if C.shape[1] != X.shape[1]:
        raise ValueError("C and X must have the same number of features")
    if C.shape[0] < 2:
        return float(np.inf)
    m = float(m)

    # distances (n,k)
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    num = float(np.sum((U**m) * d2))

    # min squared distance between distinct centers
    diffs = C[:, None, :] - C[None, :, :]
    dist2 = (diffs * diffs).sum(axis=2)
    k = C.shape[0]
    idx = np.arange(k)
    dist2[idx, idx] = np.inf
    den = float(np.min(dist2))
    den = max(den, float(epsilon))
    return num / (float(X.shape[0]) * den)


class ANFISMetrics:
    """Metrics calculator utilities for ANFIS models."""

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Return a suite of regression metrics for predictions vs. targets."""
        mse = mean_squared_error(y_true, y_pred)
        residuals = y_true - y_pred
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": ANFISMetrics._mean_absolute_percentage_error(y_true, y_pred),
            "max_error": float(np.max(np.abs(residuals))),
            "std_error": float(np.std(residuals)),
        }

    @staticmethod
    def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error expressed in percent, ignoring zero targets."""
        mask = y_true != 0
        if not np.any(mask):
            return float(np.inf)
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def model_complexity_metrics(model: ANFIS) -> dict[str, int]:
        """Compute structural statistics for an ANFIS model instance."""
        n_inputs = model.n_inputs
        n_rules = model.n_rules

        n_premise_params = 0
        for mfs in model.membership_layer.membership_functions.values():
            for mf in mfs:
                n_premise_params += len(mf.parameters)

        n_consequent_params = model.consequent_layer.parameters.size

        return {
            "n_inputs": n_inputs,
            "n_rules": n_rules,
            "n_premise_parameters": n_premise_params,
            "n_consequent_parameters": int(n_consequent_params),
            "total_parameters": n_premise_params + int(n_consequent_params),
        }


def quick_evaluate(
    model: ANFIS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    print_results: bool = True,
) -> dict[str, float]:
    """Evaluate a trained ANFIS model on test data and optionally print a summary."""
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

    return metrics
