"""Common metrics utilities for ANFIS Toolbox.

This module provides lightweight, dependency-free metrics that are useful
for training and evaluating ANFIS models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .model import TSKANFIS as ANFIS


ArrayLike: TypeAlias = npt.ArrayLike
MetricValue: TypeAlias = float | np.ndarray
MetricFn: TypeAlias = Callable[[np.ndarray, np.ndarray], float]

_EPSILON: float = 1e-12


@runtime_checkable
class _PredictorLike(Protocol):
    """Minimal protocol for objects exposing a ``predict`` method."""

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - typing helper
        """Return predictions for the provided samples."""


def _to_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _coerce_regression_targets(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    try:
        yt_b, yp_b = np.broadcast_arrays(yt, yp)
    except ValueError as exc:  # pragma: no cover - exercised via callers
        raise ValueError("regression targets must be broadcastable to the same shape") from exc
    return yt_b.reshape(-1), yp_b.reshape(-1)


def _flatten_float(values: ArrayLike) -> np.ndarray:
    return _to_float_array(values).reshape(-1)


def _coerce_labels(y_true: ArrayLike) -> np.ndarray:
    labels = np.asarray(y_true)
    if labels.ndim == 0:
        return labels.reshape(1).astype(int)
    if labels.ndim == 2:
        return np.argmax(labels, axis=1).astype(int)
    return labels.reshape(-1).astype(int)


def _ensure_probabilities(y_prob: ArrayLike) -> np.ndarray:
    proba = _to_float_array(y_prob)
    if proba.ndim != 2:
        raise ValueError("Probabilities must be a 2D array with shape (n_samples, n_classes)")
    row_sums = np.sum(proba, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Each probability row must have positive sum")
    proba = proba / row_sums
    proba = np.clip(proba, _EPSILON, 1.0)
    proba = proba / np.sum(proba, axis=1, keepdims=True)
    return proba


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    index = {label: idx for idx, label in enumerate(classes)}
    matrix = np.zeros((classes.size, classes.size), dtype=int)
    for yt, yp in zip(y_true, y_pred, strict=False):
        matrix[index[yt], index[yp]] += 1
    return matrix, classes


def _safe_divide(num: float, den: float) -> float:
    return num / den if den > 0.0 else 0.0


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
    yt, yp = _coerce_regression_targets(y_true, y_pred)
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
    yt, yp = _coerce_regression_targets(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def root_mean_squared_error(y_true, y_pred) -> float:
    """Compute the root mean squared error (RMSE).

    This is simply the square root of mean_squared_error.
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def mean_absolute_percentage_error(
    y_true,
    y_pred,
    epsilon: float = 1e-12,
    *,
    ignore_zero_targets: bool = False,
) -> float:
    """Compute the mean absolute percentage error (MAPE) in percent.

    MAPE = mean( abs((y_true - y_pred) / max(abs(y_true), epsilon)) ) * 100

    Parameters:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted values, broadcastable to y_true.
        epsilon: Small constant to avoid division by zero when y_true == 0.
        ignore_zero_targets: When True, drop samples where |y_true| <= epsilon; if all
            targets are (near) zero, returns ``np.inf`` to signal undefined percentage.

    Returns:
        MAPE value as a percentage (float).
    """
    yt, yp = _coerce_regression_targets(y_true, y_pred)
    if ignore_zero_targets:
        mask = np.abs(yt) > float(epsilon)
        if not np.any(mask):
            return float(np.inf)
        yt = yt[mask]
        yp = yp[mask]
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
    yt, yp = _coerce_regression_targets(y_true, y_pred)
    denom = np.maximum(np.abs(yt) + np.abs(yp), float(epsilon))
    return float(np.mean(200.0 * np.abs(yt - yp) / denom))


def r2_score(y_true, y_pred, epsilon: float = 1e-12) -> float:
    """Compute the coefficient of determination R^2.

    R^2 = 1 - SS_res / SS_tot, where SS_res = sum((y - y_hat)^2)
    and SS_tot = sum((y - mean(y))^2). If SS_tot is ~0 (constant target),
    returns 1.0 when predictions match the constant target (SS_res ~0),
    otherwise 0.0.
    """
    yt, yp = _coerce_regression_targets(y_true, y_pred)
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
    yt, yp = _coerce_regression_targets(y_true, y_pred)
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
    yt, yp = _coerce_regression_targets(y_true, y_pred)
    if np.any(yt < 0) or np.any(yp < 0):
        raise ValueError("mean_squared_logarithmic_error requires non-negative y_true and y_pred")
    diff = np.log1p(yt) - np.log1p(yp)
    return float(np.mean(diff * diff))


def explained_variance_score(y_true, y_pred, epsilon: float = _EPSILON) -> float:
    """Compute the explained variance score for regression predictions."""
    yt, yp = _coerce_regression_targets(y_true, y_pred)
    diff = yt - yp
    var_true = float(np.var(yt))
    var_residual = float(np.var(diff))
    if var_true <= float(epsilon):
        return 1.0 if var_residual <= float(epsilon) else 0.0
    return 1.0 - var_residual / var_true


def median_absolute_error(y_true, y_pred) -> float:
    """Return the median absolute deviation between predictions and targets."""
    yt, yp = _coerce_regression_targets(y_true, y_pred)
    return float(np.median(np.abs(yt - yp)))


def mean_bias_error(y_true, y_pred) -> float:
    """Compute the mean signed error, positive when predictions overshoot."""
    yt, yp = _coerce_regression_targets(y_true, y_pred)
    return float(np.mean(yp - yt))


def balanced_accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the macro-average recall, balancing performance across classes."""
    labels_true = _coerce_labels(y_true)
    labels_pred = _coerce_labels(y_pred)
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")
    matrix, _ = _confusion_matrix(labels_true, labels_pred)
    recalls = []
    for idx in range(matrix.shape[0]):
        tp = float(matrix[idx, idx])
        fn = float(np.sum(matrix[idx, :]) - tp)
        recalls.append(_safe_divide(tp, tp + fn))
    return float(np.mean(recalls)) if recalls else 0.0


def precision_recall_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: Literal["macro", "micro", "binary"] = "macro",
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 score with the requested averaging."""
    labels_true = _coerce_labels(y_true)
    labels_pred = _coerce_labels(y_pred)
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")
    matrix, classes = _confusion_matrix(labels_true, labels_pred)
    if average == "micro":
        tp = float(np.trace(matrix))
        fp = float(np.sum(np.sum(matrix, axis=0) - np.diag(matrix)))
        fn = float(np.sum(np.sum(matrix, axis=1) - np.diag(matrix)))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        return precision, recall, f1

    per_class_precision: list[float] = []
    per_class_recall: list[float] = []
    for idx, _ in enumerate(classes):
        tp = float(matrix[idx, idx])
        fp = float(np.sum(matrix[:, idx]) - tp)
        fn = float(np.sum(matrix[idx, :]) - tp)
        prec = _safe_divide(tp, tp + fp)
        rec = _safe_divide(tp, tp + fn)
        per_class_precision.append(prec)
        per_class_recall.append(rec)

    if average == "binary":
        if len(per_class_precision) != 2:
            raise ValueError("average='binary' is only defined for binary classification")
        precision = per_class_precision[1]
        recall = per_class_recall[1]
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        return precision, recall, f1

    precision = float(np.mean(per_class_precision)) if per_class_precision else 0.0
    recall = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


# -----------------------------
# Classification metrics and helpers
# -----------------------------


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along a given axis."""
    z = _to_float_array(logits)
    zmax = np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z - zmax)
    den = np.sum(ez, axis=axis, keepdims=True)
    den = np.clip(den, _EPSILON, None)
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
    logits = _to_float_array(logits)
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
    P = _to_float_array(y_prob)
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
    yt_labels = _coerce_labels(y_true)
    yp_arr = np.asarray(y_pred)
    if yp_arr.ndim == 2:
        yp_labels = np.argmax(yp_arr, axis=1)
    else:
        yp_labels = yp_arr.reshape(-1).astype(int)
    if yt_labels.shape[0] != yp_labels.shape[0]:
        raise ValueError("y_true and y_pred must have same number of samples")
    return float(np.mean(yt_labels == yp_labels))


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
    """Xie-Beni index (XB). Lower is better.

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


def _regression_metrics_dict(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    yt, yp = _coerce_regression_targets(y_true, y_pred)
    residuals = yt - yp
    mse = float(np.mean(residuals * residuals)) if residuals.size else 0.0
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals))) if residuals.size else 0.0
    median_ae = float(np.median(np.abs(residuals))) if residuals.size else 0.0
    mean_bias = float(np.mean(yp - yt)) if residuals.size else 0.0
    max_error = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    std_error = float(np.std(residuals)) if residuals.size else 0.0
    explained_var = explained_variance_score(yt, yp)
    r2 = r2_score(yt, yp)
    mape = mean_absolute_percentage_error(yt, yp, ignore_zero_targets=True)
    smape = symmetric_mean_absolute_percentage_error(yt, yp)
    try:
        msle = mean_squared_logarithmic_error(yt, yp)
    except ValueError:
        msle = float(np.nan)
    pearson = pearson_correlation(yt, yp)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "median_absolute_error": median_ae,
        "mean_bias_error": mean_bias,
        "max_error": max_error,
        "std_error": std_error,
        "explained_variance": explained_var,
        "r2": r2,
        "mape": mape,
        "smape": smape,
        "msle": msle,
        "pearson": pearson,
    }


def _classification_metrics_dict(
    y_true: ArrayLike,
    y_pred_labels: ArrayLike,
    probabilities: np.ndarray | None,
) -> dict[str, MetricValue]:
    labels_true = _coerce_labels(y_true)
    labels_pred = _coerce_labels(y_pred_labels)
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")

    matrix, classes = _confusion_matrix(labels_true, labels_pred)
    accuracy_val = float(np.mean(labels_true == labels_pred)) if labels_true.size else 0.0
    bal_acc = balanced_accuracy_score(labels_true, labels_pred)
    prec_macro, rec_macro, f1_macro = precision_recall_f1(labels_true, labels_pred, average="macro")
    prec_micro, rec_micro, f1_micro = precision_recall_f1(labels_true, labels_pred, average="micro")

    values: dict[str, MetricValue] = {
        "accuracy": accuracy_val,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
        "confusion_matrix": matrix,
        "classes": classes,
    }

    if probabilities is not None:
        values["log_loss"] = log_loss(labels_true, probabilities)
    else:
        values["log_loss"] = float("nan")

    return values


@dataclass(frozen=True)
class MetricReport:
    """Immutable container exposing computed metrics by key or attribute."""

    task: Literal["regression", "classification"]
    _values: Mapping[str, MetricValue]

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        """Sanitize stored NumPy scalars/arrays to prevent accidental mutation."""
        sanitized: dict[str, MetricValue] = {}
        for key, value in self._values.items():
            if isinstance(value, np.ndarray):
                sanitized[key] = value.copy()
            elif isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = value
        object.__setattr__(self, "_values", sanitized)

    def to_dict(self) -> dict[str, MetricValue]:
        """Return a shallow copy of the underlying metric mapping."""
        return {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in self._values.items()}

    def __getitem__(self, key: str) -> MetricValue:
        """Provide dictionary-style access to metric values."""
        return self._values[key]

    def __getattr__(self, item: str) -> MetricValue:
        """Allow attribute-style access to stored metrics."""
        try:
            return self._values[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def keys(self):  # pragma: no cover - simple passthrough
        """Expose the metric key iterator from the backing mapping."""
        return self._values.keys()


def compute_metrics(
    y_true: ArrayLike,
    *,
    y_pred: ArrayLike | None = None,
    y_proba: ArrayLike | None = None,
    logits: ArrayLike | None = None,
    task: Literal["auto", "regression", "classification"] = "auto",
    metrics: Sequence[str] | None = None,
    custom_metrics: Mapping[str, MetricFn] | None = None,
) -> MetricReport:
    """Compute regression or classification metrics and return a report."""
    resolved_task: Literal["regression", "classification"]

    if task == "regression":
        resolved_task = "regression"
    elif task == "classification":
        resolved_task = "classification"
    else:
        arr_pred = None if y_pred is None else np.asarray(y_pred)
        if y_proba is not None or logits is not None:
            resolved_task = "classification"
        elif arr_pred is not None and arr_pred.ndim == 2:
            resolved_task = "classification"
        elif arr_pred is not None and arr_pred.ndim == 1 and np.issubdtype(arr_pred.dtype, np.integer):
            resolved_task = "classification"
        else:
            resolved_task = "regression"

    values: dict[str, MetricValue]

    if resolved_task == "regression":
        if y_pred is None:
            raise ValueError("Regression metrics require 'y_pred'.")
        values = _regression_metrics_dict(y_true, y_pred)
        if custom_metrics:
            yt_arr, yp_arr = _coerce_regression_targets(y_true, y_pred)
            for name, fn in custom_metrics.items():
                values[name] = float(fn(yt_arr, yp_arr))
    else:
        probabilities: np.ndarray | None = None
        if logits is not None:
            probabilities = softmax(_to_float_array(logits), axis=1)
        elif y_proba is not None:
            probabilities = _ensure_probabilities(y_proba)

        if y_pred is not None:
            pred_labels = y_pred
        elif probabilities is not None:
            pred_labels = np.argmax(probabilities, axis=1)
        else:
            raise ValueError("Classification metrics require 'y_pred', 'y_proba', or 'logits'.")

        values = _classification_metrics_dict(y_true, pred_labels, probabilities)

        if custom_metrics:
            labels_true = _coerce_labels(y_true)
            labels_pred = _coerce_labels(pred_labels)
            for name, fn in custom_metrics.items():
                values[name] = float(fn(labels_true, labels_pred))

    if metrics is not None:
        missing = [name for name in metrics if name not in values]
        if missing:
            raise KeyError(f"Requested metric(s) not available: {', '.join(missing)}")
        values = {name: values[name] for name in metrics}

    return MetricReport(task=resolved_task, _values=values)


class ANFISMetrics:
    """Metrics calculator utilities for ANFIS models."""

    @staticmethod
    def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, MetricValue]:
        """Return a suite of regression metrics for predictions vs. targets."""
        report = compute_metrics(y_true, y_pred=y_pred, task="regression")
        return report.to_dict()

    @staticmethod
    def classification_metrics(
        y_true: ArrayLike,
        y_pred: ArrayLike | None = None,
        *,
        y_proba: ArrayLike | None = None,
        logits: ArrayLike | None = None,
    ) -> dict[str, MetricValue]:
        """Return common classification metrics for encoded targets and predictions."""
        report = compute_metrics(
            y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            logits=logits,
            task="classification",
        )
        return report.to_dict()

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


def _resolve_predictor(model: object) -> _PredictorLike:
    """Return an object exposing ``predict`` for use in :func:`quick_evaluate`."""
    predict_fn = getattr(model, "predict", None)
    if callable(predict_fn):
        return cast(_PredictorLike, model)

    underlying = getattr(model, "model_", None)
    if underlying is not None:
        predict_fn = getattr(underlying, "predict", None)
        if callable(predict_fn):
            return cast(_PredictorLike, underlying)

    raise TypeError(
        "quick_evaluate requires an object with a callable 'predict' method. Pass a fitted ANFIS "
        "model or estimator such as ANFISRegressor."
    )


def quick_evaluate(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    print_results: bool = True,
    task: Literal["auto", "regression", "classification"] = "auto",
) -> dict[str, float]:
    """Evaluate a trained ANFIS model or estimator on test data."""
    predictor = _resolve_predictor(model)
    X_arr = np.asarray(X_test, dtype=float)
    y_vec = np.asarray(y_test)
    y_pred_raw = predictor.predict(X_arr)

    y_proba = None
    predict_proba = getattr(predictor, "predict_proba", None)
    if callable(predict_proba):
        y_proba = predict_proba(X_arr)

    report = compute_metrics(
        y_vec,
        y_pred=y_pred_raw,
        y_proba=y_proba,
        task=task,
    )
    metrics = report.to_dict()

    if print_results:
        print("=" * 50)  # noqa: T201
        print("ANFIS Model Evaluation Results")  # noqa: T201
        print("=" * 50)  # noqa: T201
        if report.task == "regression":
            print(f"Mean Squared Error (MSE):     {metrics['mse']:.6f}")  # noqa: T201
            print(f"Root Mean Squared Error:      {metrics['rmse']:.6f}")  # noqa: T201
            print(f"Mean Absolute Error (MAE):    {metrics['mae']:.6f}")  # noqa: T201
            print(f"Median Absolute Error:        {metrics['median_absolute_error']:.6f}")  # noqa: T201
            print(f"R-squared (R²):               {metrics['r2']:.4f}")  # noqa: T201
            print(f"Explained Variance:           {metrics['explained_variance']:.4f}")  # noqa: T201
            print(f"Symmetric MAPE:               {metrics['smape']:.2f}%")  # noqa: T201
            print(f"Max Error:                    {metrics['max_error']:.6f}")  # noqa: T201
            print(f"Std. of Error:                {metrics['std_error']:.6f}")  # noqa: T201
        else:
            print(f"Accuracy:                     {metrics['accuracy']:.4f}")  # noqa: T201
            print(f"Balanced Accuracy:            {metrics['balanced_accuracy']:.4f}")  # noqa: T201
            if not np.isnan(metrics.get("log_loss", float("nan"))):
                print(f"Log Loss:                     {metrics['log_loss']:.6f}")  # noqa: T201
            print(f"Precision (macro):            {metrics['precision_macro']:.4f}")  # noqa: T201
            print(f"Recall (macro):               {metrics['recall_macro']:.4f}")  # noqa: T201
            print(f"F1-score (macro):             {metrics['f1_macro']:.4f}")  # noqa: T201
        print("=" * 50)  # noqa: T201

    # For backward compatibility keep returning plain dict but include rich metrics.
    return {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in metrics.items()}
