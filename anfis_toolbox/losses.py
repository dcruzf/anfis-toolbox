"""Loss functions and their gradients for ANFIS Toolbox.

This module centralizes the loss definitions used during training to make it
explicit which objective is being optimized. Trainers can import from here so
the chosen loss is clear in one place.
"""

from __future__ import annotations

import numpy as np

from .metrics import cross_entropy as _cross_entropy
from .metrics import mean_squared_error as _mse
from .metrics import softmax as _softmax


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error (MSE) loss.

    Parameters:
        y_true: Array-like true targets of shape (n, d) or (n,).
        y_pred: Array-like predictions of same shape as y_true.

    Returns:
        Scalar MSE value.
    """
    return float(_mse(y_true, y_pred))


def mse_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Gradient of MSE w.r.t. predictions.

    d/dy_pred MSE = 2 * (y_pred - y_true) / n
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = max(1, yt.shape[0])
    return 2.0 * (yp - yt) / float(n)


def cross_entropy_loss(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Cross-entropy loss from labels (int or one-hot) and logits.

    This delegates to metrics.cross_entropy for the scalar value.
    """
    return float(_cross_entropy(y_true, logits))


def cross_entropy_grad(y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """Gradient of cross-entropy w.r.t logits.

    Accepts integer labels (n,) or one-hot (n,k). Returns gradient with the
    same shape as logits: (n,k).
    """
    logits = np.asarray(logits, dtype=float)
    n, k = logits.shape[0], logits.shape[1]
    yt = np.asarray(y_true)
    if yt.ndim == 1:
        oh = np.zeros((n, k), dtype=float)
        oh[np.arange(n), yt.astype(int)] = 1.0
        yt = oh
    # probs
    probs = _softmax(logits, axis=1)
    return (probs - yt) / float(n)
