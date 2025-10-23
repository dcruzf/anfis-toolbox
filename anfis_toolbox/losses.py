"""Loss functions and their gradients for ANFIS Toolbox.

This module centralizes the loss definitions used during training to make it
explicit which objective is being optimized. Trainers can import from here so
the chosen loss is clear in one place.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import cross_entropy as _cross_entropy
from .metrics import mean_squared_error as _mse
from .metrics import softmax as _softmax


class LossFunction:
    """Base interface for losses used by trainers."""

    def prepare_targets(self, y: Any, *, model: Any | None = None) -> np.ndarray:
        """Return targets in a format compatible with forward/gradient computations."""
        return np.asarray(y, dtype=float)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:  # pragma: no cover - interface
        """Compute the scalar loss for the given targets and predictions."""
        raise NotImplementedError

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        """Return the gradient of the loss with respect to the predictions."""
        raise NotImplementedError


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


class MSELoss(LossFunction):
    """Mean squared error loss packaged for trainer consumption."""

    def prepare_targets(self, y: Any, *, model: Any | None = None) -> np.ndarray:
        """Convert 1D targets into column vectors expected by MSE computations."""
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        return y_arr

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Delegate to :func:`mse_loss` for value computation."""
        return mse_loss(y_true, y_pred)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Delegate to :func:`mse_grad` for gradient computation."""
        return mse_grad(y_true, y_pred)


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
    elif yt.shape != logits.shape:
        raise ValueError("y_true one-hot must have same shape as logits")
    else:
        yt = yt.astype(float)
    # probs
    probs = _softmax(logits, axis=1)
    return (probs - yt) / float(n)


class CrossEntropyLoss(LossFunction):
    """Categorical cross-entropy loss operating on logits."""

    def prepare_targets(self, y: Any, *, model: Any | None = None) -> np.ndarray:
        """Convert labels or one-hot encodings into dense float matrices."""
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            n_classes_attr = getattr(model, "n_classes", None) if model is not None else None
            if n_classes_attr is not None:
                n_classes = int(n_classes_attr)
            else:
                n_classes = int(np.max(y_arr)) + 1
            oh = np.zeros((y_arr.shape[0], n_classes), dtype=float)
            oh[np.arange(y_arr.shape[0]), y_arr.astype(int)] = 1.0
            return oh
        if y_arr.ndim != 2:
            raise ValueError("y for cross-entropy must be 1D labels or 2D one-hot encoded")
        expected_attr = getattr(model, "n_classes", None) if model is not None else None
        if expected_attr is not None:
            expected = int(expected_attr)
            if y_arr.shape[1] != expected:
                raise ValueError(f"y one-hot must have {expected} columns")
        return y_arr.astype(float)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Delegate to :func:`cross_entropy_loss` for value computation."""
        return cross_entropy_loss(y_true, y_pred)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Delegate to :func:`cross_entropy_grad` for gradient computation."""
        return cross_entropy_grad(y_true, y_pred)


LOSS_REGISTRY: dict[str, type[LossFunction]] = {
    "mse": MSELoss,
    "mean_squared_error": MSELoss,
    "cross_entropy": CrossEntropyLoss,
    "crossentropy": CrossEntropyLoss,
    "cross-entropy": CrossEntropyLoss,
}


def resolve_loss(loss: str | LossFunction | None) -> LossFunction:
    """Resolve user-provided loss spec into a concrete ``LossFunction`` instance."""
    if loss is None:
        return MSELoss()
    if isinstance(loss, LossFunction):
        return loss
    if isinstance(loss, str):
        key = loss.lower()
        if key not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss '{loss}'. Available: {sorted(LOSS_REGISTRY)}")
        return LOSS_REGISTRY[key]()
    raise TypeError("loss must be None, str, or a LossFunction instance")


__all__ = [
    "LossFunction",
    "MSELoss",
    "CrossEntropyLoss",
    "mse_loss",
    "mse_grad",
    "cross_entropy_loss",
    "cross_entropy_grad",
    "resolve_loss",
]
