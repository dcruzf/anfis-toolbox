"""Loss functions and their gradients for ANFIS Toolbox.

This module centralizes the loss definitions used during training to make it
explicit which objective is being optimized. Trainers can import from here so
the chosen loss is clear in one place.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np


class LossFunction:
    """Base interface for losses used by trainers.

    This abstract class defines the contract that all loss functions must implement.
    Subclasses should override the `loss`, `gradient`, and optionally `prepare_targets`
    methods to implement specific loss functions.

    The typical workflow is:
        1. Call `prepare_targets` to format raw targets into the expected format
        2. Call `loss` to compute the scalar loss value
        3. Call `gradient` to compute loss gradients for backpropagation
    """

    def prepare_targets(self, y: Any, *, model: Any | None = None) -> np.ndarray:
        """Return targets in a format compatible with forward/gradient computations."""
        return np.asarray(y, dtype=float)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:  # pragma: no cover - interface
        """Compute the scalar loss for the given targets and predictions."""
        raise NotImplementedError

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        """Return the gradient of the loss with respect to the predictions."""
        raise NotImplementedError


class MSELoss(LossFunction):
    """Mean squared error loss packaged for trainer consumption.

    Implements the MSE loss function commonly used for regression tasks.
    MSE measures the average squared difference between predicted and actual values.

    The loss is defined as:
        L = (1/n) * Σ(y_pred - y_true)²

    And its gradient with respect to predictions is:
        ∇L = (2/n) * (y_pred - y_true)
    """

    def prepare_targets(self, y: Any, *, model: Any | None = None) -> np.ndarray:
        """Convert 1D targets into column vectors expected by MSE computations.

        Parameters:
            y: Array-like target values. Can be 1D or already 2D.
            model: Optional model instance (unused for MSE).

        Returns:
            np.ndarray: Targets as a 2D column vector of shape (n_samples, 1).
        """
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        return y_arr

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient of MSE with respect to predictions.

        The gradient is computed as: ∇L = (2/n) * (y_pred - y_true)

        Parameters:
            y_true: True target values, shape (n_samples, 1).
            y_pred: Predicted values, same shape as y_true.

        Returns:
            np.ndarray: Gradient array with same shape as y_pred.
        """
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        n = max(1, yt.shape[0])
        return cast(np.ndarray, 2.0 * (yp - yt) / float(n))


class CrossEntropyLoss(LossFunction):
    """Categorical cross-entropy loss operating on logits.

    Implements cross-entropy loss for multi-class classification tasks.
    Accepts raw logits (unbounded scores) and computes numerically stable loss
    using log-softmax formulation.

    The loss is defined as:
        L = -(1/n) * Σ Σ y_true[i,j] * log(softmax(logits)[i,j])

    And its gradient with respect to logits is:
        ∇L = (1/n) * (softmax(logits) - y_true)

    Numerical stability is achieved through:
        - Stable log-softmax computation in `loss` method
        - Stable softmax via maximum subtraction in `gradient` method
    """

    def _stable_softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Compute softmax with numerical stability.

        Implements the numerically stable softmax by subtracting the maximum
        value along each row before exponentiation. This prevents overflow errors
        that would occur with large logits.

        Formula: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

        Parameters:
            x: Input logits array, shape (..., n_classes).
            axis: Axis along which to compute softmax (typically 1 for batch).

        Returns:
            np.ndarray: Normalized probabilities with same shape as input,
                       values in range [0, 1] summing to 1 along the specified axis.
        """
        zmax = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - zmax)
        return cast(np.ndarray, exp_x / np.sum(exp_x, axis=axis, keepdims=True))

    def prepare_targets(self, y: Any, *, model: Any | None = None) -> np.ndarray:
        """Convert labels or one-hot encodings into dense float matrices.

        Accepts either:
            - 1D integer class labels (0 to n_classes-1)
            - 2D one-hot encoded targets

        If 1D labels are provided, automatically converts to one-hot encoding.
        If model is provided with an n_classes attribute, validates consistency.

        Parameters:
            y: Target labels as 1D array of integers or 2D one-hot array.
            model: Optional model instance. If provided, uses model.n_classes
                  to infer number of classes and validate dimensions.

        Returns:
            np.ndarray: One-hot encoded targets of shape (n_samples, n_classes).

        Raises:
            ValueError: If y dimension is not 1 or 2, or if dimensions don't match model.
        """
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
        """Compute mean cross-entropy from integer labels or one-hot vs logits.

        Uses stable log-softmax computation to prevent numerical underflow.
        Handles both integer class labels and one-hot encoded targets.

        Parameters:
            y_true: Array of shape (n_samples,) of integer class labels (0 to n_classes-1),
                   or one-hot encoded array of shape (n_samples, n_classes).
            y_pred: Raw logit scores of shape (n_samples, n_classes).

        Returns:
            float: Mean cross-entropy loss across all samples.

        Notes:
            - Returns 0.0 if batch is empty (n_samples == 0)
            - Numerically stable for arbitrarily large or small logit values
        """
        logits = np.asarray(y_pred, dtype=float)
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

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient of cross-entropy with respect to logits.

        The gradient simplifies to: softmax(logits) - one_hot(y_true)
        This form is derived from the chain rule applied to the cross-entropy loss.

        Accepts integer labels or one-hot encoded targets. Returns gradient
        with the same shape as logits.

        Parameters:
            y_true: Array of shape (n_samples,) of integer class labels, or
                   one-hot encoded array of shape (n_samples, n_classes).
            y_pred: Raw logit scores of shape (n_samples, n_classes).

        Returns:
            np.ndarray: Gradient of shape (n_samples, n_classes) with values typically
                       in range [-1, 1] indicating direction to decrease loss.

        Raises:
            ValueError: If one-hot y_true shape doesn't match logits shape.
        """
        logits = np.asarray(y_pred, dtype=float)
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
        probs = self._stable_softmax(logits, axis=1)
        return cast(np.ndarray, (probs - yt) / float(n))


LOSS_REGISTRY: dict[str, type[LossFunction]] = {
    "mse": MSELoss,
    "mean_squared_error": MSELoss,
    "cross_entropy": CrossEntropyLoss,
    "crossentropy": CrossEntropyLoss,
    "cross-entropy": CrossEntropyLoss,
}


def resolve_loss(loss: str | LossFunction | None) -> LossFunction:
    """Resolve user-provided loss spec into a concrete ``LossFunction`` instance.

    Provides flexible loss specification allowing string names, instances, or None.

    Parameters:
        loss: Loss specification as one of:
            - None: Returns MSELoss() as default
            - str: Key from LOSS_REGISTRY (case-insensitive)
            - LossFunction: Returned as-is

    Returns:
        LossFunction: Instantiated loss function ready for use.

    Raises:
        ValueError: If string loss is not in LOSS_REGISTRY.
        TypeError: If loss is not None, str, or LossFunction instance.

    Examples:
        >>> loss1 = resolve_loss(None)  # Returns MSELoss()
        >>> loss2 = resolve_loss("mse")
        >>> loss3 = resolve_loss("cross_entropy")
        >>> loss4 = resolve_loss(CrossEntropyLoss())
    """
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
    "resolve_loss",
]
