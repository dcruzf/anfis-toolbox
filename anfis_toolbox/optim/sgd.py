from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..losses import LossFunction
from .base import BaseTrainer, ModelLike


@dataclass
class SGDTrainer(BaseTrainer):
    """Stochastic gradient descent trainer for ANFIS.

    Parameters:
        learning_rate: Step size for gradient descent.
        epochs: Number of passes over the data.
        batch_size: Mini-batch size; if None uses full batch.
        shuffle: Whether to shuffle data each epoch.
        verbose: Whether to log progress (delegated to model logging settings).

    Notes:
        Uses the configurable loss provided via ``loss`` (defaults to mean squared error).
        The selected loss is responsible for adapting target shapes via ``prepare_targets``.
        When used with ``ANFISClassifier`` and ``loss="cross_entropy"`` it trains on logits with the
        appropriate softmax gradient.
    """

    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: None | int = None
    shuffle: bool = True
    verbose: bool = False
    loss: LossFunction | str | None = None
    _loss_fn: LossFunction = field(init=False, repr=False)

    def init_state(self, model: ModelLike, X: np.ndarray, y: np.ndarray) -> None:
        """SGD has no persistent optimizer state; returns None."""
        return None

    def train_step(self, model: ModelLike, Xb: np.ndarray, yb: np.ndarray, state: Any) -> tuple[float, Any]:
        """Perform one SGD step on a batch and return (loss, state)."""
        loss = self._compute_loss_backward_and_update(model, Xb, yb)
        return loss, state

    def _compute_loss_backward_and_update(self, model: ModelLike, Xb: np.ndarray, yb: np.ndarray) -> float:
        """Forward -> MSE -> backward -> update parameters; returns loss."""
        loss_fn = self._get_loss_fn()
        model.reset_gradients()
        y_pred = model.forward(Xb)
        loss = loss_fn.loss(yb, y_pred)
        dL_dy = loss_fn.gradient(yb, y_pred)
        model.backward(dL_dy)
        model.update_parameters(self.learning_rate)
        return loss

    def compute_loss(self, model: ModelLike, X: np.ndarray, y: np.ndarray) -> float:
        """Return the loss for ``(X, y)`` without mutating ``model``."""
        loss_fn = self._get_loss_fn()
        preds = model.forward(X)
        return float(loss_fn.loss(y, preds))
