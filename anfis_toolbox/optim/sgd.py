from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..losses import LossFunction, resolve_loss
from .base import BaseTrainer


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
    verbose: bool = True
    loss: LossFunction | str | None = None
    _loss_fn: LossFunction = field(init=False, repr=False)

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train the model using pure backpropagation.

        Uses the model's forward/backward/update APIs directly, without requiring
        a model.train_step method. Returns a list of loss values per epoch.
        Loss is MSE, computed as ``mean((y_pred - y)**2)``; 1D ``y`` is reshaped
        to ``(n,1)`` for convenience.
        """
        self._loss_fn = resolve_loss(self.loss)
        X = np.asarray(X, dtype=float)
        y_prepared = self._loss_fn.prepare_targets(y, model=model)
        if y_prepared.shape[0] != X.shape[0]:
            raise ValueError("Target array must have same number of rows as X")

        n_samples = X.shape[0]
        losses: list[float] = []

        for _ in range(self.epochs):
            if self.batch_size is None:
                # Full-batch gradient descent
                loss = self._compute_loss_backward_and_update(model, X, y_prepared)
                losses.append(loss)
            else:
                # Mini-batch SGD
                indices = np.arange(n_samples)
                if self.shuffle:
                    np.random.shuffle(indices)
                batch_losses: list[float] = []
                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    batch_idx = indices[start:end]
                    batch_loss = self._compute_loss_backward_and_update(model, X[batch_idx], y_prepared[batch_idx])
                    batch_losses.append(batch_loss)
                # For compatibility, record epoch loss as mean of batch losses
                losses.append(float(np.mean(batch_losses)))

        return losses

    def init_state(self, model, X: np.ndarray, y: np.ndarray):
        """SGD has no persistent optimizer state; returns None."""
        return None

    def train_step(self, model, Xb: np.ndarray, yb: np.ndarray, state):
        """Perform one SGD step on a batch and return (loss, state)."""
        loss_fn = self._ensure_loss_fn()
        yb_prepared = loss_fn.prepare_targets(yb, model=model)
        loss = self._compute_loss_backward_and_update(model, Xb, yb_prepared)
        return loss, state

    def _compute_loss_backward_and_update(self, model, Xb: np.ndarray, yb: np.ndarray) -> float:
        """Forward -> MSE -> backward -> update parameters; returns loss."""
        model.reset_gradients()
        y_pred = model.forward(Xb)
        loss = self._loss_fn.loss(yb, y_pred)
        dL_dy = self._loss_fn.gradient(yb, y_pred)
        model.backward(dL_dy)
        model.update_parameters(self.learning_rate)
        return loss

    def _ensure_loss_fn(self) -> LossFunction:
        if not hasattr(self, "_loss_fn"):
            self._loss_fn = resolve_loss(self.loss)
        return self._loss_fn
