from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
                - Optimizes mean squared error (MSE) between ``model.forward(X)`` and ``y``.
                    For regression ANFIS this is the standard objective.
                - With ``ANFISClassifier``, this trainer will still run but will minimize MSE
                    on logits/probabilities rather than cross‑entropy. For classification, prefer
                    using ``ANFISClassifier.fit(...)`` which uses softmax + cross‑entropy.
    """

    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: None | int = None
    shuffle: bool = True
    verbose: bool = True

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train the model using pure backpropagation.

        Uses the model's forward/backward/update APIs directly, without requiring
        a model.train_step method. Returns a list of loss values per epoch.
        Loss is MSE, computed as ``mean((y_pred - y)**2)``; 1D ``y`` is reshaped
        to ``(n,1)`` for convenience.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        losses: list[float] = []

        for _ in range(self.epochs):
            if self.batch_size is None:
                # Full-batch gradient descent
                model.reset_gradients()
                y_pred = model.forward(X)
                loss = float(np.mean((y_pred - y) ** 2))
                dL_dy = 2 * (y_pred - y) / y.shape[0]
                model.backward(dL_dy)
                model.update_parameters(self.learning_rate)
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
                    model.reset_gradients()
                    y_pred_b = model.forward(X[batch_idx])
                    batch_loss = float(np.mean((y_pred_b - y[batch_idx]) ** 2))
                    dL_dy_b = 2 * (y_pred_b - y[batch_idx]) / y[batch_idx].shape[0]
                    model.backward(dL_dy_b)
                    model.update_parameters(self.learning_rate)
                    batch_losses.append(batch_loss)
                # For compatibility, record epoch loss as mean of batch losses
                losses.append(float(np.mean(batch_losses)))

        return losses
