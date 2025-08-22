from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SGDTrainer:
    """Stochastic gradient descent trainer for ANFIS.

    Parameters:
        learning_rate: Step size for gradient descent.
        epochs: Number of passes over the data.
        batch_size: Mini-batch size; if None uses full batch.
        shuffle: Whether to shuffle data each epoch.
        verbose: Whether to log progress (delegated to model logging settings).
    """

    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: None | int = None
    shuffle: bool = True
    verbose: bool = True

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train the model using pure backpropagation.

        Returns a list of loss values per epoch to preserve current API.
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
                loss = model.train_step(X, y, learning_rate=self.learning_rate)
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
                    batch_loss = model.train_step(X[batch_idx], y[batch_idx], learning_rate=self.learning_rate)
                    batch_losses.append(batch_loss)
                # For compatibility, record epoch loss as mean of batch losses
                losses.append(float(np.mean(batch_losses)))

        return losses
