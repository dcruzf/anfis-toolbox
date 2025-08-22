from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HybridTrainer:
    """Original Jang (1993) hybrid training: LSM for consequents + GD for antecedents."""

    learning_rate: float = 0.01
    epochs: int = 100
    verbose: bool = True

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Trains the given model using the hybrid training algorithm for a specified number of epochs.

        Parameters
        ----------
        model : object
            The model to be trained. Must implement a `hybrid_train_step` method.
        X : np.ndarray
            Input features, expected as a NumPy array.
        y : np.ndarray
            Target values, expected as a NumPy array.

        Returns:
        -------
        list[float]
            A list containing the loss value for each epoch.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        losses: list[float] = []
        for _ in range(self.epochs):
            loss = model.hybrid_train_step(X, y, learning_rate=self.learning_rate)
            losses.append(loss)
        return losses
