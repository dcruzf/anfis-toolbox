from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..losses import mse_grad, mse_loss
from .base import BaseTrainer


def _zeros_like_structure(params):
    """Create a zero-structure matching model.get_parameters() format.

    Returns a dict with:
      - 'consequent': np.zeros_like(params['consequent'])
      - 'membership': { name: [ {param_name: 0.0, ...} ] }
    """
    out = {"consequent": np.zeros_like(params["consequent"]), "membership": {}}
    for name, mf_list in params["membership"].items():
        out["membership"][name] = []
        for mf_params in mf_list:
            out["membership"][name].append(dict.fromkeys(mf_params.keys(), 0.0))
    return out


@dataclass
class RMSPropTrainer(BaseTrainer):
    """RMSProp optimizer-based trainer for ANFIS.

    Parameters:
        learning_rate: Base step size (alpha).
        rho: Exponential decay rate for the squared gradient moving average.
        epsilon: Small constant for numerical stability.
        epochs: Number of passes over the dataset.
        batch_size: If None, use full-batch; otherwise mini-batches of this size.
        shuffle: Whether to shuffle the data at each epoch when using mini-batches.
        verbose: Unused here; kept for API parity.

    Notes:
        - Optimizes mean squared error (MSE) between ``model.forward(X)`` and ``y``.
          For regression ANFIS this is the intended objective.
        - With ``ANFISClassifier``, this trainer will still execute (treating integer
          labels reshaped to ``(n,1)`` or one-hot targets) but it will minimize MSE on
          logits/probabilities. For classification tasks, prefer ``ANFISClassifier.fit``
          which uses cross-entropy with softmax.
    """

    learning_rate: float = 0.001
    rho: float = 0.9
    epsilon: float = 1e-8
    epochs: int = 100
    batch_size: None | int = None
    shuffle: bool = True
    verbose: bool = True

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train the model using RMSProp optimization.

        Steps per update:
        1) forward -> compute MSE loss
        2) backward -> obtain gradients via ``model.get_gradients()``
        3) update parameters with RMSProp rule using per-parameter caches
        """
        X, y = self._prepare_data(X, y)

        n_samples = X.shape[0]

        # Parameter structures and RMSProp caches
        params = model.get_parameters()
        cache = _zeros_like_structure(params)

        losses: list[float] = []
        for _ in range(self.epochs):
            if self.batch_size is None:
                # Full-batch RMSProp step
                loss, grads = self._compute_mse_and_grads(model, X, y)
                self._apply_rmsprop_step(model, params, cache, grads)
                losses.append(loss)
            else:
                indices = np.arange(n_samples)
                if self.shuffle:
                    np.random.shuffle(indices)
                batch_losses: list[float] = []
                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    batch_idx = indices[start:end]
                    batch_loss, grads_b = self._compute_mse_and_grads(model, X[batch_idx], y[batch_idx])
                    self._apply_rmsprop_step(model, params, cache, grads_b)
                    batch_losses.append(batch_loss)
                losses.append(float(np.mean(batch_losses)))

        return losses

    def init_state(self, model, X: np.ndarray, y: np.ndarray):
        """Initialize RMSProp caches for consequents and membership scalars."""
        params = model.get_parameters()
        return {"params": params, "cache": _zeros_like_structure(params)}

    def train_step(self, model, Xb: np.ndarray, yb: np.ndarray, state):
        """One RMSProp step on a batch; returns (loss, updated_state)."""
        loss, grads = self._compute_mse_and_grads(model, Xb, yb)
        self._apply_rmsprop_step(model, state["params"], state["cache"], grads)
        return loss, state

    @staticmethod
    def _prepare_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert inputs to float numpy arrays and ensure y is 2D.

        Returns X, y where y has shape (n, 1) if originally 1D.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return X, y

    def _compute_mse_and_grads(self, model, Xb: np.ndarray, yb: np.ndarray) -> tuple[float, dict]:
        """Forward pass, MSE loss, backward pass, and gradients for a batch.

        Returns (loss, grads) where grads follows model.get_gradients() structure.
        """
        model.reset_gradients()
        y_pred = model.forward(Xb)
        loss = mse_loss(yb, y_pred)
        dL_dy = mse_grad(yb, y_pred)
        model.backward(dL_dy)
        grads = model.get_gradients()
        return loss, grads

    def _apply_rmsprop_step(
        self,
        model,
        params: dict,
        cache: dict,
        grads: dict,
    ) -> None:
        """Apply one RMSProp update to params using grads and caches.

        Updates both consequent array parameters and membership scalar parameters.
        """
        # Consequent is a numpy array
        g = grads["consequent"]
        c = cache["consequent"]
        c[:] = self.rho * c + (1.0 - self.rho) * (g * g)
        params["consequent"] = params["consequent"] - self.learning_rate * g / (np.sqrt(c) + self.epsilon)

        # Membership are scalars in nested dicts
        for name in params["membership"].keys():
            for i in range(len(params["membership"][name])):
                for key in params["membership"][name][i].keys():
                    gk = float(grads["membership"][name][i][key])
                    ck = cache["membership"][name][i][key]
                    ck = self.rho * ck + (1.0 - self.rho) * (gk * gk)
                    step = self.learning_rate * gk / (np.sqrt(ck) + self.epsilon)
                    params["membership"][name][i][key] -= float(step)
                    cache["membership"][name][i][key] = ck

        # Push updated params back into the model
        model.set_parameters(params)
