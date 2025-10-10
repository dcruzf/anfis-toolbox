from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..losses import LossFunction, resolve_loss
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
        Supports configurable losses via the ``loss`` parameter. Defaults to mean squared error for
        regression tasks but can be switched to other differentiable objectives such as categorical
        cross-entropy when training ``ANFISClassifier`` models.
    """

    learning_rate: float = 0.001
    rho: float = 0.9
    epsilon: float = 1e-8
    epochs: int = 100
    batch_size: None | int = None
    shuffle: bool = True
    verbose: bool = False
    loss: LossFunction | str | None = None
    _loss_fn: LossFunction = field(init=False, repr=False)

    def _prepare_training_data(self, model, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._loss_fn = resolve_loss(self.loss)
        X_arr = np.asarray(X, dtype=float)
        y_arr = self._loss_fn.prepare_targets(y, model=model)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("Target array must have same number of rows as X")
        return X_arr, y_arr

    def _prepare_validation_data(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X_val, dtype=float)
        y_arr = self._loss_fn.prepare_targets(y_val, model=model)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("Validation targets must match input rows")
        return X_arr, y_arr

    def init_state(self, model, X: np.ndarray, y: np.ndarray):
        """Initialize RMSProp caches for consequents and membership scalars."""
        params = model.get_parameters()
        return {"params": params, "cache": _zeros_like_structure(params)}

    def train_step(self, model, Xb: np.ndarray, yb: np.ndarray, state):
        """One RMSProp step on a batch; returns (loss, updated_state)."""
        loss, grads = self._compute_loss_and_grads(model, Xb, yb)
        self._apply_rmsprop_step(model, state["params"], state["cache"], grads)
        return loss, state

    def _compute_loss_and_grads(self, model, Xb: np.ndarray, yb: np.ndarray) -> tuple[float, dict]:
        """Forward pass, MSE loss, backward pass, and gradients for a batch.

        Returns (loss, grads) where grads follows model.get_gradients() structure.
        """
        model.reset_gradients()
        y_pred = model.forward(Xb)
        loss = self._loss_fn.loss(yb, y_pred)
        dL_dy = self._loss_fn.gradient(yb, y_pred)
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

    def compute_loss(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Return the current loss value for ``(X, y)`` without modifying state."""
        preds = model.forward(X)
        return float(self._loss_fn.loss(y, preds))

    def _ensure_loss_fn(self) -> LossFunction:
        if not hasattr(self, "_loss_fn"):
            self._loss_fn = resolve_loss(self.loss)
        return self._loss_fn
