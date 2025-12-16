from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ..losses import LossFunction
from .base import BaseTrainer
from .parameter_utils import (
    iterate_membership_params_with_state,
    update_membership_param,
)


def _zeros_like_structure(params: dict[str, Any]) -> dict[str, Any]:
    """Create a zero-structure matching model.get_parameters() format.

    Returns a dict with:
      - 'consequent': np.zeros_like(params['consequent'])
      - 'membership': { name: [ {param_name: 0.0, ...} ] }
    """
    out: dict[str, Any] = {"consequent": np.zeros_like(params["consequent"]), "membership": {}}
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

    def init_state(self, model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Initialize RMSProp caches for consequents and membership scalars."""
        params = model.get_parameters()
        return {"params": params, "cache": _zeros_like_structure(params)}

    def train_step(
        self, model: Any, Xb: np.ndarray, yb: np.ndarray, state: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """One RMSProp step on a batch; returns (loss, updated_state)."""
        loss, grads = self._compute_loss_and_grads(model, Xb, yb)
        self._apply_rmsprop_step(model, state["params"], state["cache"], grads)
        return loss, state

    def _compute_loss_and_grads(self, model: Any, Xb: np.ndarray, yb: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Forward pass, MSE loss, backward pass, and gradients for a batch.

        Returns (loss, grads) where grads follows model.get_gradients() structure.
        """
        loss_fn = self._get_loss_fn()
        model.reset_gradients()
        y_pred = model.forward(Xb)
        loss = loss_fn.loss(yb, y_pred)
        dL_dy = loss_fn.gradient(yb, y_pred)
        model.backward(dL_dy)
        grads = model.get_gradients()
        return loss, grads

    def _apply_rmsprop_step(
        self,
        model: Any,
        params: dict[str, Any],
        cache: dict[str, Any],
        grads: dict[str, Any],
    ) -> None:
        """Apply one RMSProp update to params using grads and caches.

        Updates both consequent array parameters and membership scalar parameters.
        """
        # Consequent is a numpy array
        g = grads["consequent"]
        c = cache["consequent"]
        c[:] = self.rho * c + (1.0 - self.rho) * (g * g)
        params["consequent"] = params["consequent"] - self.learning_rate * g / (np.sqrt(c) + self.epsilon)

        # Membership parameters (scalars in nested dicts)
        for path, param_val, cache_val, grad in iterate_membership_params_with_state(params, cache, grads):
            grad = cast(float, grad)  # grads_dict is provided in this context
            cache_val = self.rho * cache_val + (1.0 - self.rho) * (grad * grad)
            step = self.learning_rate * grad / (np.sqrt(cache_val) + self.epsilon)
            new_param = param_val - step
            update_membership_param(params, path, new_param)
            cache["membership"][path[0]][path[1]][path[2]] = cache_val

        # Push updated params back into the model
        model.set_parameters(params)

    def compute_loss(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Return the current loss value for ``(X, y)`` without modifying state."""
        loss_fn = self._get_loss_fn()
        preds = model.forward(X)
        return float(loss_fn.loss(y, preds))
