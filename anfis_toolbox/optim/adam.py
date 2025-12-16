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


def _adam_update(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    t: int,
) -> None:
    """Compute Adam update for numpy arrays (param, grad, m, v)."""
    m[:] = beta1 * m + (1.0 - beta1) * grad
    v[:] = beta2 * v + (1.0 - beta2) * (grad * grad)
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    param[:] = param - lr * m_hat / (np.sqrt(v_hat) + eps)


@dataclass
class AdamTrainer(BaseTrainer):
    """Adam optimizer-based trainer for ANFIS.

    Parameters:
        learning_rate: Base step size (alpha).
        beta1: Exponential decay rate for the first moment estimates.
        beta2: Exponential decay rate for the second moment estimates.
        epsilon: Small constant for numerical stability.
        epochs: Number of passes over the dataset.
        batch_size: If None, use full-batch; otherwise mini-batches of this size.
        shuffle: Whether to shuffle the data at each epoch when using mini-batches.
        verbose: Unused here; kept for API parity.

    Notes:
        Supports configurable losses via the ``loss`` parameter. Defaults to mean squared error for
        regression, but can minimize other differentiable objectives such as categorical
        cross-entropy when used with ``ANFISClassifier``.
    """

    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    epochs: int = 100
    batch_size: None | int = None
    shuffle: bool = True
    verbose: bool = False
    loss: LossFunction | str | None = None
    _loss_fn: LossFunction = field(init=False, repr=False)

    def init_state(self, model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Initialize Adam's first and second moments and time step.

        Returns a dict with keys: params, m, v, t.
        """
        params = model.get_parameters()
        return {
            "params": params,
            "m": _zeros_like_structure(params),
            "v": _zeros_like_structure(params),
            "t": 0,
        }

    def train_step(
        self, model: Any, Xb: np.ndarray, yb: np.ndarray, state: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """One Adam step on a batch; returns (loss, updated_state)."""
        loss, grads = self._compute_loss_and_grads(model, Xb, yb)
        t_val = cast(int, state["t"])
        t_new = self._apply_adam_step(model, state["params"], grads, state["m"], state["v"], t_val)
        state["t"] = t_new
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

    def _apply_adam_step(
        self,
        model: Any,
        params: dict[str, Any],
        grads: dict[str, Any],
        m: dict[str, Any],
        v: dict[str, Any],
        t: int,
    ) -> int:
        """Apply one Adam update to params using grads and moments; returns new time step.

        Updates both consequent array parameters and membership scalar parameters.
        """
        t += 1
        _adam_update(
            params["consequent"],
            grads["consequent"],
            m["consequent"],
            v["consequent"],
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            t,
        )
        # Membership parameters (scalars in nested dicts)
        for path, param_val, m_val, grad in iterate_membership_params_with_state(params, m, grads):
            grad = cast(float, grad)  # grads_dict is provided in this context
            m_val = self.beta1 * m_val + (1.0 - self.beta1) * grad
            v_val = v["membership"][path[0]][path[1]][path[2]]
            v_val = self.beta2 * v_val + (1.0 - self.beta2) * (grad * grad)
            m_hat = m_val / (1.0 - self.beta1**t)
            v_hat = v_val / (1.0 - self.beta2**t)
            step = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            new_param = param_val - step
            update_membership_param(params, path, new_param)
            m["membership"][path[0]][path[1]][path[2]] = m_val
            v["membership"][path[0]][path[1]][path[2]] = v_val

        # Push updated params back into the model
        model.set_parameters(params)
        return t

    def compute_loss(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the configured loss on ``(X, y)`` without updating parameters."""
        loss_fn = self._get_loss_fn()
        preds = model.forward(X)
        return float(loss_fn.loss(y, preds))
