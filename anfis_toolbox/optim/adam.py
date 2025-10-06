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


def _adam_update(param, grad, m, v, lr, beta1, beta2, eps, t):
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
    verbose: bool = True
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

    def train_step(self, model, Xb: np.ndarray, yb: np.ndarray, state):
        """One Adam step on a batch; returns (loss, updated_state)."""
        loss, grads = self._compute_loss_and_grads(model, Xb, yb)
        t_new = self._apply_adam_step(model, state["params"], grads, state["m"], state["v"], state["t"])
        state["t"] = t_new
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

    def _apply_adam_step(
        self,
        model,
        params: dict,
        grads: dict,
        m: dict,
        v: dict,
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
        # Membership (nested scalars)
        for name in params["membership"].keys():
            for i in range(len(params["membership"][name])):
                for key in params["membership"][name][i].keys():
                    g = float(grads["membership"][name][i][key])
                    m_val = m["membership"][name][i][key]
                    v_val = v["membership"][name][i][key]
                    m_val = self.beta1 * m_val + (1.0 - self.beta1) * g
                    v_val = self.beta2 * v_val + (1.0 - self.beta2) * (g * g)
                    m_hat = m_val / (1.0 - self.beta1**t)
                    v_hat = v_val / (1.0 - self.beta2**t)
                    step = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    params["membership"][name][i][key] -= float(step)
                    m["membership"][name][i][key] = m_val
                    v["membership"][name][i][key] = v_val

        # Push updated params back into the model
        model.set_parameters(params)
        return t

    def compute_loss(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the configured loss on ``(X, y)`` without updating parameters."""
        preds = model.forward(X)
        return float(self._loss_fn.loss(y, preds))
