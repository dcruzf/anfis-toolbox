from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..losses import MSELoss
from .adam import _zeros_like_structure
from .base import BaseTrainer
from .parameter_utils import (
    iterate_membership_params_with_state,
    update_membership_param,
)


@dataclass
class HybridAdamTrainer(BaseTrainer):
    """Hybrid training: LSM for consequents + Adam for antecedents.

    Notes:
        This variant also targets the regression ANFIS. It is not compatible with the
        classification head (:class:`~anfis_toolbox.model.TSKANFISClassifier`) or
        :class:`~anfis_toolbox.classifier.ANFISClassifier`.
    """

    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    epochs: int = 100
    verbose: bool = False
    _loss_fn: MSELoss = MSELoss()

    def init_state(self, model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Initialize Adam moment tensors for membership parameters."""
        params = model.get_parameters()
        zero_struct = _zeros_like_structure(params)["membership"]
        return {"m": deepcopy(zero_struct), "v": deepcopy(zero_struct), "t": 0}

    def train_step(
        self, model: Any, Xb: np.ndarray, yb: np.ndarray, state: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """Execute one hybrid iteration combining LSM and Adam updates."""
        model.reset_gradients()
        Xb, yb = self._prepare_data(Xb, yb)
        membership_outputs = model.membership_layer.forward(Xb)
        rule_strengths = model.rule_layer.forward(membership_outputs)
        normalized_weights = model.normalization_layer.forward(rule_strengths)
        ones_col = np.ones((Xb.shape[0], 1), dtype=float)
        x_bar = np.concatenate([Xb, ones_col], axis=1)
        A_blocks = [normalized_weights[:, j : j + 1] * x_bar for j in range(model.n_rules)]
        A = np.concatenate(A_blocks, axis=1)
        try:
            regularization = 1e-6 * np.eye(A.shape[1])
            ATA_reg = A.T @ A + regularization
            theta = np.linalg.solve(ATA_reg, A.T @ yb.flatten())
        except np.linalg.LinAlgError:
            logging.getLogger(__name__).warning("Matrix singular in LSM, using pseudo-inverse")
            theta = np.linalg.pinv(A) @ yb.flatten()
        model.consequent_layer.parameters = theta.reshape(model.n_rules, model.n_inputs + 1)

        # Adam for antecedents
        y_pred = model.consequent_layer.forward(Xb, normalized_weights)
        loss = self._loss_fn.loss(yb, y_pred)
        dL_dy = self._loss_fn.gradient(yb, y_pred)
        dL_dnorm_w, _ = model.consequent_layer.backward(dL_dy)
        dL_dw = model.normalization_layer.backward(dL_dnorm_w)
        gradients = model.rule_layer.backward(dL_dw)
        grad_struct = model.membership_layer.backward(gradients)
        self._apply_adam_update(model, grad_struct, state)
        return float(loss), state

    def _apply_adam_update(self, model: Any, grad_struct: dict[str, Any], state: dict[str, Any]) -> None:
        params = model.get_parameters()
        m = {"membership": state["m"]}
        v = {"membership": state["v"]}
        t = state["t"] = state["t"] + 1
        for path, param_val, m_val, grad in iterate_membership_params_with_state(params, m, grad_struct):
            grad = cast(float, grad)
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
        model.set_parameters(params)

    def compute_loss(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate mean squared error on provided data without updates."""
        X_arr, y_arr = self._prepare_data(X, y)
        membership_outputs = model.membership_layer.forward(X_arr)
        rule_strengths = model.rule_layer.forward(membership_outputs)
        normalized_weights = model.normalization_layer.forward(rule_strengths)
        preds = model.consequent_layer.forward(X_arr, normalized_weights)
        return float(self._loss_fn.loss(y_arr, preds))

    @staticmethod
    def _prepare_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return X, y
