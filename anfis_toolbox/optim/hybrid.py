from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..losses import MSELoss
from .base import BaseTrainer


@dataclass
class HybridTrainer(BaseTrainer):
    """Original Jang (1993) hybrid training: LSM for consequents + GD for antecedents.

    Notes:
        This trainer assumes a single-output regression head. It is not compatible with
        :class:`~anfis_toolbox.model.TSKANFISClassifier` or the high-level
        :class:`~anfis_toolbox.classifier.ANFISClassifier` facade.
    """

    learning_rate: float = 0.01
    epochs: int = 100
    verbose: bool = False
    _loss_fn: MSELoss = MSELoss()

    def init_state(self, model: Any, X: np.ndarray, y: np.ndarray) -> None:
        """Hybrid trainer doesn't maintain optimizer state; returns None."""
        return None

    def train_step(self, model: Any, Xb: np.ndarray, yb: np.ndarray, state: None) -> tuple[float, None]:
        """Perform one hybrid step on a batch and return (loss, state).

        Equivalent to one iteration of the hybrid algorithm on the given batch.
        """
        Xb, yb = self._prepare_training_data(model, Xb, yb)
        # Forward to get normalized weights
        membership_outputs = model.membership_layer.forward(Xb)
        rule_strengths = model.rule_layer.forward(membership_outputs)
        normalized_weights = model.normalization_layer.forward(rule_strengths)

        # Build LSM system for batch
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

        # Loss and backward for antecedents only
        y_pred = model.consequent_layer.forward(Xb, normalized_weights)
        loss = self._loss_fn.loss(yb, y_pred)
        dL_dy = self._loss_fn.gradient(yb, y_pred)
        dL_dnorm_w, _ = model.consequent_layer.backward(dL_dy)
        dL_dw = model.normalization_layer.backward(dL_dnorm_w)
        gradients = model.rule_layer.backward(dL_dw)
        model.membership_layer.backward(gradients)
        model.update_membership_parameters(self.learning_rate)
        return float(loss), state

    def compute_loss(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the hybrid MSE loss on prepared data without side effects."""
        X_arr, y_arr = self._prepare_validation_data(model, X, y)
        membership_outputs = model.membership_layer.forward(X_arr)
        rule_strengths = model.rule_layer.forward(membership_outputs)
        normalized_weights = model.normalization_layer.forward(rule_strengths)
        preds = model.consequent_layer.forward(X_arr, normalized_weights)
        return float(self._loss_fn.loss(y_arr, preds))
