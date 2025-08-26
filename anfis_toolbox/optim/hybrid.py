from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .base import BaseTrainer


@dataclass
class HybridTrainer(BaseTrainer):
    """Original Jang (1993) hybrid training: LSM for consequents + GD for antecedents."""

    learning_rate: float = 0.01
    epochs: int = 100
    verbose: bool = True

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train using the hybrid algorithm (LSM for consequents + GD for antecedents).

        This does not require any special method on the model besides its existing
        forward/backward-capable layers and membership/consequent parameter accessors.
        """
        X, y = self._prepare_data(X, y)

        logger = logging.getLogger(__name__)
        losses: list[float] = []
        for _ in range(self.epochs):
            # Ensure gradients are reset
            model.reset_gradients()

            # Forward through layers to compute normalized rule weights
            membership_outputs = model.membership_layer.forward(X)
            rule_strengths = model.rule_layer.forward(membership_outputs)
            normalized_weights = model.normalization_layer.forward(rule_strengths)

            # Build least-squares design matrix A for consequent parameters
            batch_size = X.shape[0]
            ones_col = np.ones((batch_size, 1), dtype=float)
            x_bar = np.concatenate([X, ones_col], axis=1)
            A_blocks = [normalized_weights[:, j : j + 1] * x_bar for j in range(model.n_rules)]
            A = np.concatenate(A_blocks, axis=1)

            # Solve for consequent parameters with small Tikhonov regularization
            try:
                regularization = 1e-6 * np.eye(A.shape[1])
                ATA_reg = A.T @ A + regularization
                theta = np.linalg.solve(ATA_reg, A.T @ y.flatten())
            except np.linalg.LinAlgError:
                logger.warning("Matrix singular in LSM, using pseudo-inverse")
                theta = np.linalg.pinv(A) @ y.flatten()

            model.consequent_layer.parameters = theta.reshape(model.n_rules, model.n_inputs + 1)

            # Compute output and loss with updated consequents
            y_pred = model.consequent_layer.forward(X, normalized_weights)
            loss = float(np.mean((y_pred - y) ** 2))

            # Backpropagate for antecedent (membership) parameters only
            dL_dy = 2 * (y_pred - y) / y.shape[0]
            dL_dnorm_w, _ = model.consequent_layer.backward(dL_dy)
            dL_dw = model.normalization_layer.backward(dL_dnorm_w)
            gradients = model.rule_layer.backward(dL_dw)
            model.membership_layer.backward(gradients)

            # Apply only membership parameter updates
            model._apply_membership_gradients(self.learning_rate)

            losses.append(loss)
        return losses

    def init_state(self, model, X: np.ndarray, y: np.ndarray):
        """Hybrid trainer doesn't maintain optimizer state; returns None."""
        return None

    def train_step(self, model, Xb: np.ndarray, yb: np.ndarray, state):
        """Perform one hybrid step on a batch and return (loss, state).

        Equivalent to one iteration of the hybrid algorithm on the given batch.
        """
        Xb, yb = self._prepare_data(Xb, yb)
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
            theta = np.linalg.pinv(A) @ yb.flatten()
        model.consequent_layer.parameters = theta.reshape(model.n_rules, model.n_inputs + 1)

        # Loss and backward for antecedents only
        y_pred = model.consequent_layer.forward(Xb, normalized_weights)
        loss = float(np.mean((y_pred - yb) ** 2))
        dL_dy = 2 * (y_pred - yb) / yb.shape[0]
        dL_dnorm_w, _ = model.consequent_layer.backward(dL_dy)
        dL_dw = model.normalization_layer.backward(dL_dnorm_w)
        gradients = model.rule_layer.backward(dL_dw)
        model.membership_layer.backward(gradients)
        model._apply_membership_gradients(self.learning_rate)
        return loss, state

    @staticmethod
    def _prepare_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensure X, y are float arrays and y is 2D (n, 1) if originally 1D."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return X, y
