"""Base classes and interfaces for ANFIS trainers.

Defines the shared training loop and contracts used by all optimizers. Concrete
trainers specialize the ``train_step`` (and related helpers) while the base
class takes care of batching, epoch bookkeeping, optional validation, and
logging.

Model contract expected by trainers:
- For pure backprop trainers (e.g., SGD/Adam): the model must provide
  ``reset_gradients()``, ``forward(X)``, ``backward(dL_dy)``, and
  ``update_parameters(lr)``.
- For the HybridTrainer, the model must expose the usual ANFIS layers
  (``membership_layer``, ``rule_layer``, ``normalization_layer``,
  ``consequent_layer``) to build the least-squares system internally.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np

from ..model import TSKANFIS, TrainingHistory, TSKANFISClassifier

ModelLike: TypeAlias = TSKANFIS | TSKANFISClassifier

__all__ = ["BaseTrainer", "ModelLike"]


class BaseTrainer(ABC):
    """Shared training loop for ANFIS trainers."""

    def fit(
        self,
        model: ModelLike,
        X: np.ndarray,
        y: np.ndarray,
        *,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        validation_frequency: int = 1,
    ) -> TrainingHistory:
        """Train ``model`` on ``(X, y)`` and optionally evaluate on validation data.

        Returns a dictionary containing the per-epoch training losses and, when
        ``validation_data`` is provided, the validation losses (aligned with the
        training epochs; epochs without validation are recorded as ``None``).
        """
        if validation_frequency < 1:
            raise ValueError("validation_frequency must be >= 1")

        X_train, y_train = self._prepare_training_data(model, X, y)
        state = self.init_state(model, X_train, y_train)

        prepared_val: tuple[np.ndarray, np.ndarray] | None = None
        if validation_data is not None:
            prepared_val = self._prepare_validation_data(model, *validation_data)

        epochs = int(getattr(self, "epochs", 1))
        batch_size = getattr(self, "batch_size", None)
        shuffle = bool(getattr(self, "shuffle", True))
        verbose = bool(getattr(self, "verbose", False))

        train_history: list[float] = []
        val_history: list[float | None] = [] if prepared_val is not None else []

        n_samples = X_train.shape[0]
        for epoch_idx in range(epochs):
            epoch_losses: list[float] = []
            if batch_size is None:
                loss, state = self.train_step(model, X_train, y_train, state)
                epoch_losses.append(float(loss))
            else:
                indices = np.arange(n_samples)
                if shuffle:
                    np.random.shuffle(indices)
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]
                    loss, state = self.train_step(
                        model,
                        X_train[batch_idx],
                        y_train[batch_idx],
                        state,
                    )
                    epoch_losses.append(float(loss))

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            train_history.append(epoch_loss)

            val_loss: float | None = None
            if prepared_val is not None:
                if (epoch_idx + 1) % validation_frequency == 0:
                    X_val, y_val = prepared_val
                    val_loss = float(self.compute_loss(model, X_val, y_val))
                val_history.append(val_loss)

            self._log_epoch(epoch_idx, epoch_loss, val_loss, verbose)

        result: TrainingHistory = {"train": train_history}
        if prepared_val is not None:
            result["val"] = val_history
        return result

    @abstractmethod
    def init_state(self, model: ModelLike, X: np.ndarray, y: np.ndarray) -> Any:  # pragma: no cover - abstract
        """Initialize and return any optimizer-specific state.

        Called once before training begins. Trainers that don't require state may
        return None.

        Parameters:
            model: The model to be trained.
            X (np.ndarray): The full training inputs.
            y (np.ndarray): The full training targets.

        Returns:
            Any: Optimizer state (or None) to be threaded through ``train_step``.
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self, model: ModelLike, Xb: np.ndarray, yb: np.ndarray, state: Any
    ) -> tuple[float, Any]:  # pragma: no cover - abstract
        """Perform a single training step on a batch and return (loss, new_state).

        Parameters:
            model: The model to be trained.
            Xb (np.ndarray): A batch of inputs.
            yb (np.ndarray): A batch of targets.
            state: Optimizer state produced by ``init_state``.

        Returns:
            tuple[float, Any]: The batch loss and the updated optimizer state.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, model: ModelLike, X: np.ndarray, y: np.ndarray) -> float:  # pragma: no cover - abstract
        """Compute loss for the provided data without mutating the model."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_training_data(self, model: ModelLike, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("Target array must have same number of rows as X")
        return X_arr, y_arr

    def _prepare_validation_data(
        self,
        model: ModelLike,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._prepare_training_data(model, X_val, y_val)

    def _log_epoch(
        self,
        epoch_idx: int,
        train_loss: float,
        val_loss: float | None,
        verbose: bool,
    ) -> None:
        if not verbose:
            return
        logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        message = f"Epoch {epoch_idx + 1} - train_loss: {train_loss:.6f}"
        if val_loss is not None:
            message += f" - val_loss: {val_loss:.6f}"
        logger.info(message)
