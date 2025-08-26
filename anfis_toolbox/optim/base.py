"""Base classes and interfaces for ANFIS trainers.

Defines an abstract base class that all trainers should inherit from, ensuring a
consistent ``fit(model, X, y) -> list[float]`` interface.

Model contract expected by trainers:
- For pure backprop trainers (e.g., SGD/Adam): the model must provide
  ``reset_gradients()``, ``forward(X)``, ``backward(dL_dy)``, and
  ``update_parameters(lr)``.
- For the HybridTrainer, the model must expose the usual ANFIS layers
  (``membership_layer``, ``rule_layer``, ``normalization_layer``,
  ``consequent_layer``) to build the least-squares system internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTrainer(ABC):
    """Abstract base class for ANFIS trainers.

    Subclasses must implement ``fit`` and return a list of per-epoch loss values.
    """

    @abstractmethod
    def fit(self, model, X: np.ndarray, y: np.ndarray) -> list[float]:  # pragma: no cover - abstract
        """Train the given model on (X, y).

        Parameters:
            model: ANFIS-like model instance providing the methods required by
                   the specific trainer (see module docstring).
            X (np.ndarray): Input array of shape (n_samples, n_features).
            y (np.ndarray): Target array of shape (n_samples,) or (n_samples, 1)
                            for regression; shape may vary for classification
                            depending on the trainer being used.

        Returns:
            list[float]: Sequence of loss values, one per epoch.
        """
        raise NotImplementedError
