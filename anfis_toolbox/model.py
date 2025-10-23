"""ANFIS Model Implementation.

This module implements the complete Adaptive Neuro-Fuzzy Inference System (ANFIS)
model that combines all the individual layers into a unified architecture.
"""

import logging
from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np

from .layers import ClassificationConsequentLayer, ConsequentLayer, MembershipLayer, NormalizationLayer, RuleLayer
from .losses import LossFunction, resolve_loss
from .membership import MembershipFunction
from .metrics import softmax
from .optim.base import BaseTrainer, TrainingHistory


@runtime_checkable
class TrainerProtocol(Protocol):
    """Minimal interface required for external trainers."""

    def fit(self, model: Any, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> TrainingHistory:
        """Train ``model`` using ``X`` and ``y`` and return a history mapping."""


TrainerLike: TypeAlias = BaseTrainer | TrainerProtocol

# Setup logger for ANFIS
logger = logging.getLogger(__name__)


class TSKANFIS:
    """Adaptive Neuro-Fuzzy Inference System (legacy TSK ANFIS) model.

    Implements the classic 4-layer ANFIS architecture:

    1) MembershipLayer — fuzzification of inputs
    2) RuleLayer — rule strength computation (T-norm)
    3) NormalizationLayer — weight normalization
    4) ConsequentLayer — final output via a TSK model

    Supports forward/backward passes for training, parameter access/update,
    and a simple prediction API.

    Attributes:
        input_mfs (dict[str, list[MembershipFunction]]): Mapping from input name
            to its list of membership functions.
        membership_layer (MembershipLayer): Layer 1 — fuzzification.
        rule_layer (RuleLayer): Layer 2 — rule strength computation.
        normalization_layer (NormalizationLayer): Layer 3 — weight normalization.
        consequent_layer (ConsequentLayer): Layer 4 — final TSK output.
        input_names (list[str]): Ordered list of input variable names.
        n_inputs (int): Number of input variables (features).
        n_rules (int): Number of fuzzy rules used by the system.
    """

    def __init__(
        self,
        input_mfs: dict[str, list[MembershipFunction]],
        rules: Sequence[Sequence[int]] | None = None,
    ):
        """Initialize the ANFIS model.

        Args:
            input_mfs (dict[str, list[MembershipFunction]]): Mapping from input
                name to a list of membership functions. Example:
                ``{"x1": [GaussianMF(0,1), ...], "x2": [...]}``.
            rules: Optional explicit set of rules, each specifying one membership index per
                input. When ``None``, the Cartesian product of all membership functions is used.

        Examples:
            >>> from anfis_toolbox.membership import GaussianMF
            >>> input_mfs = {
            ...     'x1': [GaussianMF(0, 1), GaussianMF(1, 1)],
            ...     'x2': [GaussianMF(0, 1), GaussianMF(1, 1)]
            ... }
            >>> model = ANFIS(input_mfs)
        """
        self.input_mfs = input_mfs
        self.input_names = list(input_mfs.keys())
        self.n_inputs = len(input_mfs)

        # Calculate number of membership functions per input
        mf_per_input = [len(mfs) for mfs in input_mfs.values()]

        # Initialize all layers
        self.membership_layer = MembershipLayer(input_mfs)
        self.rule_layer = RuleLayer(self.input_names, mf_per_input, rules=rules)
        self.n_rules = self.rule_layer.n_rules
        self.normalization_layer = NormalizationLayer()
        self.consequent_layer = ConsequentLayer(self.n_rules, self.n_inputs)

    @property
    def membership_functions(self) -> dict[str, list[MembershipFunction]]:
        """Return the membership functions grouped by input.

        Returns:
            dict[str, list[MembershipFunction]]: Mapping from input name to
            its list of membership functions.
        """
        return self.input_mfs

    @property
    def rules(self) -> list[tuple[int, ...]]:
        """Return the fuzzy rule definitions used by the model."""
        return list(self.rule_layer.rules)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run a forward pass through the model.

        Args:
            x (np.ndarray): Input array of shape ``(batch_size, n_inputs)``.

        Returns:
            np.ndarray: Output array of shape ``(batch_size, 1)``.
        """
        # Layer 1: Fuzzification - convert crisp inputs to membership degrees
        membership_outputs = self.membership_layer.forward(x)

        # Layer 2: Rule strength computation using T-norm (product)
        rule_strengths = self.rule_layer.forward(membership_outputs)

        # Layer 3: Normalization - ensure rule weights sum to 1.0
        normalized_weights = self.normalization_layer.forward(rule_strengths)

        # Layer 4: Consequent computation and final output
        output = self.consequent_layer.forward(x, normalized_weights)

        return output

    def backward(self, dL_dy: np.ndarray):
        """Run a backward pass through all layers.

        Propagates gradients from the output back through all layers and stores
        parameter gradients for a later update step.

        Args:
            dL_dy (np.ndarray): Gradient of the loss w.r.t. the model output,
                shape ``(batch_size, 1)``.
        """
        # Backward pass through Layer 4: Consequent layer
        dL_dnorm_w, _ = self.consequent_layer.backward(dL_dy)

        # Backward pass through Layer 3: Normalization layer
        dL_dw = self.normalization_layer.backward(dL_dnorm_w)

        # Backward pass through Layer 2: Rule layer
        gradients = self.rule_layer.backward(dL_dw)

        # Backward pass through Layer 1: Membership layer
        self.membership_layer.backward(gradients)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the current model parameters.

        Accepts Python lists, 1D or 2D arrays and coerces to the expected shape.

        Args:
            x (np.ndarray | list[float]): Input data. If 1D, must have
                exactly ``n_inputs`` elements; if 2D, must be
                ``(batch_size, n_inputs)``.

        Returns:
            np.ndarray: Predictions of shape ``(batch_size, 1)``.

        Raises:
            ValueError: If input dimensionality or feature count does not match
                the model configuration.
        """
        # Accept Python lists or 1D arrays by coercing to correct 2D shape
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            # Single sample; ensure feature count matches
            if x_arr.size != self.n_inputs:
                raise ValueError(f"Expected {self.n_inputs} features, got {x_arr.size} in 1D input")
            x_arr = x_arr.reshape(1, self.n_inputs)
        elif x_arr.ndim == 2:
            # Validate feature count
            if x_arr.shape[1] != self.n_inputs:
                raise ValueError(f"Expected input with {self.n_inputs} features, got {x_arr.shape[1]}")
        else:
            raise ValueError("Expected input with shape (batch_size, n_inputs)")

        return self.forward(x_arr)

    def reset_gradients(self):
        """Reset all accumulated gradients to zero.

        Call this before each optimization step to avoid mixing gradients
        across iterations.
        """
        # Reset membership function gradients
        self.membership_layer.reset()

        # Reset consequent layer gradients
        self.consequent_layer.reset()

    def get_parameters(self) -> dict[str, np.ndarray]:
        """Return a snapshot of all trainable parameters.

        Returns:
                dict[str, np.ndarray | dict]:
                        Dictionary with two entries:

                        - ``"membership"``: dict mapping input name to a list of MF
                            parameter dicts (one per membership function).
                        - ``"consequent"``: numpy array with consequent parameters.
        """
        parameters = {"membership": {}, "consequent": self.consequent_layer.parameters.copy()}

        # Extract membership function parameters
        for name in self.input_names:
            parameters["membership"][name] = []
            for mf in self.input_mfs[name]:
                mf_params = mf.parameters.copy()
                parameters["membership"][name].append(mf_params)

        return parameters

    def set_parameters(self, parameters: dict[str, np.ndarray]):
        """Load parameters into the model.

        Args:
            parameters (dict[str, np.ndarray | dict]): Dictionary with the same
                structure as returned by :meth:`get_parameters`.
        """
        # Set consequent layer parameters
        if "consequent" in parameters:
            self.consequent_layer.parameters = parameters["consequent"].copy()

        # Set membership function parameters
        if "membership" in parameters:
            membership_params = parameters["membership"]
            for name in self.input_names:
                mf_params_list = membership_params.get(name)
                if not mf_params_list:
                    continue
                # Only update up to the available MFs for this input
                for mf, mf_params in zip(self.input_mfs[name], mf_params_list, strict=False):
                    mf.parameters = mf_params.copy()

    def get_gradients(self) -> dict[str, np.ndarray]:
        """Return the latest computed gradients.

        Returns:
                dict[str, np.ndarray | dict]: Dictionary with two entries:

                - ``"membership"``: dict mapping input name to a list of MF
                    gradient dicts (one per membership function).
                - ``"consequent"``: numpy array with consequent gradients.
        """
        gradients = {"membership": {}, "consequent": self.consequent_layer.gradients.copy()}

        # Extract membership function gradients
        for name in self.input_names:
            gradients["membership"][name] = []
            for mf in self.input_mfs[name]:
                mf_grads = mf.gradients.copy()
                gradients["membership"][name].append(mf_grads)

        return gradients

    def update_parameters(self, learning_rate: float):
        """Apply a single gradient descent update step.

        Args:
            learning_rate (float): Step size used to update parameters.
        """
        # Update consequent layer parameters
        self.consequent_layer.parameters -= learning_rate * self.consequent_layer.gradients

        # Update membership function parameters
        for name in self.input_names:
            for mf in self.input_mfs[name]:
                for param_name, gradient in mf.gradients.items():
                    mf.parameters[param_name] -= learning_rate * gradient

    def _apply_membership_gradients(self, learning_rate: float) -> None:
        """Apply gradient descent to membership function parameters only.

        Args:
            learning_rate (float): Step size for MF parameters.
        """
        for name in self.input_names:
            for mf in self.input_mfs[name]:
                for param_name, gradient in mf.gradients.items():
                    mf.parameters[param_name] -= learning_rate * gradient

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = False,
        trainer: TrainerLike | None = None,
        *,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        validation_frequency: int = 1,
    ) -> TrainingHistory:
        """Train the ANFIS model.

        If a trainer is provided (see ``anfis_toolbox.optim``), delegate training
        to it while preserving a scikit-learn-style ``fit(X, y)`` entry point. If
        no trainer is provided, a default ``HybridTrainer`` is used with the given
        hyperparameters.

        Args:
            x (np.ndarray): Training inputs of shape ``(n_samples, n_inputs)``.
            y (np.ndarray): Training targets of shape ``(n_samples, 1)`` for
                regression.
            epochs (int, optional): Number of epochs. Defaults to ``100``.
            learning_rate (float, optional): Learning rate. Defaults to ``0.01``.
            verbose (bool, optional): Whether to log progress. Defaults to ``False``.
            trainer (TrainerLike | None, optional): External trainer implementing
                ``fit(model, X, y)``. Defaults to ``None``.
            validation_data (tuple[np.ndarray, np.ndarray] | None, optional): Optional
                validation inputs and targets evaluated according to ``validation_frequency``.
            validation_frequency (int, optional): Evaluate validation loss every N epochs.

        Returns:
            TrainingHistory: Dictionary with ``"train"`` losses and optional ``"val"`` losses.
        """
        if trainer is None:
            # Lazy import to avoid unnecessary dependency at module import time
            from .optim import HybridTrainer

            trainer_instance: TrainerLike = HybridTrainer(
                learning_rate=learning_rate,
                epochs=epochs,
                verbose=verbose,
            )
        else:
            trainer_instance = trainer
            if not isinstance(trainer_instance, (BaseTrainer, TrainerProtocol)):
                raise TypeError("trainer must implement fit(model, X, y)")

        # Delegate training to the provided or default trainer
        fit_kwargs: dict[str, Any] = {}
        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data
        if validation_frequency != 1 or validation_data is not None:
            fit_kwargs["validation_frequency"] = validation_frequency

        history = trainer_instance.fit(self, x, y, **fit_kwargs)
        if not isinstance(history, dict):
            raise TypeError("Trainer.fit must return a TrainingHistory dictionary")
        return history

    def __str__(self) -> str:
        """Returns string representation of the ANFIS model."""
        return (
            f"TSKANFIS Model:\n"
            f"  - Inputs: {self.n_inputs} ({', '.join(self.input_names)})\n"
            f"  - Rules: {self.n_rules}\n"
            f"  - Membership Functions: {[len(mfs) for mfs in self.input_mfs.values()]}\n"
            f"  - Parameters: \
                    {sum(len(mfs) * 2 for mfs in self.input_mfs.values()) + self.n_rules * (self.n_inputs + 1)}"
        )

    def __repr__(self) -> str:
        """Returns detailed representation of the ANFIS model."""
        return f"TSKANFIS(n_inputs={self.n_inputs}, n_rules={self.n_rules})"


class TSKANFISClassifier:
    """Adaptive Neuro-Fuzzy classifier with a softmax head (TSK variant).

    Aggregates per-rule linear consequents into per-class logits and trains
    with cross-entropy loss.
    """

    def __init__(
        self,
        input_mfs: dict[str, list[MembershipFunction]],
        n_classes: int,
        random_state: int | None = None,
        rules: Sequence[Sequence[int]] | None = None,
    ):
        """Initialize the ANFIS model for classification.

        Args:
            input_mfs (dict[str, list[MembershipFunction]]): Mapping from input
                variable name to its list of membership functions.
            n_classes (int): Number of output classes (>= 2).
            random_state (int | None): Optional random seed for parameter init.
            rules (Sequence[Sequence[int]] | None): Optional explicit rule definitions
                where each inner sequence lists the membership-function index per input.
                When ``None``, all combinations are used.

        Raises:
            ValueError: If ``n_classes < 2``.

        Attributes:
            input_mfs (dict[str, list[MembershipFunction]]): Membership functions per input.
            input_names (list[str]): Input variable names.
            n_inputs (int): Number of input variables.
            n_classes (int): Number of classes.
            n_rules (int): Number of fuzzy rules (product of MFs per input).
            membership_layer (MembershipLayer): Computes membership degrees.
            rule_layer (RuleLayer): Evaluates rule activations.
            normalization_layer (NormalizationLayer): Normalizes rule strengths.
            consequent_layer (ClassificationConsequentLayer): Computes class logits.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.input_mfs = input_mfs
        self.input_names = list(input_mfs.keys())
        self.n_inputs = len(input_mfs)
        self.n_classes = int(n_classes)
        mf_per_input = [len(mfs) for mfs in input_mfs.values()]
        self.membership_layer = MembershipLayer(input_mfs)
        self.rule_layer = RuleLayer(self.input_names, mf_per_input, rules=rules)
        self.n_rules = self.rule_layer.n_rules
        self.normalization_layer = NormalizationLayer()
        self.consequent_layer = ClassificationConsequentLayer(
            self.n_rules, self.n_inputs, self.n_classes, random_state=random_state
        )

    @property
    def membership_functions(self) -> dict[str, list[MembershipFunction]]:
        """Return the membership functions grouped by input.

        Returns:
            dict[str, list[MembershipFunction]]: Mapping from input name to
            its list of membership functions.
        """
        return self.input_mfs

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run a forward pass through the classifier.

        Args:
            x (np.ndarray): Input array of shape ``(batch_size, n_inputs)``.

        Returns:
            np.ndarray: Logits of shape ``(batch_size, n_classes)``.
        """
        membership_outputs = self.membership_layer.forward(x)
        rule_strengths = self.rule_layer.forward(membership_outputs)
        normalized_weights = self.normalization_layer.forward(rule_strengths)
        logits = self.consequent_layer.forward(x, normalized_weights)  # (b, k)
        return logits

    def backward(self, dL_dlogits: np.ndarray):
        """Backpropagate gradients through all layers.

        Args:
            dL_dlogits (np.ndarray): Gradient of the loss w.r.t. logits,
                shape ``(batch_size, n_classes)``.
        """
        dL_dnorm_w, _ = self.consequent_layer.backward(dL_dlogits)
        dL_dw = self.normalization_layer.backward(dL_dnorm_w)
        gradients = self.rule_layer.backward(dL_dw)
        self.membership_layer.backward(gradients)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict per-class probabilities for the given inputs.

        Args:
            x (np.ndarray | list[float]): Inputs. If 1D, must have exactly
                ``n_inputs`` elements; if 2D, must be ``(batch_size, n_inputs)``.

        Returns:
            np.ndarray: Probabilities of shape ``(batch_size, n_classes)``.

        Raises:
            ValueError: If input dimensionality or feature count is invalid.
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            if x_arr.size != self.n_inputs:
                raise ValueError(f"Expected {self.n_inputs} features, got {x_arr.size} in 1D input")
            x_arr = x_arr.reshape(1, self.n_inputs)
        elif x_arr.ndim == 2:
            if x_arr.shape[1] != self.n_inputs:
                raise ValueError(f"Expected input with {self.n_inputs} features, got {x_arr.shape[1]}")
        else:
            raise ValueError("Expected input with shape (batch_size, n_inputs)")
        logits = self.forward(x_arr)
        return softmax(logits, axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the most likely class label for each sample.

        Args:
            x (np.ndarray | list[float]): Inputs. If 1D, must have exactly
                ``n_inputs`` elements; if 2D, must be ``(batch_size, n_inputs)``.

        Returns:
            np.ndarray: Predicted labels of shape ``(batch_size,)``.
        """
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

    def reset_gradients(self):
        """Reset gradients accumulated in the model layers to zero."""
        self.membership_layer.reset()
        self.consequent_layer.reset()

    def get_parameters(self) -> dict[str, np.ndarray]:
        """Return a snapshot of all trainable parameters.

        Returns:
                dict[str, np.ndarray | dict]: Dictionary containing:

                - ``"membership"``: nested dict mapping input name to a list of MF
                    parameter dicts.
                - ``"consequent"``: numpy array with consequent parameters.
        """
        params = {"membership": {}, "consequent": self.consequent_layer.parameters.copy()}
        for name in self.input_names:
            params["membership"][name] = []
            for mf in self.input_mfs[name]:
                params["membership"][name].append(mf.parameters.copy())
        return params

    @property
    def rules(self) -> list[tuple[int, ...]]:
        """Return the fuzzy rule definitions used by the classifier."""
        return list(self.rule_layer.rules)

    def set_parameters(self, parameters: dict[str, np.ndarray]):
        """Load parameters into the classifier.

        Args:
            parameters (dict[str, np.ndarray | dict]): Dictionary with the same
                structure as returned by :meth:`get_parameters`.
        """
        if "consequent" in parameters:
            self.consequent_layer.parameters = parameters["consequent"].copy()
        if "membership" in parameters:
            membership_params = parameters["membership"]
            for name in self.input_names:
                mf_params_list = membership_params.get(name)
                if not mf_params_list:
                    continue
                for mf, mf_params in zip(self.input_mfs[name], mf_params_list, strict=False):
                    mf.parameters = mf_params.copy()

    def get_gradients(self) -> dict[str, np.ndarray]:
        """Return the latest computed gradients.

        Returns:
                dict[str, np.ndarray | dict]: Dictionary containing:

                - ``"membership"``: nested dict mapping input name to a list of MF
                    gradient dicts.
                - ``"consequent"``: numpy array with consequent gradients.
        """
        grads = {"membership": {}, "consequent": self.consequent_layer.gradients.copy()}
        for name in self.input_names:
            grads["membership"][name] = []
            for mf in self.input_mfs[name]:
                grads["membership"][name].append(mf.gradients.copy())
        return grads

    def update_parameters(self, learning_rate: float):
        """Updates the parameters of the model using gradient descent.

        This method applies the specified learning rate to update both the consequent layer parameters
        and the parameters of each membership function (MF) in the input layers. The update is performed
        by subtracting the product of the learning rate and the corresponding gradients from each parameter.

        Args:
            learning_rate (float): The step size used for updating the parameters during gradient descent.

        Returns:
            None
        """
        self.consequent_layer.parameters -= learning_rate * self.consequent_layer.gradients
        for name in self.input_names:
            for mf in self.input_mfs[name]:
                for param_name, gradient in mf.gradients.items():
                    mf.parameters[param_name] -= learning_rate * gradient

    def _apply_membership_gradients(self, learning_rate: float) -> None:
        """Apply gradient descent to membership function parameters only.

        Args:
            learning_rate (float): Step size for MF parameters.
        """
        for name in self.input_names:
            for mf in self.input_mfs[name]:
                for param_name, gradient in mf.gradients.items():
                    mf.parameters[param_name] -= learning_rate * gradient

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = False,
        trainer: TrainerLike | None = None,
        loss: LossFunction | str | None = None,
        *,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        validation_frequency: int = 1,
    ) -> TrainingHistory:
        """Fits the ANFIS model to the provided training data using the specified optimization strategy.

        Parameters:
            X (np.ndarray): Input features for training.
            y (np.ndarray): Target values for training.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.
            verbose (bool, optional): If True, prints training progress. Defaults to False.
            trainer (TrainerLike | None, optional): Custom trainer instance. If None,
                uses AdamTrainer. Defaults to None.
            loss (LossFunction, str, or None, optional): Loss function to use.
                If None, defaults to cross-entropy for classification.
            validation_data (tuple[np.ndarray, np.ndarray] | None, optional): Optional validation dataset.
            validation_frequency (int, optional): Evaluate validation metrics every N epochs.

        Returns:
            TrainingHistory: Dictionary containing ``"train"`` and optionally ``"val"`` loss curves.
        """
        if loss is None:
            resolved_loss = resolve_loss("cross_entropy")
        else:
            resolved_loss = resolve_loss(loss)

        if trainer is None:
            from .optim import AdamTrainer

            trainer_instance: TrainerLike = AdamTrainer(
                learning_rate=learning_rate,
                epochs=epochs,
                verbose=verbose,
                loss=resolved_loss,
            )
        else:
            trainer_instance = trainer
            if not isinstance(trainer_instance, (BaseTrainer, TrainerProtocol)):
                raise TypeError("trainer must implement fit(model, X, y)")
            if hasattr(trainer_instance, "loss"):
                trainer_instance.loss = resolved_loss

        fit_kwargs: dict[str, Any] = {}
        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data
        if validation_frequency != 1 or validation_data is not None:
            fit_kwargs["validation_frequency"] = validation_frequency

        history = trainer_instance.fit(self, X, y, **fit_kwargs)
        if not isinstance(history, dict):
            raise TypeError("Trainer.fit must return a TrainingHistory dictionary")
        return history

    def __repr__(self) -> str:
        """Return a string representation of the ANFISClassifier.

        Returns:
            str: A formatted string describing the classifier configuration.
        """
        return f"TSKANFISClassifier(n_inputs={self.n_inputs}, n_rules={self.n_rules}, n_classes={self.n_classes})"


# Backwards compatibility alias: legacy low-level TSK ANFIS engine
ANFIS = TSKANFIS
ANFISClassifier = TSKANFISClassifier
