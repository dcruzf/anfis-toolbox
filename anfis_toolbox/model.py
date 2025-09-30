"""ANFIS Model Implementation.

This module implements the complete Adaptive Neuro-Fuzzy Inference System (ANFIS)
model that combines all the individual layers into a unified architecture.
"""

import logging

import numpy as np

from .layers import ClassificationConsequentLayer, ConsequentLayer, MembershipLayer, NormalizationLayer, RuleLayer
from .losses import LossFunction, resolve_loss
from .membership import MembershipFunction
from .metrics import softmax

# Setup logger for ANFIS
logger = logging.getLogger(__name__)


class TSKANFIS:
    """Adaptive Neuro-Fuzzy Inference System (legacy TSK ANFIS) model.

    This class implements the complete ANFIS architecture with 4 layers:
    1. MembershipLayer: Fuzzification of inputs
    2. RuleLayer: Computing rule strengths using T-norm
    3. NormalizationLayer: Normalizing rule weights
    4. ConsequentLayer: Computing final output using TSK model

    The model supports both forward and backward passes for training,
    and provides a clean interface for prediction and parameter updates.

    Attributes:
        input_mfs (dict): Dictionary mapping input names to membership functions.
        membership_layer (MembershipLayer): Layer 1 - Fuzzification.
        rule_layer (RuleLayer): Layer 2 - Rule strength computation.
        normalization_layer (NormalizationLayer): Layer 3 - Weight normalization.
        consequent_layer (ConsequentLayer): Layer 4 - Final output computation.
        input_names (list): List of input variable names.
        n_inputs (int): Number of input variables.
        n_rules (int): Number of fuzzy rules.
    """

    def __init__(self, input_mfs: dict[str, list[MembershipFunction]]):
        """Initializes the ANFIS model with input membership functions.

        Parameters:
            input_mfs (dict): Dictionary mapping input names to lists of membership functions.
                             Format: {input_name: [MembershipFunction, ...]}

        Example:
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

        # Calculate total number of rules (Cartesian product)
        self.n_rules = np.prod(mf_per_input)

        # Initialize all layers
        self.membership_layer = MembershipLayer(input_mfs)
        self.rule_layer = RuleLayer(self.input_names, mf_per_input)
        self.normalization_layer = NormalizationLayer()
        self.consequent_layer = ConsequentLayer(self.n_rules, self.n_inputs)

    @property
    def membership_functions(self) -> dict[str, list[MembershipFunction]]:
        """Alias for input_mfs to provide a standardized interface.

        Returns:
            dict: Dictionary mapping input names to lists of membership functions.
        """
        return self.input_mfs

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs forward pass through the entire ANFIS model.

        Parameters:
            x (np.ndarray): Input data with shape (batch_size, n_inputs).

        Returns:
            np.ndarray: Model output with shape (batch_size, 1).
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
        """Performs backward pass through the entire ANFIS model.

        This method propagates gradients from the output back through all layers,
        updating the gradients in membership functions and consequent parameters.

        Parameters:
            dL_dy (np.ndarray): Gradient of loss with respect to model output.
                               Shape: (batch_size, 1)

        Returns:
            None: Gradients are accumulated in the respective layers and functions.
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
        """Makes predictions using the trained ANFIS model.

        Parameters:
            x (np.ndarray): Input data with shape (batch_size, n_inputs).

        Returns:
            np.ndarray: Predictions with shape (batch_size, 1).
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
        """Resets all gradients in the model to zero.

        This should be called before each training step to clear
        accumulated gradients from previous iterations.

        Returns:
            None
        """
        # Reset membership function gradients
        self.membership_layer.reset()

        # Reset consequent layer gradients
        self.consequent_layer.reset()

    def get_parameters(self) -> dict[str, np.ndarray]:
        """Retrieves all trainable parameters from the model.

        Returns:
            dict: Dictionary containing all model parameters:
                - 'membership': Dict with membership function parameters
                - 'consequent': Consequent layer parameters
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
        """Sets model parameters from a dictionary.

        Parameters:
            parameters (dict): Dictionary containing model parameters with same
                             format as returned by get_parameters().

        Returns:
            None
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
        """Retrieves all gradients from the model.

        Returns:
            dict: Dictionary containing all model gradients:
                - 'membership': Dict with membership function gradients
                - 'consequent': Consequent layer gradients
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
        """Updates model parameters using accumulated gradients (gradient descent).

        Parameters:
            learning_rate (float): Learning rate for parameter updates.

        Returns:
            None
        """
        # Update consequent layer parameters
        self.consequent_layer.parameters -= learning_rate * self.consequent_layer.gradients

        # Update membership function parameters
        for name in self.input_names:
            for mf in self.input_mfs[name]:
                for param_name, gradient in mf.gradients.items():
                    mf.parameters[param_name] -= learning_rate * gradient

    def _apply_membership_gradients(self, learning_rate: float) -> None:
        """Apply gradient descent update to membership function parameters only."""
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
        verbose: bool = True,
        trainer: None | object = None,
    ) -> list[float]:
        """Trains the ANFIS model.

        If a trainer is provided (see `anfis_toolbox.optim`), delegates training to it,
        preserving a scikit-learn-style `fit(X, y)` entry point. If no trainer is
        provided, uses a default SGDTrainer from `anfis_toolbox.optim` configured with
        the given `epochs`, `learning_rate`, and `verbose`.
        """
        if trainer is None:
            # Lazy import to avoid unnecessary dependency at module import time
            from .optim import HybridTrainer

            trainer = HybridTrainer(learning_rate=learning_rate, epochs=epochs, verbose=verbose)

        # Delegate training to the provided or default trainer
        return trainer.fit(self, x, y)

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

    Produces per-class logits aggregated from per-rule linear consequents and
    uses cross-entropy loss during training.
    """

    def __init__(self, input_mfs: dict[str, list[MembershipFunction]], n_classes: int, random_state: int | None = None):
        """Initialize the ANFIS model for classification.

        Args:
            input_mfs (dict[str, list[MembershipFunction]]):
                Dictionary mapping input variable names to lists of their associated membership functions.
            n_classes (int):
                Number of output classes. Must be greater than or equal to 2.
            random_state (int | None): Random seed for parameter initialization.

        Raises:
            ValueError: If n_classes is less than 2.

        Attributes:
            input_mfs (dict[str, list[MembershipFunction]]): Membership functions for each input.
            input_names (list[str]): Names of input variables.
            n_inputs (int): Number of input variables.
            n_classes (int): Number of output classes.
            n_rules (int): Number of fuzzy rules, computed as the product of membership functions per input.
            membership_layer (MembershipLayer): Layer for computing membership degrees.
            rule_layer (RuleLayer): Layer for rule evaluation.
            normalization_layer (NormalizationLayer): Layer for normalizing rule strengths.
            consequent_layer (ClassificationConsequentLayer): Layer for computing class outputs.
        """
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.input_mfs = input_mfs
        self.input_names = list(input_mfs.keys())
        self.n_inputs = len(input_mfs)
        self.n_classes = int(n_classes)
        mf_per_input = [len(mfs) for mfs in input_mfs.values()]
        self.n_rules = int(np.prod(mf_per_input))
        self.membership_layer = MembershipLayer(input_mfs)
        self.rule_layer = RuleLayer(self.input_names, mf_per_input)
        self.normalization_layer = NormalizationLayer()
        self.consequent_layer = ClassificationConsequentLayer(
            self.n_rules, self.n_inputs, self.n_classes, random_state=random_state
        )

    @property
    def membership_functions(self) -> dict[str, list[MembershipFunction]]:
        """Returns the membership functions used in the model."""
        return self.input_mfs

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs a forward pass through the network."""
        membership_outputs = self.membership_layer.forward(x)
        rule_strengths = self.rule_layer.forward(membership_outputs)
        normalized_weights = self.normalization_layer.forward(rule_strengths)
        logits = self.consequent_layer.forward(x, normalized_weights)  # (b, k)
        return logits

    def backward(self, dL_dlogits: np.ndarray):
        """Backpropagates the gradients through the network."""
        dL_dnorm_w, _ = self.consequent_layer.backward(dL_dlogits)
        dL_dw = self.normalization_layer.backward(dL_dnorm_w)
        gradients = self.rule_layer.backward(dL_dw)
        self.membership_layer.backward(gradients)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predicts the class probabilities for the given input."""
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
        """Predicts the class labels for the given input."""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

    def reset_gradients(self):
        """Resets the gradients of the model's layers."""
        self.membership_layer.reset()
        self.consequent_layer.reset()

    def get_parameters(self) -> dict[str, np.ndarray]:
        """Retrieves the parameters of the model.

        Returns:
            dict[str, np.ndarray]: A dictionary containing:
                - "membership": A nested dictionary where each input name maps to a list of parameter arrays
                  for its associated membership functions.
                - "consequent": An array of parameters for the consequent layer.
        """
        params = {"membership": {}, "consequent": self.consequent_layer.parameters.copy()}
        for name in self.input_names:
            params["membership"][name] = []
            for mf in self.input_mfs[name]:
                params["membership"][name].append(mf.parameters.copy())
        return params

    def set_parameters(self, parameters: dict[str, np.ndarray]):
        """Sets the parameters for the ANFIS model's layers.

        Parameters
        ----------
        parameters : dict[str, np.ndarray]
            A dictionary containing parameter arrays for the model layers.
            - "consequent": Parameters for the consequent layer.
            - "membership": Dictionary mapping input names to lists of membership function parameters.
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
        """Computes and returns the gradients of the model parameters.

        Returns:
            dict[str, np.ndarray]: A dictionary containing gradients for both membership functions
            and consequent layer parameters.
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
        verbose: bool = True,
        trainer: None | object = None,
        loss: LossFunction | str | None = None,
    ) -> list[float]:
        """Train the classifier using a trainer with configurable loss.

        When ``trainer`` is not provided, defaults to ``AdamTrainer`` minimizing
        categorical cross-entropy. Targets may be provided as integer labels or
        one-hot arrays; the configured loss takes care of the conversion.
        """
        # Resolve loss: default to cross-entropy for classification when unspecified
        if loss is None:
            resolved_loss = resolve_loss("cross_entropy")
        else:
            resolved_loss = resolve_loss(loss)

        if trainer is None:
            from .optim import AdamTrainer

            trainer = AdamTrainer(
                learning_rate=learning_rate,
                epochs=epochs,
                verbose=verbose,
                loss=resolved_loss,
            )
        elif hasattr(trainer, "loss"):
            trainer.loss = resolved_loss

        return trainer.fit(self, X, y)

    def __repr__(self) -> str:
        """Return a string representation of the ANFISClassifier.

        Returns:
            str: A formatted string describing the classifier's configuration.
        """
        return f"TSKANFISClassifier(n_inputs={self.n_inputs}, n_rules={self.n_rules}, n_classes={self.n_classes})"


# Backwards compatibility alias: legacy low-level TSK ANFIS engine
ANFIS = TSKANFIS
ANFISClassifier = TSKANFISClassifier
