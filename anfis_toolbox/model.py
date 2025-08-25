"""ANFIS Model Implementation.

This module implements the complete Adaptive Neuro-Fuzzy Inference System (ANFIS)
model that combines all the individual layers into a unified architecture.
"""

import logging

import numpy as np

from .layers import ConsequentLayer, MembershipLayer, NormalizationLayer, RuleLayer
from .membership import MembershipFunction

# Setup logger for ANFIS
logger = logging.getLogger(__name__)


class ANFIS:
    """Adaptive Neuro-Fuzzy Inference System (ANFIS) Model.

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
            f"ANFIS Model:\n"
            f"  - Inputs: {self.n_inputs} ({', '.join(self.input_names)})\n"
            f"  - Rules: {self.n_rules}\n"
            f"  - Membership Functions: {[len(mfs) for mfs in self.input_mfs.values()]}\n"
            f"  - Parameters: \
                    {sum(len(mfs) * 2 for mfs in self.input_mfs.values()) + self.n_rules * (self.n_inputs + 1)}"
        )

    def __repr__(self) -> str:
        """Returns detailed representation of the ANFIS model."""
        return f"ANFIS(n_inputs={self.n_inputs}, n_rules={self.n_rules})"
