from abc import ABC, abstractmethod

import numpy as np


class MembershipFunction(ABC):
    """Abstract base class for membership functions.

    This class defines the interface that all membership functions must implement
    in the ANFIS system. It provides common functionality for parameter management,
    gradient computation, and forward/backward propagation.

    Attributes:
        parameters (dict): Dictionary containing the function's parameters.
        gradients (dict): Dictionary containing gradients for each parameter.
        last_input (np.ndarray): Last input processed by the function.
        last_output (np.ndarray): Last output computed by the function.
    """

    def __init__(self):
        """Initializes the membership function with empty parameters and gradients."""
        self.parameters = {}
        self.gradients = {}
        self.last_input = None
        self.last_output = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass of the membership function.

        Parameters:
            x (np.ndarray): Input array for which the membership values are to be computed.

        Returns:
            np.ndarray: Output array containing the computed membership values.
        """
        pass  # pragma: no cover

    @abstractmethod
    def backward(self, dL_dy: np.ndarray):
        """Performs the backward pass for the membership function during backpropagation.

        Parameters:
            dL_dy (np.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            None
        """
        pass  # pragma: no cover

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Calls the forward method to compute membership values.

        Parameters:
            x (np.ndarray): Input array for which the membership values are to be computed.

        Returns:
            np.ndarray: Output array containing the computed membership values.
        """
        return self.forward(x)

    def reset(self):
        """Resets the membership function to its initial state.

        Returns:
            None
        """
        self.gradients = dict.fromkeys(self.parameters, 0.0)
        self.last_input = None
        self.last_output = None


class GaussianMF(MembershipFunction):
    """Gaussian Membership Function.

    Implements a Gaussian (bell-shaped) membership function using the formula:
    μ(x) = exp(-((x - mean)² / (2 * sigma²)))

    This function is commonly used in fuzzy logic systems due to its smooth
    and differentiable properties.
    """

    def __init__(self, mean: float = 0.0, sigma: float = 1.0):
        """Initializes the Gaussian membership function with mean and standard deviation.

        Parameters:
            mean (float): Mean of the Gaussian function. Controls the center position. Defaults to 0.0.
            sigma (float): Standard deviation of the Gaussian function. Controls the width. Defaults to 1.0.
        """
        super().__init__()
        self.parameters = {"mean": mean, "sigma": sigma}
        # Initialize gradients to zero for all parameters
        self.gradients = dict.fromkeys(self.parameters.keys(), 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the Gaussian membership values for the input x.

        Parameters:
            x (np.ndarray): Input array for which the membership values are to be computed.

        Returns:
            np.ndarray: Output array containing the Gaussian membership values.
        """
        mean = self.parameters["mean"]
        sigma = self.parameters["sigma"]
        self.last_input = x
        self.last_output = np.exp(-((x - mean) ** 2) / (2 * sigma**2))
        return self.last_output

    def backward(self, dL_dy: np.ndarray):
        """Computes the gradients for the parameters based on the loss gradient.

        Parameters:
            dL_dy (np.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            None
        """
        mean = self.parameters["mean"]
        sigma = self.parameters["sigma"]

        x = self.last_input
        y = self.last_output

        z = (x - mean) / sigma

        # Derivatives of the Gaussian function
        dy_dmean = y * z / sigma
        dy_dsigma = y * (z**2) / sigma

        # Gradient with respect to mean
        dL_dmean = np.sum(dL_dy * dy_dmean)

        # Gradient with respect to sigma
        dL_dsigma = np.sum(dL_dy * dy_dsigma)

        # Update gradients
        self.gradients["mean"] += dL_dmean
        self.gradients["sigma"] += dL_dsigma
