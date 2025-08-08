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

    def reset(self):
        """Resets gradients to zero.

        This method should be called before each training step to clear
        accumulated gradients from previous iterations.
        """
        for key in self.gradients:
            self.gradients[key] = 0.0
        self.last_input = None
        self.last_output = None

    def __str__(self) -> str:
        """Returns string representation of the Gaussian membership function."""
        return f"GaussianMF(mean={self.parameters['mean']:.3f}, sigma={self.parameters['sigma']:.3f})"

    def __repr__(self) -> str:
        """Returns detailed representation of the Gaussian membership function."""
        return f"GaussianMF(mean={self.parameters['mean']}, sigma={self.parameters['sigma']})"


class TriangularMF(MembershipFunction):
    """Triangular Membership Function.

    Implements a triangular membership function using piecewise linear segments:
    μ(x) = { 0,           x ≤ a or x ≥ c
           { (x-a)/(b-a), a < x < b
           { (c-x)/(c-b), b ≤ x < c

    This function is commonly used in fuzzy logic systems due to its simplicity,
    computational efficiency, and good linguistic interpretability.

    Parameters:
        a (float): Left base point of the triangle (lower support bound).
        b (float): Peak point of the triangle (core point where μ(x) = 1).
        c (float): Right base point of the triangle (upper support bound).

    Note:
        Parameters must satisfy: a ≤ b ≤ c for a valid triangular function.
    """

    def __init__(self, a: float, b: float, c: float):
        """Initializes the Triangular membership function with three control points.

        Parameters:
            a (float): Left base point (μ(a) = 0).
            b (float): Peak point (μ(b) = 1).
            c (float): Right base point (μ(c) = 0).

        Raises:
            ValueError: If parameters don't satisfy a ≤ b ≤ c.
        """
        super().__init__()

        # Validate parameters
        if not (a <= b <= c):
            raise ValueError(f"Triangular MF parameters must satisfy a ≤ b ≤ c, got a={a}, b={b}, c={c}")

        if a == c:
            raise ValueError("Parameters 'a' and 'c' cannot be equal (zero width triangle)")

        self.parameters = {"a": float(a), "b": float(b), "c": float(c)}
        # Initialize gradients to zero for all parameters
        self.gradients = dict.fromkeys(self.parameters.keys(), 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the triangular membership values for the input x.

        Parameters:
            x (np.ndarray): Input array for which the membership values are to be computed.

        Returns:
            np.ndarray: Output array containing the triangular membership values.
        """
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]

        self.last_input = x

        # Initialize output with zeros
        output = np.zeros_like(x)

        # Left slope: (x - a) / (b - a) for a < x < b
        if b > a:  # Avoid division by zero
            left_mask = (x > a) & (x < b)
            output[left_mask] = (x[left_mask] - a) / (b - a)

        # Peak: μ(x) = 1 at x = b
        peak_mask = x == b
        output[peak_mask] = 1.0

        # Right slope: (c - x) / (c - b) for b < x < c
        if c > b:  # Avoid division by zero
            right_mask = (x > b) & (x < c)
            output[right_mask] = (c - x[right_mask]) / (c - b)

        # Values outside [a, c] are already zero

        self.last_output = output
        return output

    def backward(self, dL_dy: np.ndarray):
        """Computes the gradients for the parameters based on the loss gradient.

        The gradients are computed analytically for the piecewise linear function:
        - ∂μ/∂a: Affects the left slope
        - ∂μ/∂b: Affects both slopes as it's the peak point
        - ∂μ/∂c: Affects the right slope

        Parameters:
            dL_dy (np.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            None: Gradients are accumulated in self.gradients.
        """
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]

        x = self.last_input

        # Initialize gradients
        dL_da = 0.0
        dL_db = 0.0
        dL_dc = 0.0

        # Left slope region: a < x < b, μ(x) = (x-a)/(b-a)
        if b > a:
            left_mask = (x > a) & (x < b)
            if np.any(left_mask):
                x_left = x[left_mask]
                dL_dy_left = dL_dy[left_mask]

                # ∂μ/∂a = -1/(b-a) for left slope
                dmu_da_left = -1.0 / (b - a)
                dL_da += np.sum(dL_dy_left * dmu_da_left)

                # ∂μ/∂b = -(x-a)/(b-a)² for left slope
                dmu_db_left = -(x_left - a) / ((b - a) ** 2)
                dL_db += np.sum(dL_dy_left * dmu_db_left)

        # Right slope region: b < x < c, μ(x) = (c-x)/(c-b)
        if c > b:
            right_mask = (x > b) & (x < c)
            if np.any(right_mask):
                x_right = x[right_mask]
                dL_dy_right = dL_dy[right_mask]

                # ∂μ/∂b = (x-c)/(c-b)² for right slope
                dmu_db_right = (x_right - c) / ((c - b) ** 2)
                dL_db += np.sum(dL_dy_right * dmu_db_right)

                # ∂μ/∂c = -1/(c-b) for right slope (derivative of (c-x)/(c-b) w.r.t. c)
                dmu_dc_right = (x_right - b) / ((c - b) ** 2)
                dL_dc += np.sum(dL_dy_right * dmu_dc_right)

        # Update gradients (accumulate for batch processing)
        self.gradients["a"] += dL_da
        self.gradients["b"] += dL_db
        self.gradients["c"] += dL_dc

    def reset(self):
        """Resets gradients to zero.

        This method should be called before each training step to clear
        accumulated gradients from previous iterations.
        """
        for key in self.gradients:
            self.gradients[key] = 0.0
        self.last_input = None
        self.last_output = None

    def __str__(self) -> str:
        """Returns string representation of the triangular membership function."""
        return f"TriangularMF(a={self.parameters['a']:.3f}, b={self.parameters['b']:.3f}, c={self.parameters['c']:.3f})"

    def __repr__(self) -> str:
        """Returns detailed representation of the triangular membership function."""
        return f"TriangularMF(a={self.parameters['a']}, b={self.parameters['b']}, c={self.parameters['c']})"
