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


class TrapezoidalMF(MembershipFunction):
    """Trapezoidal Membership Function.

    Implements a trapezoidal membership function using piecewise linear segments:
    μ(x) = { 0,           x ≤ a or x ≥ d
           { (x-a)/(b-a), a < x < b
           { 1,           b ≤ x ≤ c
           { (d-x)/(d-c), c < x < d

    This function is commonly used in fuzzy logic systems when you need a plateau
    region of full membership, providing robustness to noise and uncertainty.

    Parameters:
        a (float): Left base point of the trapezoid (lower support bound).
        b (float): Left peak point (start of plateau where μ(x) = 1).
        c (float): Right peak point (end of plateau where μ(x) = 1).
        d (float): Right base point of the trapezoid (upper support bound).

    Note:
        Parameters must satisfy: a ≤ b ≤ c ≤ d for a valid trapezoidal function.
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        """Initializes the Trapezoidal membership function with four control points.

        Parameters:
            a (float): Left base point (μ(a) = 0).
            b (float): Left peak point (μ(b) = 1, start of plateau).
            c (float): Right peak point (μ(c) = 1, end of plateau).
            d (float): Right base point (μ(d) = 0).

        Raises:
            ValueError: If parameters don't satisfy a ≤ b ≤ c ≤ d.
        """
        super().__init__()

        # Validate parameters
        if not (a <= b <= c <= d):
            raise ValueError(f"Trapezoidal MF parameters must satisfy a ≤ b ≤ c ≤ d, got a={a}, b={b}, c={c}, d={d}")

        if a == d:
            raise ValueError("Parameters 'a' and 'd' cannot be equal (zero width trapezoid)")

        self.parameters = {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}
        # Initialize gradients to zero for all parameters
        self.gradients = dict.fromkeys(self.parameters.keys(), 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the trapezoidal membership values for the input x.

        Parameters:
            x (np.ndarray): Input array for which the membership values are to be computed.

        Returns:
            np.ndarray: Output array containing the trapezoidal membership values.
        """
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]
        d = self.parameters["d"]

        self.last_input = x

        # Initialize output with zeros
        output = np.zeros_like(x)

        # Left slope: (x - a) / (b - a) for a < x < b
        if b > a:  # Avoid division by zero
            left_mask = (x > a) & (x < b)
            output[left_mask] = (x[left_mask] - a) / (b - a)

        # Plateau: μ(x) = 1 for b ≤ x ≤ c
        plateau_mask = (x >= b) & (x <= c)
        output[plateau_mask] = 1.0

        # Right slope: (d - x) / (d - c) for c < x < d
        if d > c:  # Avoid division by zero
            right_mask = (x > c) & (x < d)
            output[right_mask] = (d - x[right_mask]) / (d - c)

        # Values outside [a, d] are already zero

        self.last_output = output
        return output

    def backward(self, dL_dy: np.ndarray):
        """Computes the gradients for the parameters based on the loss gradient.

        The gradients are computed analytically for the piecewise linear function:
        - ∂μ/∂a: Affects the left slope
        - ∂μ/∂b: Affects the left slope and plateau transition
        - ∂μ/∂c: Affects the right slope and plateau transition
        - ∂μ/∂d: Affects the right slope

        Parameters:
            dL_dy (np.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            None: Gradients are accumulated in self.gradients.
        """
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]
        d = self.parameters["d"]

        x = self.last_input

        # Initialize gradients
        dL_da = 0.0
        dL_db = 0.0
        dL_dc = 0.0
        dL_dd = 0.0

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

        # Plateau region: b ≤ x ≤ c, μ(x) = 1
        # No gradients for plateau region (constant function)

        # Right slope region: c < x < d, μ(x) = (d-x)/(d-c)
        if d > c:
            right_mask = (x > c) & (x < d)
            if np.any(right_mask):
                x_right = x[right_mask]
                dL_dy_right = dL_dy[right_mask]

                # ∂μ/∂c = (x-d)/(d-c)² for right slope
                dmu_dc_right = (x_right - d) / ((d - c) ** 2)
                dL_dc += np.sum(dL_dy_right * dmu_dc_right)

                # ∂μ/∂d = (x-c)/(d-c)² for right slope (derivative of (d-x)/(d-c) w.r.t. d)
                dmu_dd_right = (x_right - c) / ((d - c) ** 2)
                dL_dd += np.sum(dL_dy_right * dmu_dd_right)

        # Update gradients (accumulate for batch processing)
        self.gradients["a"] += dL_da
        self.gradients["b"] += dL_db
        self.gradients["c"] += dL_dc
        self.gradients["d"] += dL_dd

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
        """Returns string representation of the trapezoidal membership function."""
        return (
            f"TrapezoidalMF(a={self.parameters['a']:.3f}, b={self.parameters['b']:.3f}, "
            f"c={self.parameters['c']:.3f}, d={self.parameters['d']:.3f})"
        )

    def __repr__(self) -> str:
        """Returns detailed representation of the trapezoidal membership function."""
        return (
            f"TrapezoidalMF(a={self.parameters['a']}, b={self.parameters['b']}, "
            f"c={self.parameters['c']}, d={self.parameters['d']})"
        )


class BellMF(MembershipFunction):
    """Bell-shaped (Generalized Bell) Membership Function.

    Implements a bell-shaped membership function using the formula:
    μ(x) = 1 / (1 + |((x - c) / a)|^(2b))

    This function is a generalization of the Gaussian function and provides
    more flexibility in controlling the shape through the 'b' parameter.
    It's particularly useful when you need asymmetric membership functions
    or want to fine-tune the slope characteristics.

    Parameters:
        a (float): Width parameter (positive). Controls the width of the curve.
        b (float): Slope parameter (positive). Controls the steepness of the curve.
        c (float): Center parameter. Controls the center position of the curve.

    Note:
        Parameters 'a' and 'b' must be positive for a valid bell function.
    """

    def __init__(self, a: float = 1.0, b: float = 2.0, c: float = 0.0):
        """Initializes the Bell membership function with three control parameters.

        Parameters:
            a (float): Width parameter (must be positive). Defaults to 1.0.
            b (float): Slope parameter (must be positive). Defaults to 2.0.
            c (float): Center parameter. Defaults to 0.0.

        Raises:
            ValueError: If parameters 'a' or 'b' are not positive.
        """
        super().__init__()

        # Validate parameters
        if a <= 0:
            raise ValueError(f"Parameter 'a' must be positive, got a={a}")

        if b <= 0:
            raise ValueError(f"Parameter 'b' must be positive, got b={b}")

        self.parameters = {"a": float(a), "b": float(b), "c": float(c)}
        # Initialize gradients to zero for all parameters
        self.gradients = dict.fromkeys(self.parameters.keys(), 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the bell membership values for the input x.

        Parameters:
            x (np.ndarray): Input array for which the membership values are to be computed.

        Returns:
            np.ndarray: Output array containing the bell membership values.
        """
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]

        self.last_input = x

        # Compute the bell function: μ(x) = 1 / (1 + |((x - c) / a)|^(2b))
        # To avoid numerical issues, we use the absolute value and handle edge cases

        # Compute (x - c) / a
        normalized = (x - c) / a

        # Compute |normalized|^(2b)
        # Use np.abs to handle negative values properly
        abs_normalized = np.abs(normalized)

        # Handle the case where abs_normalized is very close to zero
        with np.errstate(divide="ignore", invalid="ignore"):
            power_term = np.power(abs_normalized, 2 * b)
            # Replace any inf or nan with a very large number to make output close to 0
            power_term = np.where(np.isfinite(power_term), power_term, 1e10)

        # Compute the final result
        output = 1.0 / (1.0 + power_term)

        self.last_output = output
        return output

    def backward(self, dL_dy: np.ndarray):
        """Computes the gradients for the parameters based on the loss gradient.

        The gradients are computed analytically:
        - ∂μ/∂a: Affects the width of the curve
        - ∂μ/∂b: Affects the steepness of the curve
        - ∂μ/∂c: Affects the center position of the curve

        Parameters:
            dL_dy (np.ndarray): The gradient of the loss with respect to the output of this layer.

        Returns:
            None: Gradients are accumulated in self.gradients.
        """
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]

        x = self.last_input
        y = self.last_output  # This is μ(x)

        # Intermediate calculations
        normalized = (x - c) / a
        abs_normalized = np.abs(normalized)

        # Avoid division by zero and numerical issues
        # Only compute gradients where abs_normalized > epsilon
        epsilon = 1e-12
        valid_mask = abs_normalized > epsilon

        if not np.any(valid_mask):
            # If all values are at the peak (x ≈ c), gradients are zero
            return

        # Initialize gradients
        dL_da = 0.0
        dL_db = 0.0
        dL_dc = 0.0

        # Only compute where we have valid values
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        dL_dy_valid = dL_dy[valid_mask]
        normalized_valid = (x_valid - c) / a
        abs_normalized_valid = np.abs(normalized_valid)

        # Power term: |normalized|^(2b)
        power_term_valid = np.power(abs_normalized_valid, 2 * b)

        # For the bell function μ = 1/(1 + z) where z = |normalized|^(2b)
        # ∂μ/∂z = -1/(1 + z)² = -μ²
        dmu_dz = -y_valid * y_valid

        # Chain rule: ∂L/∂param = ∂L/∂μ × ∂μ/∂z × ∂z/∂param

        # ∂z/∂a = ∂(|normalized|^(2b))/∂a
        # = 2b × |normalized|^(2b-1) × ∂|normalized|/∂a
        # = 2b × |normalized|^(2b-1) × sign(normalized) × ∂normalized/∂a
        # = 2b × |normalized|^(2b-1) × sign(normalized) × (-(x-c)/a²)
        # = -2b × |normalized|^(2b-1) × sign(normalized) × (x-c)/a²

        sign_normalized = np.sign(normalized_valid)
        dz_da = -2 * b * np.power(abs_normalized_valid, 2 * b - 1) * sign_normalized * (x_valid - c) / (a * a)
        dL_da += np.sum(dL_dy_valid * dmu_dz * dz_da)

        # ∂z/∂b = ∂(|normalized|^(2b))/∂b
        # = |normalized|^(2b) × ln(|normalized|) × 2
        # But ln(|normalized|) can be problematic near zero, so we use a safe version
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_abs_normalized = np.log(abs_normalized_valid)
            ln_abs_normalized = np.where(np.isfinite(ln_abs_normalized), ln_abs_normalized, 0.0)

        dz_db = 2 * power_term_valid * ln_abs_normalized
        dL_db += np.sum(dL_dy_valid * dmu_dz * dz_db)

        # ∂z/∂c = ∂(|normalized|^(2b))/∂c
        # = 2b × |normalized|^(2b-1) × sign(normalized) × ∂normalized/∂c
        # = 2b × |normalized|^(2b-1) × sign(normalized) × (-1/a)
        # = -2b × |normalized|^(2b-1) × sign(normalized) / a

        dz_dc = -2 * b * np.power(abs_normalized_valid, 2 * b - 1) * sign_normalized / a
        dL_dc += np.sum(dL_dy_valid * dmu_dz * dz_dc)

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
        """Returns string representation of the bell membership function."""
        return f"BellMF(a={self.parameters['a']:.3f}, b={self.parameters['b']:.3f}, c={self.parameters['c']:.3f})"

    def __repr__(self) -> str:
        """Returns detailed representation of the bell membership function."""
        return f"BellMF(a={self.parameters['a']}, b={self.parameters['b']}, c={self.parameters['c']})"
