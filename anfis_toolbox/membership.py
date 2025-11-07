from abc import ABC, abstractmethod
from typing import cast

import numpy as np


# Shared helpers for smooth S/Z transitions
def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Cubic smoothstep S(t) = 3t^2 - 2t^3 for t in [0,1]."""
    return 3.0 * t**2 - 2.0 * t**3


def _dsmoothstep_dt(t: np.ndarray) -> np.ndarray:
    """Derivative of smoothstep: dS/dt = 6t(1-t)."""
    return 6.0 * t * (1.0 - t)


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

    def __init__(self) -> None:
        """Initializes the membership function with empty parameters and gradients."""
        self.parameters: dict[str, float] = {}
        self.gradients: dict[str, float] = {}
        self.last_input: np.ndarray | None = None
        self.last_output: np.ndarray | None = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass of the membership function.

        Args:
            x: Input array for which the membership values are computed.

        Returns:
            np.ndarray: Array with the computed membership values.
        """
        pass  # pragma: no cover

    @abstractmethod
    def backward(self, dL_dy: np.ndarray) -> None:
        """Perform the backward pass for backpropagation.

        Args:
            dL_dy: Gradient of the loss with respect to the output of this layer.

        Returns:
            None
        """
        pass  # pragma: no cover

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call forward to compute membership values.

        Args:
            x: Input array for which the membership values are computed.

        Returns:
            np.ndarray: Array with the computed membership values.
        """
        return self.forward(x)

    def reset(self) -> None:
        """Reset internal state and accumulated gradients.

        Returns:
            None
        """
        for key in self.gradients:
            self.gradients[key] = 0.0
        self.last_input = None
        self.last_output = None

    def __str__(self) -> str:
        """Return a concise string representation of this membership function."""
        params = ", ".join(f"{key}={value:.3f}" for key, value in self.parameters.items())
        return f"{self.__class__.__name__}({params})"

    def __repr__(self) -> str:
        """Return a detailed representation of this membership function."""
        return self.__str__()


class GaussianMF(MembershipFunction):
    """Gaussian Membership Function.

    Implements a Gaussian (bell-shaped) membership function using the formula:
    μ(x) = exp(-((x - mean)² / (2 * sigma²)))

    This function is commonly used in fuzzy logic systems due to its smooth
    and differentiable properties.
    """

    def __init__(self, mean: float = 0.0, sigma: float = 1.0):
        """Initialize with mean and standard deviation.

        Args:
            mean: Mean of the Gaussian (center). Defaults to 0.0.
            sigma: Standard deviation (width). Defaults to 1.0.
        """
        super().__init__()
        self.parameters = {"mean": mean, "sigma": sigma}
        # Initialize gradients to zero for all parameters
        self.gradients = dict.fromkeys(self.parameters.keys(), 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute Gaussian membership values.

        Args:
            x: Input array for which the membership values are computed.

        Returns:
            np.ndarray: Array of Gaussian membership values.
        """
        mean = self.parameters["mean"]
        sigma = self.parameters["sigma"]
        self.last_input = x
        self.last_output = np.exp(-((x - mean) ** 2) / (2 * sigma**2))
        return self.last_output

    def backward(self, dL_dy: np.ndarray) -> None:
        """Compute gradients w.r.t. parameters given upstream gradient.

        Args:
            dL_dy: Gradient of the loss with respect to the output of this layer.

        Returns:
            None
        """
        mean = self.parameters["mean"]
        sigma = self.parameters["sigma"]

        if self.last_input is None or self.last_output is None:
            raise RuntimeError("forward must be called before backward.")

        x = cast(np.ndarray, self.last_input)
        y = cast(np.ndarray, self.last_output)

        z = (x - mean) / sigma

        # Derivatives of the Gaussian function
        dy_dmean = -y * z / sigma
        dy_dsigma = y * (z**2) / sigma

        # Gradient with respect to mean
        dL_dmean = np.sum(dL_dy * dy_dmean)

        # Gradient with respect to sigma
        dL_dsigma = np.sum(dL_dy * dy_dsigma)

        # Update gradients
        self.gradients["mean"] += dL_dmean
        self.gradients["sigma"] += dL_dsigma


class Gaussian2MF(MembershipFunction):
    """Gaussian combination Membership Function (two-sided Gaussian).

    This membership function uses Gaussian tails on both sides with an optional
    flat region in the middle.

    Parameters:
        sigma1 (float): Standard deviation of the left Gaussian tail (must be > 0).
        c1 (float): Center of the left Gaussian tail.
        sigma2 (float): Standard deviation of the right Gaussian tail (must be > 0).
        c2 (float): Center of the right Gaussian tail. Must satisfy c1 <= c2.

    Definition (with c1 <= c2):
        - For x < c1: μ(x) = exp(-((x - c1)^2) / (2*sigma1^2))
        - For c1 <= x <= c2: μ(x) = 1
        - For x > c2: μ(x) = exp(-((x - c2)^2) / (2*sigma2^2))

    Special case (c1 == c2): asymmetric Gaussian centered at c1 with sigma1 on the
    left side and sigma2 on the right side (no flat region).
    """

    def __init__(self, sigma1: float = 1.0, c1: float = 0.0, sigma2: float = 1.0, c2: float = 0.0):
        """Initialize the membership function with two Gaussian components.

        Args:
            sigma1 (float, optional): Standard deviation of the first Gaussian. Must be positive. Defaults to 1.0.
            c1 (float, optional): Center of the first Gaussian. Defaults to 0.0.
            sigma2 (float, optional): Standard deviation of the second Gaussian. Must be positive. Defaults to 1.0.
            c2 (float, optional): Center of the second Gaussian. Must satisfy c1 <= c2. Defaults to 0.0.

        Raises:
            ValueError: If sigma1 or sigma2 are not positive.
            ValueError: If c1 > c2.

        Attributes:
            parameters (dict): Dictionary containing the parameters 'sigma1', 'c1', 'sigma2', 'c2'.
            gradients (dict): Dictionary containing the gradients for each parameter, initialized to 0.0.
        """
        super().__init__()
        if sigma1 <= 0:
            raise ValueError(f"Parameter 'sigma1' must be positive, got sigma1={sigma1}")
        if sigma2 <= 0:
            raise ValueError(f"Parameter 'sigma2' must be positive, got sigma2={sigma2}")
        if c1 > c2:
            raise ValueError(f"Parameters must satisfy c1 <= c2, got c1={c1}, c2={c2}")

        self.parameters = {"sigma1": float(sigma1), "c1": float(c1), "sigma2": float(sigma2), "c2": float(c2)}
        self.gradients = {"sigma1": 0.0, "c1": 0.0, "sigma2": 0.0, "c2": 0.0}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute two-sided Gaussian membership values.

        The input space is divided by c1 and c2 into:
        - x < c1: left Gaussian tail with sigma1 centered at c1
        - c1 <= x <= c2: flat region (1.0)
        - x > c2: right Gaussian tail with sigma2 centered at c2

        Args:
            x: Input array of values.

        Returns:
            np.ndarray: Membership degrees for each input value.
        """
        x = np.asarray(x, dtype=float)
        self.last_input = x

        s1 = self.parameters["sigma1"]
        c1 = self.parameters["c1"]
        s2 = self.parameters["sigma2"]
        c2 = self.parameters["c2"]

        y = np.zeros_like(x, dtype=float)

        # Regions
        left_mask = x < c1
        mid_mask = (x >= c1) & (x <= c2)
        right_mask = x > c2

        if np.any(left_mask):
            xl = x[left_mask]
            y[left_mask] = np.exp(-((xl - c1) ** 2) / (2.0 * s1 * s1))

        if np.any(mid_mask):
            y[mid_mask] = 1.0

        if np.any(right_mask):
            xr = x[right_mask]
            y[right_mask] = np.exp(-((xr - c2) ** 2) / (2.0 * s2 * s2))

        self.last_output = y
        return y

    def backward(self, dL_dy: np.ndarray) -> None:
        """Accumulate parameter gradients for the two-sided Gaussian.

        The flat middle region contributes no gradients.

        Args:
            dL_dy: Upstream gradient of the loss w.r.t. the output.

        Returns:
            None
        """
        if self.last_input is None or self.last_output is None:
            return

        x = self.last_input
        dL_dy = np.asarray(dL_dy)

        s1 = self.parameters["sigma1"]
        c1 = self.parameters["c1"]
        s2 = self.parameters["sigma2"]
        c2 = self.parameters["c2"]

        # Regions
        left_mask = x < c1
        mid_mask = (x >= c1) & (x <= c2)
        right_mask = x > c2

        # Left Gaussian tail contributions (treat like a GaussianMF on that region)
        if np.any(left_mask):
            xl = x[left_mask]
            yl = np.exp(-((xl - c1) ** 2) / (2.0 * s1 * s1))
            z1 = (xl - c1) / s1
            # Match GaussianMF derivative conventions
            dmu_dc1 = yl * z1 / s1
            dmu_dsigma1 = yl * (z1**2) / s1

            dL_dc1 = np.sum(dL_dy[left_mask] * dmu_dc1)
            dL_dsigma1 = np.sum(dL_dy[left_mask] * dmu_dsigma1)

            self.gradients["c1"] += float(dL_dc1)
            self.gradients["sigma1"] += float(dL_dsigma1)

        # Mid region (flat) contributes no gradients
        _ = mid_mask  # placeholder to document intentional no-op

        # Right Gaussian tail contributions
        if np.any(right_mask):
            xr = x[right_mask]
            yr = np.exp(-((xr - c2) ** 2) / (2.0 * s2 * s2))
            z2 = (xr - c2) / s2
            dmu_dc2 = yr * z2 / s2
            dmu_dsigma2 = yr * (z2**2) / s2

            dL_dc2 = np.sum(dL_dy[right_mask] * dmu_dc2)
            dL_dsigma2 = np.sum(dL_dy[right_mask] * dmu_dsigma2)

            self.gradients["c2"] += float(dL_dc2)
            self.gradients["sigma2"] += float(dL_dsigma2)


class TriangularMF(MembershipFunction):
    """Triangular Membership Function.

    Implements a triangular membership function using piecewise linear segments:
    μ(x) = { 0,           x ≤ a or x ≥ c
           { (x-a)/(b-a), a < x < b
           { (c-x)/(c-b), b ≤ x < c

    Parameters:
        a (float): Left base point of the triangle.
        b (float): Peak point of the triangle (μ(b) = 1).
        c (float): Right base point of the triangle.

    Note:
        Must satisfy: a ≤ b ≤ c
    """

    def __init__(self, a: float, b: float, c: float):
        """Initialize the triangular membership function.

        Args:
            a: Left base point (must satisfy a ≤ b).
            b: Peak point (must satisfy a ≤ b ≤ c).
            c: Right base point (must satisfy b ≤ c).

        Raises:
            ValueError: If parameters do not satisfy a ≤ b ≤ c or if a == c (zero width).
        """
        super().__init__()

        if not (a <= b <= c):
            raise ValueError(f"Triangular MF parameters must satisfy a ≤ b ≤ c, got a={a}, b={b}, c={c}")
        if a == c:
            raise ValueError("Parameters 'a' and 'c' cannot be equal (zero width triangle)")

        self.parameters = {"a": float(a), "b": float(b), "c": float(c)}
        self.gradients = dict.fromkeys(self.parameters.keys(), 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute triangular membership values μ(x).

        Uses piecewise linear segments defined by (a, b, c):
        - 0 outside [a, c]
        - rising slope in (a, b)
        - peak 1 at x == b
        - falling slope in (b, c)

        Args:
            x: Input array.

        Returns:
            np.ndarray: Membership values in [0, 1] with the same shape as x.
        """
        a, b, c = self.parameters["a"], self.parameters["b"], self.parameters["c"]
        self.last_input = x

        output = np.zeros_like(x, dtype=float)

        # Left slope
        if b > a:
            left_mask = (x > a) & (x < b)
            output[left_mask] = (x[left_mask] - a) / (b - a)

        # Peak
        peak_mask = x == b
        output[peak_mask] = 1.0

        # Right slope
        if c > b:
            right_mask = (x > b) & (x < c)
            output[right_mask] = (c - x[right_mask]) / (c - b)

        # Clip for numerical stability
        output = np.clip(output, 0.0, 1.0)

        self.last_output = output
        return output

    def backward(self, dL_dy: np.ndarray) -> None:
        """Accumulate gradients for a, b, c given upstream gradient.

        Computes analytical derivatives for the rising (a, b) and falling (b, c)
        regions and sums them over the batch.

        Args:
            dL_dy: Gradient of the loss w.r.t. μ(x); same shape or broadcastable to output.

        Returns:
            None
        """
        if self.last_input is None or self.last_output is None:
            return

        a, b, c = self.parameters["a"], self.parameters["b"], self.parameters["c"]
        x = self.last_input

        dL_da = 0.0
        dL_db = 0.0
        dL_dc = 0.0

        # Left slope: a < x < b
        if b > a:
            left_mask = (x > a) & (x < b)
            if np.any(left_mask):
                x_left = x[left_mask]
                dL_dy_left = dL_dy[left_mask]

                # ∂μ/∂a = (x - b) / (b - a)^2
                dmu_da_left = (x_left - b) / ((b - a) ** 2)
                dL_da += np.sum(dL_dy_left * dmu_da_left)

                # ∂μ/∂b = -(x - a) / (b - a)^2
                dmu_db_left = -(x_left - a) / ((b - a) ** 2)
                dL_db += np.sum(dL_dy_left * dmu_db_left)

        # Right slope: b < x < c
        if c > b:
            right_mask = (x > b) & (x < c)
            if np.any(right_mask):
                x_right = x[right_mask]
                dL_dy_right = dL_dy[right_mask]

                # ∂μ/∂b = (x - c) / (c - b)^2
                dmu_db_right = (x_right - c) / ((c - b) ** 2)
                dL_db += np.sum(dL_dy_right * dmu_db_right)

                # ∂μ/∂c = (x - b) / (c - b)^2
                dmu_dc_right = (x_right - b) / ((c - b) ** 2)
                dL_dc += np.sum(dL_dy_right * dmu_dc_right)

        # Update gradients
        self.gradients["a"] += dL_da
        self.gradients["b"] += dL_db
        self.gradients["c"] += dL_dc


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
        """Initialize the trapezoidal membership function.

        Args:
            a: Left base point (μ(a) = 0).
            b: Left peak point (μ(b) = 1, start of plateau).
            c: Right peak point (μ(c) = 1, end of plateau).
            d: Right base point (μ(d) = 0).

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
        """Compute trapezoidal membership values.

        Args:
            x: Input array.

        Returns:
            np.ndarray: Array containing the trapezoidal membership values.
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

    def backward(self, dL_dy: np.ndarray) -> None:
        """Compute gradients for parameters based on upstream loss gradient.

        Analytical gradients for the piecewise linear function:
        - ∂μ/∂a: left slope
        - ∂μ/∂b: left slope and plateau transition
        - ∂μ/∂c: right slope and plateau transition
        - ∂μ/∂d: right slope

        Args:
            dL_dy: Gradient of the loss w.r.t. the output of this layer.

        Returns:
            None
        """
        if self.last_input is None or self.last_output is None:
            return

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
        """Initialize with width, slope, and center parameters.

        Args:
            a: Width parameter (must be positive). Defaults to 1.0.
            b: Slope parameter (must be positive). Defaults to 2.0.
            c: Center parameter. Defaults to 0.0.

        Raises:
            ValueError: If 'a' or 'b' are not positive.
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
        """Compute bell membership values.

        Args:
            x: Input array for which the membership values are computed.

        Returns:
            np.ndarray: Array of bell membership values.
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

    def backward(self, dL_dy: np.ndarray) -> None:
        """Compute parameter gradients given upstream gradient.

        Analytical gradients:
        - ∂μ/∂a: width
        - ∂μ/∂b: steepness
        - ∂μ/∂c: center

        Args:
            dL_dy: Gradient of the loss w.r.t. the output of this layer.

        Returns:
            None
        """
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]

        if self.last_input is None or self.last_output is None:
            raise RuntimeError("forward must be called before backward.")

        x = cast(np.ndarray, self.last_input)
        y = cast(np.ndarray, self.last_output)  # This is μ(x)

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


class SigmoidalMF(MembershipFunction):
    """Sigmoidal Membership Function.

    Implements a sigmoidal (S-shaped) membership function using the formula:
    μ(x) = 1 / (1 + exp(-a(x - c)))

    This function provides a smooth S-shaped curve that transitions from 0 to 1.
    It's particularly useful for modeling gradual transitions and is commonly
    used in neural networks and fuzzy systems.

    Parameters:
        a (float): Slope parameter. Controls the steepness of the sigmoid.
                   - Positive values: standard sigmoid (0 → 1 as x increases)
                   - Negative values: inverted sigmoid (1 → 0 as x increases)
                   - Larger |a|: steeper transition
        c (float): Center parameter. Controls the inflection point where μ(c) = 0.5.

    Note:
        Parameter 'a' cannot be zero (would result in constant function).
    """

    def __init__(self, a: float = 1.0, c: float = 0.0):
        """Initialize the sigmoidal membership function.

        Args:
            a: Slope parameter (cannot be zero). Defaults to 1.0.
            c: Center parameter (inflection point). Defaults to 0.0.

        Raises:
            ValueError: If 'a' is zero.
        """
        super().__init__()

        # Validate parameters
        if a == 0:
            raise ValueError(f"Parameter 'a' cannot be zero, got a={a}")

        self.parameters = {"a": float(a), "c": float(c)}
        # Initialize gradients to zero for all parameters
        self.gradients = dict.fromkeys(self.parameters.keys(), 0.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoidal membership values.

        Args:
            x: Input array for which the membership values are computed.

        Returns:
            np.ndarray: Array of sigmoidal membership values.
        """
        a = self.parameters["a"]
        c = self.parameters["c"]

        self.last_input = x

        # Compute the sigmoid function: μ(x) = 1 / (1 + exp(-a(x - c)))
        # To avoid numerical overflow, we use a stable implementation

        # Compute a(x - c) (note: not -a(x - c))
        z = a * (x - c)

        # Use stable sigmoid implementation to avoid overflow
        # Standard sigmoid: σ(z) = 1 / (1 + exp(-z))
        # For numerical stability:
        # If z >= 0: σ(z) = 1 / (1 + exp(-z))
        # If z < 0: σ(z) = exp(z) / (1 + exp(z))

        output = np.zeros_like(x, dtype=float)

        # Case 1: z >= 0 (standard case)
        mask_pos = z >= 0
        if np.any(mask_pos):
            output[mask_pos] = 1.0 / (1.0 + np.exp(-z[mask_pos]))

        # Case 2: z < 0 (to avoid exp overflow)
        mask_neg = z < 0
        if np.any(mask_neg):
            exp_z = np.exp(z[mask_neg])
            output[mask_neg] = exp_z / (1.0 + exp_z)

        self.last_output = output
        return output

    def backward(self, dL_dy: np.ndarray) -> None:
        """Compute parameter gradients given upstream gradient.

        For μ(x) = 1/(1 + exp(-a(x-c))):
        - ∂μ/∂a = μ(x)(1-μ(x))(x-c)
        - ∂μ/∂c = -aμ(x)(1-μ(x))

        Args:
            dL_dy: Gradient of the loss w.r.t. the output of this layer.

        Returns:
            None
        """
        a = self.parameters["a"]
        c = self.parameters["c"]

        if self.last_input is None or self.last_output is None:
            raise RuntimeError("forward must be called before backward.")

        x = cast(np.ndarray, self.last_input)
        y = cast(np.ndarray, self.last_output)  # This is μ(x)

        # For sigmoid: ∂μ/∂z = μ(1-μ) where z = -a(x-c)
        # This is a fundamental property of the sigmoid function
        dmu_dz = y * (1.0 - y)

        # Chain rule: ∂L/∂param = ∂L/∂μ × ∂μ/∂z × ∂z/∂param

        # For z = a(x-c):
        # ∂z/∂a = (x-c)
        # ∂z/∂c = -a

        # Gradient w.r.t. 'a'
        dz_da = x - c
        dL_da = np.sum(dL_dy * dmu_dz * dz_da)

        # Gradient w.r.t. 'c'
        dz_dc = -a
        dL_dc = np.sum(dL_dy * dmu_dz * dz_dc)

        # Update gradients (accumulate for batch processing)
        self.gradients["a"] += dL_da
        self.gradients["c"] += dL_dc


class DiffSigmoidalMF(MembershipFunction):
    """Difference of two sigmoidal functions.

    Implements y = s1(x) - s2(x), where each s is a logistic curve with its
    own slope and center parameters.
    """

    def __init__(self, a1: float, c1: float, a2: float, c2: float):
        """Initializes the membership function with two sets of parameters.

        Args:
            a1 (float): The first 'a' parameter for the membership function.
            c1 (float): The first 'c' parameter for the membership function.
            a2 (float): The second 'a' parameter for the membership function.
            c2 (float): The second 'c' parameter for the membership function.

        Attributes:
            parameters (dict): Dictionary containing the membership function parameters.
            gradients (dict): Dictionary containing gradients for each parameter, initialized to 0.0.
            last_input: Stores the last input value (initially None).
            last_output: Stores the last output value (initially None).
        """
        super().__init__()
        self.parameters = {
            "a1": float(a1),
            "c1": float(c1),
            "a2": float(a2),
            "c2": float(c2),
        }
        self.gradients = dict.fromkeys(self.parameters, 0.0)
        self.last_input = None
        self.last_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute y = s1(x) - s2(x).

        Args:
            x: Input array.

        Returns:
            np.ndarray: Membership values for the input.
        """
        x = np.asarray(x, dtype=float)
        self.last_input = x
        a1, c1 = self.parameters["a1"], self.parameters["c1"]
        a2, c2 = self.parameters["a2"], self.parameters["c2"]

        s1 = 1.0 / (1.0 + np.exp(-a1 * (x - c1)))
        s2 = 1.0 / (1.0 + np.exp(-a2 * (x - c2)))
        y = s1 - s2

        self.last_output = y
        self._s1, self._s2 = s1, s2  # store for backward
        return y

    def backward(self, dL_dy: np.ndarray) -> None:
        """Compute gradients w.r.t. parameters and optionally input.

        Args:
            dL_dy: Gradient of the loss w.r.t. the output.

        Returns:
            np.ndarray | None: Gradient of the loss w.r.t. the input, if available.
        """
        if self.last_input is None or self.last_output is None:
            return

        x = self.last_input
        dL_dy = np.asarray(dL_dy)
        a1, c1 = self.parameters["a1"], self.parameters["c1"]
        a2, c2 = self.parameters["a2"], self.parameters["c2"]
        s1, s2 = self._s1, self._s2

        # Sigmoid derivatives
        ds1_da1 = (x - c1) * s1 * (1 - s1)
        ds1_dc1 = -a1 * s1 * (1 - s1)
        ds2_da2 = (x - c2) * s2 * (1 - s2)
        ds2_dc2 = -a2 * s2 * (1 - s2)

        # Parameter gradients
        self.gradients["a1"] += float(np.sum(dL_dy * ds1_da1))
        self.gradients["c1"] += float(np.sum(dL_dy * ds1_dc1))
        self.gradients["a2"] += float(np.sum(dL_dy * -ds2_da2))
        self.gradients["c2"] += float(np.sum(dL_dy * -ds2_dc2))

        # Gradient w.r.t. input (optional, for chaining)
        # dmu_dx = a1 * s1 * (1 - s1) - a2 * s2 * (1 - s2)


class ProdSigmoidalMF(MembershipFunction):
    """Product of two sigmoidal functions.

    Implements μ(x) = s1(x) * s2(x) with separate parameters for each sigmoid.
    """

    def __init__(self, a1: float, c1: float, a2: float, c2: float):
        """Initializes the membership function with specified parameters.

        Args:
            a1 (float): The first parameter for the membership function.
            c1 (float): The second parameter for the membership function.
            a2 (float): The third parameter for the membership function.
            c2 (float): The fourth parameter for the membership function.

        Attributes:
            parameters (dict): Dictionary containing the membership function parameters.
            gradients (dict): Dictionary containing gradients for each parameter, initialized to 0.0.
            last_input: Stores the last input value (initialized to None).
            last_output: Stores the last output value (initialized to None).
        """
        super().__init__()
        self.parameters = {
            "a1": float(a1),
            "c1": float(c1),
            "a2": float(a2),
            "c2": float(c2),
        }
        self.gradients = dict.fromkeys(self.parameters, 0.0)
        self.last_input = None
        self.last_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the membership value(s) for input x using the product of two sigmoidal functions.

        Args:
            x (np.ndarray): Input array to the membership function.

        Returns:
            np.ndarray: Output array after applying the membership function.
        """
        x = np.asarray(x, dtype=float)
        self.last_input = x
        a1, c1 = self.parameters["a1"], self.parameters["c1"]
        a2, c2 = self.parameters["a2"], self.parameters["c2"]

        s1 = 1.0 / (1.0 + np.exp(-a1 * (x - c1)))
        s2 = 1.0 / (1.0 + np.exp(-a2 * (x - c2)))
        y = s1 * s2

        self.last_output = y
        self._s1, self._s2 = s1, s2  # store for backward
        return y

    def backward(self, dL_dy: np.ndarray) -> None:
        """Compute parameter gradients and optionally return input gradient.

        Args:
            dL_dy: Gradient of the loss w.r.t. the output.

        Returns:
            np.ndarray | None: Gradient of the loss w.r.t. the input, if available.
        """
        if self.last_input is None or self.last_output is None:
            return

        x = self.last_input
        dL_dy = np.asarray(dL_dy)
        a1, c1 = self.parameters["a1"], self.parameters["c1"]
        a2, c2 = self.parameters["a2"], self.parameters["c2"]
        s1, s2 = self._s1, self._s2

        # derivatives of sigmoids w.r.t. parameters
        ds1_da1 = (x - c1) * s1 * (1 - s1)
        ds1_dc1 = -a1 * s1 * (1 - s1)
        ds2_da2 = (x - c2) * s2 * (1 - s2)
        ds2_dc2 = -a2 * s2 * (1 - s2)

        # parameter gradients using product rule
        self.gradients["a1"] += float(np.sum(dL_dy * ds1_da1 * s2))
        self.gradients["c1"] += float(np.sum(dL_dy * ds1_dc1 * s2))
        self.gradients["a2"] += float(np.sum(dL_dy * s1 * ds2_da2))
        self.gradients["c2"] += float(np.sum(dL_dy * s1 * ds2_dc2))

        # gradient w.r.t. input (optional)
        # dmu_dx = a1 * s1 * (1 - s1) * s2 + a2 * s2 * (1 - s2) * s1


class SShapedMF(MembershipFunction):
    """S-shaped Membership Function.

    Smoothly transitions from 0 to 1 between two parameters a and b using the
    smoothstep polynomial S(t) = 3t² - 2t³. Commonly used in fuzzy logic for
    gradual onset of membership.

    Definition with a < b:
    - μ(x) = 0, for x ≤ a
    - μ(x) = 3t² - 2t³, t = (x-a)/(b-a), for a < x < b
    - μ(x) = 1, for x ≥ b

    Parameters:
        a (float): Left foot (start of transition from 0).
        b (float): Right shoulder (end of transition at 1).

    Note:
        Requires a < b.
    """

    def __init__(self, a: float, b: float):
        """Initialize the membership function with parameters 'a' and 'b'.

        Args:
            a (float): The first parameter, must be less than 'b'.
            b (float): The second parameter, must be greater than 'a'.

        Raises:
            ValueError: If 'a' is not less than 'b'.

        Attributes:
            parameters (dict): Dictionary containing 'a' and 'b' as floats.
            gradients (dict): Dictionary containing gradients for 'a' and 'b', initialized to 0.0.
        """
        super().__init__()

        if not (a < b):
            raise ValueError(f"Parameters must satisfy a < b, got a={a}, b={b}")

        self.parameters = {"a": float(a), "b": float(b)}
        self.gradients = {"a": 0.0, "b": 0.0}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute S-shaped membership values."""
        x = np.asarray(x)
        self.last_input = x.copy()

        a, b = self.parameters["a"], self.parameters["b"]

        y = np.zeros_like(x, dtype=np.float64)

        # Right side (x ≥ b): μ = 1
        mask_right = x >= b
        y[mask_right] = 1.0

        # Transition region (a < x < b): μ = smoothstep(t)
        mask_trans = (x > a) & (x < b)
        if np.any(mask_trans):
            x_t = x[mask_trans]
            t = (x_t - a) / (b - a)
            y[mask_trans] = _smoothstep(t)

        # Left side (x ≤ a) remains 0

        self.last_output = y.copy()
        return y

    def backward(self, dL_dy: np.ndarray) -> None:
        """Accumulate gradients for a and b using analytical derivatives.

        Uses S(t) = 3t² - 2t³, t = (x-a)/(b-a) on the transition region.
        """
        if self.last_input is None or self.last_output is None:
            return

        x = self.last_input
        dL_dy = np.asarray(dL_dy)

        a, b = self.parameters["a"], self.parameters["b"]

        # Only transition region contributes to parameter gradients
        mask = (x >= a) & (x <= b)
        if not (np.any(mask) and b != a):
            return

        x_t = x[mask]
        dL_dy_t = dL_dy[mask]
        t = (x_t - a) / (b - a)

        # dS/dt = 6*t*(1-t)
        dS_dt = _dsmoothstep_dt(t)

        # dt/da and dt/db
        dt_da = (x_t - b) / (b - a) ** 2
        dt_db = -(x_t - a) / (b - a) ** 2

        dS_da = dS_dt * dt_da
        dS_db = dS_dt * dt_db

        self.gradients["a"] += float(np.sum(dL_dy_t * dS_da))
        self.gradients["b"] += float(np.sum(dL_dy_t * dS_db))


class LinSShapedMF(MembershipFunction):
    """Linear S-shaped saturation Membership Function.

    Piecewise linear ramp from 0 to 1 between parameters a and b:
      - μ(x) = 0, for x ≤ a
      - μ(x) = (x - a) / (b - a), for a < x < b
      - μ(x) = 1, for x ≥ b

    Parameters:
        a (float): Left foot (start of transition from 0).
        b (float): Right shoulder (end of transition at 1). Requires a < b.
    """

    def __init__(self, a: float, b: float):
        """Initialize the membership function with parameters 'a' and 'b'.

        Args:
            a (float): The first parameter, must be less than 'b'.
            b (float): The second parameter, must be greater than 'a'.

        Raises:
            ValueError: If 'a' is not less than 'b'.

        Attributes:
            parameters (dict): Dictionary containing 'a' and 'b' as floats.
            gradients (dict): Dictionary containing gradients for 'a' and 'b', initialized to 0.0.
        """
        super().__init__()
        if not (a < b):
            raise ValueError(f"Parameters must satisfy a < b, got a={a}, b={b}")
        self.parameters = {"a": float(a), "b": float(b)}
        self.gradients = {"a": 0.0, "b": 0.0}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute linear S-shaped membership values for x.

        The rules based on a and b:
        - x >= b: 1.0 (right saturated)
        - a < x < b: linear ramp from 0 to 1
        - x <= a: 0.0 (left)

        Args:
            x: Input array of values.

        Returns:
            np.ndarray: Output array with membership values.
        """
        x = np.asarray(x, dtype=float)
        self.last_input = x
        a, b = self.parameters["a"], self.parameters["b"]
        y = np.zeros_like(x, dtype=float)
        # right saturated region
        mask_right = x >= b
        y[mask_right] = 1.0
        # linear ramp
        mask_mid = (x > a) & (x < b)
        if np.any(mask_mid):
            y[mask_mid] = (x[mask_mid] - a) / (b - a)
        # left stays 0
        self.last_output = y
        return y

    def backward(self, dL_dy: np.ndarray) -> None:
        """Accumulate gradients for 'a' and 'b' in the ramp region.

        Args:
            dL_dy: Gradient of the loss w.r.t. the output.

        Returns:
            None
        """
        if self.last_input is None or self.last_output is None:
            return
        x = self.last_input
        dL_dy = np.asarray(dL_dy)
        a, b = self.parameters["a"], self.parameters["b"]
        d = b - a
        if d == 0:
            return
        # Only ramp region contributes to parameter gradients
        mask = (x > a) & (x < b)
        if not np.any(mask):
            return
        xm = x[mask]
        g = dL_dy[mask]
        # μ = (x-a)/d with d = b-a
        # ∂μ/∂a = -(1/d) + (x-a)/d^2
        dmu_da = -(1.0 / d) + (xm - a) / (d * d)
        # ∂μ/∂b = -(x-a)/d^2
        dmu_db = -((xm - a) / (d * d))
        self.gradients["a"] += float(np.sum(g * dmu_da))
        self.gradients["b"] += float(np.sum(g * dmu_db))


class ZShapedMF(MembershipFunction):
    """Z-shaped Membership Function.

    Smoothly transitions from 1 to 0 between two parameters a and b using the
    smoothstep polynomial S(t) = 3t² - 2t³ (Z = 1 - S). Commonly used in fuzzy
    logic as the complement of the S-shaped function.

    Definition with a < b:
    - μ(x) = 1, for x ≤ a
    - μ(x) = 1 - (3t² - 2t³), t = (x-a)/(b-a), for a < x < b
    - μ(x) = 0, for x ≥ b

    Parameters:
        a (float): Left shoulder (start of transition).
        b (float): Right foot (end of transition).

    Note:
        Requires a < b. In the degenerate case a == b, the function becomes an
        instantaneous drop at x=a.
    """

    def __init__(self, a: float, b: float):
        """Initialize the membership function with parameters a and b.

        Args:
            a: Lower bound parameter.
            b: Upper bound parameter.

        Raises:
            ValueError: If a is not less than b.
        """
        super().__init__()

        if not (a < b):
            raise ValueError(f"Parameters must satisfy a < b, got a={a}, b={b}")

        self.parameters = {"a": float(a), "b": float(b)}
        self.gradients = {"a": 0.0, "b": 0.0}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute Z-shaped membership values."""
        x = np.asarray(x)
        self.last_input = x.copy()

        a, b = self.parameters["a"], self.parameters["b"]

        y = np.zeros_like(x, dtype=np.float64)

        # Left side (x ≤ a): μ = 1
        mask_left = x <= a
        y[mask_left] = 1.0

        # Transition region (a < x < b): μ = 1 - smoothstep(t)
        mask_trans = (x > a) & (x < b)
        if np.any(mask_trans):
            x_t = x[mask_trans]
            t = (x_t - a) / (b - a)
            y[mask_trans] = 1.0 - _smoothstep(t)

        # Right side (x ≥ b) remains 0

        self.last_output = y.copy()
        return y

    def backward(self, dL_dy: np.ndarray) -> None:
        """Accumulate gradients for a and b using analytical derivatives.

        Uses Z(t) = 1 - (3t² - 2t³), t = (x-a)/(b-a) on the transition region.
        """
        if self.last_input is None or self.last_output is None:
            return

        x = self.last_input
        dL_dy = np.asarray(dL_dy)

        a, b = self.parameters["a"], self.parameters["b"]

        # Only transition region contributes to parameter gradients
        mask = (x >= a) & (x <= b)
        if not (np.any(mask) and b != a):
            return

        x_t = x[mask]
        dL_dy_t = dL_dy[mask]
        t = (x_t - a) / (b - a)

        # dZ/dt = -dS/dt = 6*t*(t-1)
        dZ_dt = -_dsmoothstep_dt(t)

        # dt/da and dt/db
        dt_da = (x_t - b) / (b - a) ** 2
        dt_db = -(x_t - a) / (b - a) ** 2

        dZ_da = dZ_dt * dt_da
        dZ_db = dZ_dt * dt_db

        self.gradients["a"] += float(np.sum(dL_dy_t * dZ_da))
        self.gradients["b"] += float(np.sum(dL_dy_t * dZ_db))


class LinZShapedMF(MembershipFunction):
    """Linear Z-shaped saturation Membership Function.

    Piecewise linear ramp from 1 to 0 between parameters a and b:
      - μ(x) = 1, for x ≤ a
      - μ(x) = (b - x) / (b - a), for a < x < b
      - μ(x) = 0, for x ≥ b

    Parameters:
        a (float): Left shoulder (end of saturation at 1).
        b (float): Right foot (end of transition to 0). Requires a < b.
    """

    def __init__(self, a: float, b: float):
        """Initialize the membership function with parameters 'a' and 'b'.

        Args:
            a (float): The first parameter of the membership function. Must be less than 'b'.
            b (float): The second parameter of the membership function.

        Raises:
            ValueError: If 'a' is not less than 'b'.

        Attributes:
            parameters (dict): Dictionary containing 'a' and 'b' as floats.
            gradients (dict): Dictionary containing gradients for 'a' and 'b', initialized to 0.0.
        """
        super().__init__()
        if not (a < b):
            raise ValueError(f"Parameters must satisfy a < b, got a={a}, b={b}")
        self.parameters = {"a": float(a), "b": float(b)}
        self.gradients = {"a": 0.0, "b": 0.0}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute linear Z-shaped membership values for x.

        Rules:
        - x <= a: 1.0 (left saturated)
        - a < x < b: linear ramp from 1 to 0
        - x >= b: 0.0 (right)

        Args:
            x: Input array of values.

        Returns:
            np.ndarray: Output membership values for each input.
        """
        x = np.asarray(x, dtype=float)
        self.last_input = x
        a, b = self.parameters["a"], self.parameters["b"]
        y = np.zeros_like(x, dtype=float)

        # left saturated region
        mask_left = x <= a
        y[mask_left] = 1.0
        # linear ramp
        mask_mid = (x > a) & (x < b)
        if np.any(mask_mid):
            y[mask_mid] = (b - x[mask_mid]) / (b - a)
        # right stays 0
        self.last_output = y
        return y

    def backward(self, dL_dy: np.ndarray) -> None:
        """Accumulate gradients for 'a' and 'b'.

        Args:
            dL_dy: Gradient of the loss w.r.t. the output.

        Returns:
            None
        """
        if self.last_input is None or self.last_output is None:
            return
        x = self.last_input
        dL_dy = np.asarray(dL_dy)
        a, b = self.parameters["a"], self.parameters["b"]
        d = b - a
        if d == 0:
            return
        mask = (x > a) & (x < b)
        if not np.any(mask):
            return
        xm = x[mask]
        g = dL_dy[mask]
        # μ = (b-x)/(b-a)
        # ∂μ/∂a = (b-x)/(d^2)
        # ∂μ/∂b = (x-a)/(d^2)
        dmu_da = (b - xm) / (d * d)
        dmu_db = (xm - a) / (d * d)
        self.gradients["a"] += float(np.sum(g * dmu_da))
        self.gradients["b"] += float(np.sum(g * dmu_db))

    # No return; gradients are accumulated in-place


class PiMF(MembershipFunction):
    """Pi-shaped membership function.

    The Pi-shaped membership function is characterized by a trapezoidal-like shape
    with smooth S-shaped transitions on both sides. It is defined by four parameters
    that control the shape and position:

    Mathematical definition:
    μ(x) = S(x; a, b) for x ∈ [a, b]
         = 1 for x ∈ [b, c]
         = Z(x; c, d) for x ∈ [c, d]
         = 0 elsewhere

    Where:
    - S(x; a, b) is an S-shaped function from 0 to 1
    - Z(x; c, d) is a Z-shaped function from 1 to 0

    The S and Z functions use smooth cubic splines for differentiability:
    S(x; a, b) = 2*((x-a)/(b-a))^3 for x ∈ [a, (a+b)/2]
               = 1 - 2*((b-x)/(b-a))^3 for x ∈ [(a+b)/2, b]

    Parameters:
        a (float): Left foot of the function (where function starts rising from 0)
        b (float): Left shoulder of the function (where function reaches 1)
        c (float): Right shoulder of the function (where function starts falling from 1)
        d (float): Right foot of the function (where function reaches 0)

    Note:
        Parameters must satisfy: a < b ≤ c < d
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        """Initialize the Pi-shaped membership function.

        Args:
            a: Left foot parameter.
            b: Left shoulder parameter.
            c: Right shoulder parameter.
            d: Right foot parameter.

        Raises:
            ValueError: If parameters don't satisfy a < b ≤ c < d.
        """
        super().__init__()

        # Parameter validation
        if not (a < b <= c < d):
            raise ValueError(f"Parameters must satisfy a < b ≤ c < d, got a={a}, b={b}, c={c}, d={d}")

        self.parameters = {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}
        self.gradients = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the Pi-shaped membership function.

        Combines S and Z functions for smooth transitions:
        - Rising edge: S-function from a to b
        - Flat top: constant 1 from b to c
        - Falling edge: Z-function from c to d
        - Outside: 0

        Args:
            x: Input values.

        Returns:
            np.ndarray: Membership values μ(x) ∈ [0, 1].
        """
        x = np.asarray(x)
        self.last_input = x.copy()

        a, b, c, d = self.parameters["a"], self.parameters["b"], self.parameters["c"], self.parameters["d"]

        # Initialize output
        y = np.zeros_like(x, dtype=np.float64)

        # S-function for rising edge [a, b]
        mask_s = (x >= a) & (x <= b)
        if np.any(mask_s):
            x_s = x[mask_s]
            # Avoid division by zero
            if b != a:
                t = (x_s - a) / (b - a)  # Normalize to [0, 1]

                # Smooth S-function using smoothstep: S(t) = 3*t² - 2*t³
                # This is continuous and differentiable across the entire [0,1] interval
                y_s = _smoothstep(t)

                y[mask_s] = y_s
            else:
                # Degenerate case: instant transition
                y[mask_s] = 1.0

        # Flat region [b, c]: μ(x) = 1
        mask_flat = (x >= b) & (x <= c)
        y[mask_flat] = 1.0

        # Z-function for falling edge [c, d]
        mask_z = (x >= c) & (x <= d)
        if np.any(mask_z):
            x_z = x[mask_z]
            # Avoid division by zero
            if d != c:
                t = (x_z - c) / (d - c)  # Normalize to [0, 1]

                # Smooth Z-function (inverted smoothstep): Z(t) = 1 - S(t) = 1 - (3*t² - 2*t³)
                # This is continuous and differentiable, going from 1 to 0
                y_z = 1 - _smoothstep(t)

                y[mask_z] = y_z
            else:
                # Degenerate case: instant transition
                y[mask_z] = 0.0

        self.last_output = y.copy()
        return y

    def backward(self, dL_dy: np.ndarray) -> None:
        """Compute gradients for backpropagation.

        Analytical gradients are computed by region:
        - S-function: gradients w.r.t. a, b
        - Z-function: gradients w.r.t. c, d
        - Flat region: no gradients

        Args:
            dL_dy: Gradient of loss w.r.t. function output.
        """
        if self.last_input is None or self.last_output is None:
            return

        x = self.last_input
        dL_dy = np.asarray(dL_dy)

        a, b, c, d = self.parameters["a"], self.parameters["b"], self.parameters["c"], self.parameters["d"]

        # Initialize gradients
        grad_a = grad_b = grad_c = grad_d = 0.0

        # S-function gradients [a, b]
        mask_s = (x >= a) & (x <= b)
        if np.any(mask_s) and b != a:
            x_s = x[mask_s]
            dL_dy_s = dL_dy[mask_s]
            t = (x_s - a) / (b - a)

            # Calculate parameter derivatives
            dt_da = (x_s - b) / (b - a) ** 2  # Correct derivative
            dt_db = -(x_s - a) / (b - a) ** 2

            # For smoothstep S(t) = 3*t² - 2*t³, derivative is dS/dt = 6*t - 6*t² = 6*t*(1-t)
            dS_dt = _dsmoothstep_dt(t)

            # Apply chain rule: dS/da = dS/dt * dt/da
            dS_da = dS_dt * dt_da
            dS_db = dS_dt * dt_db

            grad_a += np.sum(dL_dy_s * dS_da)
            grad_b += np.sum(dL_dy_s * dS_db)

        # Z-function gradients [c, d]
        mask_z = (x >= c) & (x <= d)
        if np.any(mask_z) and d != c:
            x_z = x[mask_z]
            dL_dy_z = dL_dy[mask_z]
            t = (x_z - c) / (d - c)

            # Calculate parameter derivatives
            dt_dc = (x_z - d) / (d - c) ** 2  # Correct derivative
            dt_dd = -(x_z - c) / (d - c) ** 2

            # For Z(t) = 1 - S(t) = 1 - (3*t² - 2*t³), derivative is dZ/dt = -dS/dt = -6*t*(1-t) = 6*t*(t-1)
            dZ_dt = -_dsmoothstep_dt(t)

            # Apply chain rule: dZ/dc = dZ/dt * dt/dc
            dZ_dc = dZ_dt * dt_dc
            dZ_dd = dZ_dt * dt_dd

            grad_c += np.sum(dL_dy_z * dZ_dc)
            grad_d += np.sum(dL_dy_z * dZ_dd)

        # Accumulate gradients
        self.gradients["a"] += grad_a
        self.gradients["b"] += grad_b
        self.gradients["c"] += grad_c
        self.gradients["d"] += grad_d
