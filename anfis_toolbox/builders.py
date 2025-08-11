"""Builder classes for easy ANFIS model construction."""

import numpy as np

from .membership import GaussianMF, TrapezoidalMF, TriangularMF
from .model import ANFIS


class ANFISBuilder:
    """Builder class for creating ANFIS models with intuitive API."""

    def __init__(self):
        """Initialize the ANFIS builder."""
        self.input_mfs = {}
        self.input_ranges = {}

    def add_input(
        self,
        name: str,
        range_min: float,
        range_max: float,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        overlap: float = 0.5,
    ) -> "ANFISBuilder":
        """Add an input variable with automatic membership function generation.

        Parameters:
            name: Name of the input variable
            range_min: Minimum value of the input range
            range_max: Maximum value of the input range
            n_mfs: Number of membership functions (default: 3)
            mf_type: Type of membership functions ('gaussian', 'triangular', 'trapezoidal')
            overlap: Overlap factor between adjacent MFs (0.0 to 1.0)

        Returns:
            Self for method chaining
        """
        self.input_ranges[name] = (range_min, range_max)

        if mf_type.lower() == "gaussian":
            self.input_mfs[name] = self._create_gaussian_mfs(range_min, range_max, n_mfs, overlap)
        elif mf_type.lower() == "triangular":
            self.input_mfs[name] = self._create_triangular_mfs(range_min, range_max, n_mfs, overlap)
        elif mf_type.lower() == "trapezoidal":
            self.input_mfs[name] = self._create_trapezoidal_mfs(range_min, range_max, n_mfs, overlap)
        else:
            raise ValueError(f"Unknown membership function type: {mf_type}")

        return self

    def _create_gaussian_mfs(self, range_min: float, range_max: float, n_mfs: int, overlap: float) -> list[GaussianMF]:
        """Create evenly spaced Gaussian membership functions."""
        centers = np.linspace(range_min, range_max, n_mfs)
        sigma = (range_max - range_min) / (n_mfs - 1) * overlap
        return [GaussianMF(mean=center, sigma=sigma) for center in centers]

    def _create_triangular_mfs(
        self, range_min: float, range_max: float, n_mfs: int, overlap: float
    ) -> list[TriangularMF]:
        """Create evenly spaced triangular membership functions."""
        centers = np.linspace(range_min, range_max, n_mfs)
        width = (range_max - range_min) / (n_mfs - 1) * (1 + overlap)

        mfs = []
        for i, center in enumerate(centers):
            a = center - width / 2
            b = center
            c = center + width / 2

            # Adjust boundaries for edge cases
            if i == 0:
                a = range_min
            if i == n_mfs - 1:
                c = range_max

            mfs.append(TriangularMF(a, b, c))

        return mfs

    def _create_trapezoidal_mfs(
        self, range_min: float, range_max: float, n_mfs: int, overlap: float
    ) -> list[TrapezoidalMF]:
        """Create evenly spaced trapezoidal membership functions."""
        centers = np.linspace(range_min, range_max, n_mfs)
        width = (range_max - range_min) / (n_mfs - 1) * (1 + overlap)
        plateau = width * 0.3  # 30% plateau

        mfs = []
        for i, center in enumerate(centers):
            a = center - width / 2
            b = center - plateau / 2
            c = center + plateau / 2
            d = center + width / 2

            # Adjust boundaries for edge cases
            if i == 0:
                a = range_min
                b = max(b, range_min)
            if i == n_mfs - 1:
                c = min(c, range_max)
                d = range_max

            mfs.append(TrapezoidalMF(a, b, c, d))

        return mfs

    def build(self) -> ANFIS:
        """Build the ANFIS model with configured parameters."""
        if not self.input_mfs:
            raise ValueError("No input variables defined. Use add_input() to define inputs.")

        return ANFIS(self.input_mfs)


class QuickANFIS:
    """Quick setup class for common ANFIS use cases."""

    @staticmethod
    def for_regression(X: np.ndarray, n_mfs: int = 3, mf_type: str = "gaussian") -> ANFIS:
        """Create ANFIS model automatically configured for regression data.

        Parameters:
            X: Input training data (n_samples, n_features)
            n_mfs: Number of membership functions per input
            mf_type: Type of membership functions

        Returns:
            Configured ANFIS model
        """
        if X.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features)")

        builder = ANFISBuilder()

        for i in range(X.shape[1]):
            col_data = X[:, i]
            range_min = float(np.min(col_data))
            range_max = float(np.max(col_data))

            # Add some margin
            margin = (range_max - range_min) * 0.1
            range_min -= margin
            range_max += margin

            builder.add_input(f"x{i + 1}", range_min, range_max, n_mfs, mf_type)

        return builder.build()

    @staticmethod
    def for_function_approximation(input_ranges: list[tuple[float, float]], n_mfs: int = 5) -> ANFIS:
        """Create ANFIS model for function approximation.

        Parameters:
            input_ranges: List of (min, max) tuples for each input dimension
            n_mfs: Number of membership functions per input

        Returns:
            Configured ANFIS model
        """
        builder = ANFISBuilder()

        for i, (range_min, range_max) in enumerate(input_ranges):
            builder.add_input(f"x{i + 1}", range_min, range_max, n_mfs, "gaussian")

        return builder.build()
