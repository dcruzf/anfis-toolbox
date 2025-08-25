"""Builder classes for easy ANFIS model construction."""

import numpy as np

from .clustering import FuzzyCMeans
from .membership import (
    BellMF,
    GaussianMF,
    PiMF,
    SigmoidalMF,
    SShapedMF,
    TrapezoidalMF,
    TriangularMF,
    ZShapedMF,
)
from .model import ANFIS


class ANFISBuilder:
    """Builder class for creating ANFIS models with intuitive API."""

    def __init__(self):
        """Initialize the ANFIS builder."""
        self.input_mfs = {}
        self.input_ranges = {}
        # Centralized dispatch for MF creators (supports aliases)
        self._dispatch = {
            # Canonical
            "gaussian": self._create_gaussian_mfs,
            "triangular": self._create_triangular_mfs,
            "trapezoidal": self._create_trapezoidal_mfs,
            "bell": self._create_bell_mfs,
            "sigmoidal": self._create_sigmoidal_mfs,
            "sshape": self._create_sshape_mfs,
            "zshape": self._create_zshape_mfs,
            "pi": self._create_pi_mfs,
            # Aliases
            "gbell": self._create_bell_mfs,
            "sigmoid": self._create_sigmoidal_mfs,
            "s": self._create_sshape_mfs,
            "z": self._create_zshape_mfs,
            "pimf": self._create_pi_mfs,
        }

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
            mf_type: Type of membership functions. Supported:
                'gaussian', 'triangular', 'trapezoidal',
                'bell', 'sigmoidal', 'sshape', 'zshape', 'pi'
            overlap: Overlap factor between adjacent MFs (0.0 to 1.0)

        Returns:
            Self for method chaining
        """
        self.input_ranges[name] = (range_min, range_max)

        mf_key = mf_type.strip().lower()
        factory = self._dispatch.get(mf_key)
        if factory is None:
            supported = ", ".join(sorted(set(self._dispatch.keys())))
            raise ValueError(f"Unknown membership function type: {mf_type}. Supported: {supported}")
        self.input_mfs[name] = factory(range_min, range_max, n_mfs, overlap)

        return self

    def add_input_from_data(
        self,
        name: str,
        data: np.ndarray,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        overlap: float = 0.5,
        margin: float = 0.10,
        init: str = "grid",
        random_state: int | None = None,
    ) -> "ANFISBuilder":
        """Add an input inferring range_min/range_max from data with a margin.

        Parameters:
            name: Input name
            data: 1D array-like samples for this input
            n_mfs: Number of membership functions
            mf_type: Membership function type (see add_input)
            overlap: Overlap factor between adjacent MFs
            margin: Fraction of (max-min) to pad on each side
            init: Initialization strategy: "grid" (default) or "fcm". When "fcm",
                clusters from the data determine MF centers and widths (supports
                'gaussian' and 'bell').
            random_state: Optional seed for deterministic FCM initialization.
        """
        arr = np.asarray(data, dtype=float).reshape(-1)
        if init.strip().lower() == "fcm":
            self.input_mfs[name] = self._create_mfs_from_fcm(arr, n_mfs, mf_type, random_state)
            # store observed range for reference
            self.input_ranges[name] = (float(np.min(arr)), float(np.max(arr)))
            return self
        # default grid-based placement with margins
        rmin = float(np.min(arr))
        rmax = float(np.max(arr))
        pad = (rmax - rmin) * float(margin)
        return self.add_input(name, rmin - pad, rmax + pad, n_mfs, mf_type, overlap)

    # FCM-based MF creation for 1D inputs
    def _create_mfs_from_fcm(
        self,
        data_1d: np.ndarray,
        n_mfs: int,
        mf_type: str,
        random_state: int | None,
    ) -> list[GaussianMF] | list[BellMF]:
        """Create MFs from 1D data via FCM.

        - Centers are FCM cluster centers.
        - Widths come from weighted within-cluster variance with weights U^m.
        - Supports 'gaussian' and 'bell'.
        """
        x = np.asarray(data_1d, dtype=float).reshape(-1, 1)
        if x.shape[0] < n_mfs:
            raise ValueError("n_samples must be >= n_mfs for FCM initialization")
        fcm = FuzzyCMeans(n_clusters=n_mfs, m=2.0, random_state=random_state)
        fcm.fit(x)
        centers = fcm.cluster_centers_.reshape(-1)  # (k,)
        U = fcm.membership_  # (n,k)
        m = fcm.m
        # weighted variance per cluster
        # num_k = sum_i u_ik^m * (x_i - c_k)^2, den_k = sum_i u_ik^m
        diffs = x[:, 0][:, None] - centers[None, :]
        num = np.sum((U**m) * (diffs * diffs), axis=0)
        den = np.maximum(np.sum(U**m, axis=0), 1e-12)
        sigmas = np.sqrt(num / den)
        # fallback if any sigma is ~0
        spacing = np.diff(np.sort(centers))
        default_sigma = float(np.median(spacing)) if spacing.size else max(float(np.std(x)), 1e-3)
        sigmas = np.where(sigmas > 1e-12, sigmas, max(default_sigma, 1e-3))

        # Order by center for deterministic layout
        order = np.argsort(centers)
        centers = centers[order]
        sigmas = sigmas[order]

        key = mf_type.strip().lower()
        if key in {"gaussian"}:
            return [GaussianMF(mean=float(c), sigma=float(s)) for c, s in zip(centers, sigmas, strict=False)]
        if key in {"bell", "gbell"}:
            # map sigma to bell half-width a; keep b=2 by default
            return [BellMF(a=float(s), b=2.0, c=float(c)) for c, s in zip(centers, sigmas, strict=False)]
        raise ValueError("FCM init supports only 'gaussian' or 'bell' MF types")

    def _create_gaussian_mfs(self, range_min: float, range_max: float, n_mfs: int, overlap: float) -> list[GaussianMF]:
        """Create evenly spaced Gaussian membership functions."""
        centers = np.linspace(range_min, range_max, n_mfs)

        # Handle single MF case
        if n_mfs == 1:
            sigma = (range_max - range_min) * 0.25  # Use quarter of range as default sigma
        else:
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

    # New MF families
    def _create_bell_mfs(self, range_min: float, range_max: float, n_mfs: int, overlap: float) -> list[BellMF]:
        """Create evenly spaced Bell membership functions (generalized bell)."""
        centers = np.linspace(range_min, range_max, n_mfs)
        if n_mfs == 1:
            a = (range_max - range_min) * 0.25
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            a = spacing * (1 + overlap) / 2.0  # half-width
        b = 2.0  # default slope
        return [BellMF(a=a, b=b, c=float(c)) for c in centers]

    def _create_sigmoidal_mfs(
        self, range_min: float, range_max: float, n_mfs: int, overlap: float
    ) -> list[SigmoidalMF]:
        """Create a bank of sigmoids across the range with centers and slopes."""
        centers = np.linspace(range_min, range_max, n_mfs)
        if n_mfs == 1:
            width = (range_max - range_min) * 0.5
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            width = spacing * (1 + overlap)
        # Choose slope a s.t. 0.1->0.9 transition ~ width: width ≈ 4.4 / a
        a = 4.4 / max(width, 1e-8)
        return [SigmoidalMF(a=float(a), c=float(c)) for c in centers]

    def _create_sshape_mfs(self, range_min: float, range_max: float, n_mfs: int, overlap: float) -> list[SShapedMF]:
        """Create S-shaped MFs with spans around evenly spaced centers."""
        centers = np.linspace(range_min, range_max, n_mfs)
        if n_mfs == 1:
            width = (range_max - range_min) * 0.5
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            width = spacing * (1 + overlap)
        half = width / 2.0
        mfs: list[SShapedMF] = []
        for c in centers:
            a = float(c - half)
            b = float(c + half)
            # Clamp to the provided range
            a = max(a, range_min)
            b = min(b, range_max)
            if a >= b:
                # Fallback to a tiny span
                eps = 1e-6
                a, b = float(c - eps), float(c + eps)
            mfs.append(SShapedMF(a, b))
        return mfs

    def _create_zshape_mfs(self, range_min: float, range_max: float, n_mfs: int, overlap: float) -> list[ZShapedMF]:
        """Create Z-shaped MFs with spans around evenly spaced centers."""
        centers = np.linspace(range_min, range_max, n_mfs)
        if n_mfs == 1:
            width = (range_max - range_min) * 0.5
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            width = spacing * (1 + overlap)
        half = width / 2.0
        mfs: list[ZShapedMF] = []
        for c in centers:
            a = float(c - half)
            b = float(c + half)
            a = max(a, range_min)
            b = min(b, range_max)
            if a >= b:
                eps = 1e-6
                a, b = float(c - eps), float(c + eps)
            mfs.append(ZShapedMF(a, b))
        return mfs

    def _create_pi_mfs(self, range_min: float, range_max: float, n_mfs: int, overlap: float) -> list[PiMF]:
        """Create Pi-shaped MFs with smooth S/Z edges and a flat top."""
        centers = np.linspace(range_min, range_max, n_mfs)
        if n_mfs == 1:
            width = (range_max - range_min) * 0.5
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            width = spacing * (1 + overlap)
        plateau = width * 0.3
        mfs: list[PiMF] = []
        for c in centers:
            a = float(c - width / 2.0)
            d = float(c + width / 2.0)
            b = float(a + (width - plateau) / 2.0)
            c_right = float(b + plateau)
            # Clamp within the provided range
            a = max(a, range_min)
            d = min(d, range_max)
            # Ensure ordering a < b ≤ c < d
            b = max(b, a + 1e-6)
            c_right = min(c_right, d - 1e-6)
            if not (a < b <= c_right < d):
                # Fallback to a minimal valid shape around center
                eps = 1e-6
                a, b, c_right, d = c - 2 * eps, c - eps, c + eps, c + 2 * eps
            mfs.append(PiMF(a, b, c_right, d))
        return mfs

    def build(self) -> ANFIS:
        """Build the ANFIS model with configured parameters."""
        if not self.input_mfs:
            raise ValueError("No input variables defined. Use add_input() to define inputs.")

        return ANFIS(self.input_mfs)


class QuickANFIS:
    """Quick setup class for common ANFIS use cases."""

    @staticmethod
    def for_regression(
        X: np.ndarray,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        init: str = "grid",
        random_state: int | None = None,
    ) -> ANFIS:
        """Create ANFIS model automatically configured for regression data.

        Parameters:
            X: Input training data (n_samples, n_features)
            n_mfs: Number of membership functions per input
            mf_type: Type of membership functions
            init: Initialization strategy per input: 'grid' (default) or 'fcm'.
            random_state: Optional seed for deterministic FCM.

        Returns:
            Configured ANFIS model
        """
        if X.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features)")

        builder = ANFISBuilder()

        for i in range(X.shape[1]):
            col_data = X[:, i]
            if init.strip().lower() == "fcm":
                builder.add_input_from_data(
                    f"x{i + 1}",
                    col_data,
                    n_mfs=n_mfs,
                    mf_type=mf_type,
                    init="fcm",
                    random_state=random_state,
                )
            else:
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
