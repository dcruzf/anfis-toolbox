"""Builder classes for easy ANFIS model construction."""

import numpy as np

from .clustering import FuzzyCMeans
from .membership import (
    BellMF,
    DiffSigmoidalMF,
    Gaussian2MF,
    GaussianMF,
    LinSShapedMF,
    LinZShapedMF,
    PiMF,
    ProdSigmoidalMF,
    SigmoidalMF,
    SShapedMF,
    TrapezoidalMF,
    TriangularMF,
    ZShapedMF,
)
from .model import ANFIS, ANFISClassifier


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
            "gaussian2": self._create_gaussian2_mfs,
            "triangular": self._create_triangular_mfs,
            "trapezoidal": self._create_trapezoidal_mfs,
            "bell": self._create_bell_mfs,
            "sigmoidal": self._create_sigmoidal_mfs,
            "sshape": self._create_sshape_mfs,
            "zshape": self._create_zshape_mfs,
            "pi": self._create_pi_mfs,
            "linsshape": self._create_linsshape_mfs,
            "linzshape": self._create_linzshape_mfs,
            "diffsigmoidal": self._create_diff_sigmoidal_mfs,
            "prodsigmoidal": self._create_prod_sigmoidal_mfs,
            # Aliases
            "gbell": self._create_bell_mfs,
            "sigmoid": self._create_sigmoidal_mfs,
            "s": self._create_sshape_mfs,
            "z": self._create_zshape_mfs,
            "pimf": self._create_pi_mfs,
            "ls": self._create_linsshape_mfs,
            "lz": self._create_linzshape_mfs,
            "diffsigmoid": self._create_diff_sigmoidal_mfs,
            "prodsigmoid": self._create_prod_sigmoidal_mfs,
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
                'gaussian', 'gaussian2', 'triangular', 'trapezoidal',
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
    ) -> list:
        """Create MFs from 1D data via FCM.

        - Centers are FCM cluster centers.
                - Widths come from weighted within-cluster variance with weights U^m.
                - Supports: 'gaussian', 'bell'/'gbell', 'triangular', 'trapezoidal',
                    'sigmoidal'/'sigmoid', 'sshape'/'s', 'zshape'/'z', 'pi'/'pimf'.
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
        # Shared helpers
        rmin = float(np.min(x))
        rmax = float(np.max(x))
        # Map a width from sigma (ensure positive and with a reasonable floor)
        min_w = max(float(np.median(np.diff(np.sort(centers)))) if centers.size > 1 else float(np.std(x)), 1e-3)
        widths = np.maximum(2.0 * sigmas, min_w)

        if key == "gaussian":
            return [GaussianMF(mean=float(c), sigma=float(s)) for c, s in zip(centers, sigmas, strict=False)]
        if key == "gaussian2":
            mfs: list[Gaussian2MF] = []
            plateau_frac = 0.3
            for c, s, w in zip(centers, sigmas, widths, strict=False):
                half_plateau = (w * plateau_frac) / 2.0
                c1 = float(max(c - half_plateau, rmin))
                c2 = float(min(c + half_plateau, rmax))
                if not (c1 < c2):
                    eps = 1e-6
                    c1, c2 = c - eps, c + eps
                mfs.append(Gaussian2MF(sigma1=float(s), c1=c1, sigma2=float(s), c2=c2))
            return mfs
        if key in {"bell", "gbell"}:
            # map sigma to bell half-width a; keep b=2 by default
            return [BellMF(a=float(s), b=2.0, c=float(c)) for c, s in zip(centers, sigmas, strict=False)]
        if key == "triangular":
            mfs: list[TriangularMF] = []
            for c, w in zip(centers, widths, strict=False):
                a = float(max(c - w / 2.0, rmin))
                cc = float(min(c + w / 2.0, rmax))
                b = float(c)
                # ensure ordering a < b < c
                if not (a < b < cc):
                    eps = 1e-6
                    a, b, cc = c - 2 * eps, c, c + 2 * eps
                mfs.append(TriangularMF(a, b, cc))
            return mfs
        if key == "trapezoidal":
            mfs: list[TrapezoidalMF] = []
            plateau_frac = 0.3
            for c, w in zip(centers, widths, strict=False):
                a = float(c - w / 2.0)
                d = float(c + w / 2.0)
                b = float(a + (w * (1 - plateau_frac)) / 2.0)
                cr = float(b + w * plateau_frac)
                # clamp
                a = max(a, rmin)
                d = min(d, rmax)
                b = max(b, a + 1e-6)
                cr = min(cr, d - 1e-6)
                if not (a < b <= cr < d):
                    eps = 1e-6
                    a, b, cr, d = c - 2 * eps, c - eps, c + eps, c + 2 * eps
                mfs.append(TrapezoidalMF(a, b, cr, d))
            return mfs
        if key in {"sigmoidal", "sigmoid"}:
            # slope a from width: width ≈ 4.4 / a
            mfs: list[SigmoidalMF] = []
            for c, w in zip(centers, widths, strict=False):
                a = 4.4 / max(float(w), 1e-8)
                mfs.append(SigmoidalMF(a=float(a), c=float(c)))
            return mfs
        if key in {"linsshape", "ls"}:
            mfs: list[LinSShapedMF] = []
            for c, w in zip(centers, widths, strict=False):
                a = float(max(c - w / 2.0, rmin))
                b = float(min(c + w / 2.0, rmax))
                if a >= b:
                    eps = 1e-6
                    a, b = c - eps, c + eps
                mfs.append(LinSShapedMF(a, b))
            return mfs
        if key in {"linzshape", "lz"}:
            mfs: list[LinZShapedMF] = []
            for c, w in zip(centers, widths, strict=False):
                a = float(max(c - w / 2.0, rmin))
                b = float(min(c + w / 2.0, rmax))
                if a >= b:
                    eps = 1e-6
                    a, b = c - eps, c + eps
                mfs.append(LinZShapedMF(a, b))
            return mfs
        if key in {"diffsigmoidal", "diffsigmoid"}:
            # Create plateau-like bands: two increasing sigmoids with centers symmetric around c
            mfs: list[DiffSigmoidalMF] = []
            for c, w in zip(centers, widths, strict=False):
                c1 = float(max(c - w / 2.0, rmin))
                c2 = float(min(c + w / 2.0, rmax))
                if c1 >= c2:
                    eps = 1e-6
                    c1, c2 = c - eps, c + eps
                a = 4.4 / max(float(w), 1e-8)
                mfs.append(DiffSigmoidalMF(a1=float(a), c1=c1, a2=float(a), c2=c2))
            return mfs
        if key in {"prodsigmoidal", "prodsigmoid"}:
            # Create bump-like bands: product of increasing and decreasing sigmoid
            mfs: list[ProdSigmoidalMF] = []
            for c, w in zip(centers, widths, strict=False):
                c1 = float(max(c - w / 2.0, rmin))
                c2 = float(min(c + w / 2.0, rmax))
                if c1 >= c2:
                    eps = 1e-6
                    c1, c2 = c - eps, c + eps
                a = 4.4 / max(float(w), 1e-8)
                # s1 increasing at c1, s2 decreasing at c2 (use negative slope)
                mfs.append(ProdSigmoidalMF(a1=float(a), c1=c1, a2=float(-a), c2=c2))
            return mfs
        if key in {"sshape", "s"}:
            mfs: list[SShapedMF] = []
            for c, w in zip(centers, widths, strict=False):
                a = float(max(c - w / 2.0, rmin))
                b = float(min(c + w / 2.0, rmax))
                if a >= b:
                    eps = 1e-6
                    a, b = c - eps, c + eps
                mfs.append(SShapedMF(a, b))
            return mfs
        if key in {"zshape", "z"}:
            mfs: list[ZShapedMF] = []
            for c, w in zip(centers, widths, strict=False):
                a = float(max(c - w / 2.0, rmin))
                b = float(min(c + w / 2.0, rmax))
                if a >= b:
                    eps = 1e-6
                    a, b = c - eps, c + eps
                mfs.append(ZShapedMF(a, b))
            return mfs
        if key in {"pi", "pimf"}:
            mfs: list[PiMF] = []
            plateau_frac = 0.3
            for c, w in zip(centers, widths, strict=False):
                a = float(c - w / 2.0)
                d = float(c + w / 2.0)
                b = float(a + (w * (1 - plateau_frac)) / 2.0)
                cr = float(b + w * plateau_frac)
                # clamp and ensure ordering
                a = max(a, rmin)
                d = min(d, rmax)
                b = max(b, a + 1e-6)
                cr = min(cr, d - 1e-6)
                if not (a < b <= cr < d):
                    eps = 1e-6
                    a, b, cr, d = c - 2 * eps, c - eps, c + eps, c + 2 * eps
                mfs.append(PiMF(a, b, cr, d))
            return mfs
        supported = (
            "gaussian, gaussian2, bell/gbell, triangular, trapezoidal, sigmoidal/sigmoid, sshape/s, zshape/z, pi/pimf"
        )
        raise ValueError(f"FCM init supports: {supported}")

    def _create_gaussian_mfs(self, range_min: float, range_max: float, n_mfs: int, overlap: float) -> list[GaussianMF]:
        """Create evenly spaced Gaussian membership functions."""
        centers = np.linspace(range_min, range_max, n_mfs)

        # Handle single MF case
        if n_mfs == 1:
            sigma = (range_max - range_min) * 0.25  # Use quarter of range as default sigma
        else:
            sigma = (range_max - range_min) / (n_mfs - 1) * overlap

        return [GaussianMF(mean=center, sigma=sigma) for center in centers]

    def _create_gaussian2_mfs(
        self, range_min: float, range_max: float, n_mfs: int, overlap: float
    ) -> list[Gaussian2MF]:
        """Create evenly spaced two-sided Gaussian (Gaussian2) membership functions.

        Uses Gaussian tails with a small central plateau per MF. The plateau width
        is a fraction of the MF span controlled by overlap.
        """
        centers = np.linspace(range_min, range_max, n_mfs)

        # Determine spacing and widths
        if n_mfs == 1:
            spacing = range_max - range_min
            sigma = spacing * 0.25
            width = spacing * 0.5
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            sigma = spacing * overlap
            width = spacing * (1 + overlap)

        plateau_frac = 0.3
        half_plateau = (width * plateau_frac) / 2.0

        mfs: list[Gaussian2MF] = []
        for c in centers:
            c1 = float(max(c - half_plateau, range_min))
            c2 = float(min(c + half_plateau, range_max))
            if not (c1 < c2):
                eps = 1e-6
                c1, c2 = c - eps, c + eps
            mfs.append(Gaussian2MF(sigma1=float(sigma), c1=c1, sigma2=float(sigma), c2=c2))
        return mfs

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

    def _create_linsshape_mfs(
        self, range_min: float, range_max: float, n_mfs: int, overlap: float
    ) -> list[LinSShapedMF]:
        """Create linear S-shaped MFs across the range."""
        centers = np.linspace(range_min, range_max, n_mfs)
        if n_mfs == 1:
            width = (range_max - range_min) * 0.5
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            width = spacing * (1 + overlap)
        half = width / 2.0
        mfs: list[LinSShapedMF] = []
        for c in centers:
            a = float(max(c - half, range_min))
            b = float(min(c + half, range_max))
            if a >= b:
                eps = 1e-6
                a, b = c - eps, c + eps
            mfs.append(LinSShapedMF(a, b))
        return mfs

    def _create_linzshape_mfs(
        self, range_min: float, range_max: float, n_mfs: int, overlap: float
    ) -> list[LinZShapedMF]:
        """Create linear Z-shaped MFs across the range."""
        centers = np.linspace(range_min, range_max, n_mfs)
        if n_mfs == 1:
            width = (range_max - range_min) * 0.5
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            width = spacing * (1 + overlap)
        half = width / 2.0
        mfs: list[LinZShapedMF] = []
        for c in centers:
            a = float(max(c - half, range_min))
            b = float(min(c + half, range_max))
            if a >= b:
                eps = 1e-6
                a, b = c - eps, c + eps
            mfs.append(LinZShapedMF(a, b))
        return mfs

    def _create_diff_sigmoidal_mfs(
        self, range_min: float, range_max: float, n_mfs: int, overlap: float
    ) -> list[DiffSigmoidalMF]:
        """Create bands using difference of two sigmoids around evenly spaced centers."""
        centers = np.linspace(range_min, range_max, n_mfs)
        if n_mfs == 1:
            width = (range_max - range_min) * 0.5
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            width = spacing * (1 + overlap)
        mfs: list[DiffSigmoidalMF] = []
        for c in centers:
            c1 = float(max(c - width / 2.0, range_min))
            c2 = float(min(c + width / 2.0, range_max))
            if c1 >= c2:
                eps = 1e-6
                c1, c2 = c - eps, c + eps
            a = 4.4 / max(float(width), 1e-8)
            mfs.append(DiffSigmoidalMF(a1=float(a), c1=c1, a2=float(a), c2=c2))
        return mfs

    def _create_prod_sigmoidal_mfs(
        self, range_min: float, range_max: float, n_mfs: int, overlap: float
    ) -> list[ProdSigmoidalMF]:
        """Create product-of-sigmoids MFs; use increasing and decreasing pair to form a bump."""
        centers = np.linspace(range_min, range_max, n_mfs)
        if n_mfs == 1:
            width = (range_max - range_min) * 0.5
        else:
            spacing = (range_max - range_min) / (n_mfs - 1)
            width = spacing * (1 + overlap)
        mfs: list[ProdSigmoidalMF] = []
        for c in centers:
            c1 = float(max(c - width / 2.0, range_min))
            c2 = float(min(c + width / 2.0, range_max))
            if c1 >= c2:
                eps = 1e-6
                c1, c2 = c - eps, c + eps
            a = 4.4 / max(float(width), 1e-8)
            mfs.append(ProdSigmoidalMF(a1=float(a), c1=c1, a2=float(-a), c2=c2))
        return mfs

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

    @staticmethod
    def for_classification(
        X: np.ndarray,
        n_classes: int,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        init: str = "grid",
        random_state: int | None = None,
    ) -> ANFISClassifier:
        """Create ANFISClassifier configured from data.

        Mirrors for_regression but returns a classifier with n_classes.
        """
        if X.ndim != 2:
            raise ValueError("Input data must be 2D (n_samples, n_features)")

        builder = ANFISBuilder()
        for i in range(X.shape[1]):
            col_data = X[:, i]
            if init.strip().lower() == "fcm":
                builder.add_input_from_data(
                    f"x{i + 1}", col_data, n_mfs=n_mfs, mf_type=mf_type, init="fcm", random_state=random_state
                )
            else:
                range_min = float(np.min(col_data))
                range_max = float(np.max(col_data))
                margin = (range_max - range_min) * 0.1
                builder.add_input(f"x{i + 1}", range_min - margin, range_max + margin, n_mfs, mf_type)

        # Build as usual and wrap into classifier
        input_mfs = builder.input_mfs
        return ANFISClassifier(input_mfs, n_classes=n_classes)
