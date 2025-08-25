"""Clustering utilities (no external deps).

Currently includes Fuzzy C-Means (FCM).
"""

from __future__ import annotations

import numpy as np


class FuzzyCMeans:
    """Fuzzy C-Means clustering.

    Parameters:
        n_clusters: Number of clusters (>= 2).
        m: Fuzzifier (> 1). Default 2.0.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on centers.
        random_state: Optional seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        """Initialize FuzzyCMeans with hyperparameters."""
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if m <= 1:
            raise ValueError("m (fuzzifier) must be > 1")
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.cluster_centers_ = None
        self.membership_ = None

    # ---------------------
    # Helpers
    # ---------------------
    def _rng(self) -> np.random.RandomState:
        return np.random.RandomState(self.random_state)

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D array-like")
        if X.shape[0] < self.n_clusters:
            # FCM can run, but it's ill-conditioned; keep explicit error to avoid degenerate solutions
            raise ValueError("n_samples must be >= n_clusters")
        return X

    def _init_membership(self, n_samples: int) -> np.ndarray:
        rng = self._rng()
        U = rng.rand(n_samples, self.n_clusters)
        U /= np.sum(U, axis=1, keepdims=True)
        return U

    @staticmethod
    def _pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        # (n,d) vs (k,d) -> (n,k)
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    # ---------------------
    # Public API
    # ---------------------
    def fit(self, X: np.ndarray) -> FuzzyCMeans:
        """Fit the FCM model.

        Sets cluster_centers_ (k,d) and membership_ (n,k).
        """
        X = self._check_X(X)
        n, _ = X.shape
        U = self._init_membership(n)
        m = self.m

        def update_centers(Um: np.ndarray) -> np.ndarray:
            num = Um.T @ X  # (k,d)
            den = np.maximum(Um.sum(axis=0)[:, None], 1e-12)
            return num / den

        Um = U**m
        C = update_centers(Um)
        for _ in range(self.max_iter):
            d2 = np.maximum(self._pairwise_sq_dists(X, C), 1e-12)  # (n,k)
            inv = d2 ** (-1.0 / (m - 1.0))
            U_new = inv / np.sum(inv, axis=1, keepdims=True)
            Um_new = U_new**m
            C_new = update_centers(Um_new)
            if np.max(np.linalg.norm(C_new - C, axis=1)) < self.tol:
                U, C = U_new, C_new
                break
            U, C = U_new, C_new
        self.membership_ = U
        self.cluster_centers_ = C
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return hard labels via argmax of membership."""
        self.fit(X)
        return self.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard labels via argmax of predict_proba."""
        U = self.predict_proba(X)
        return np.argmax(U, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return membership degrees for samples to clusters (rows sum to 1)."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = self._check_X(X)
        C = self.cluster_centers_
        m = self.m
        d2 = np.maximum(self._pairwise_sq_dists(X, C), 1e-12)
        inv = d2 ** (-1.0 / (m - 1.0))
        return inv / np.sum(inv, axis=1, keepdims=True)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_proba."""
        return self.predict_proba(X)

    # Metrics
    def partition_coefficient(self) -> float:
        """Bezdek's Partition Coefficient (PC) in [1/k, 1]. Higher is crisper."""
        if self.membership_ is None:
            raise RuntimeError("Fit the model before calling partition_coefficient().")
        U = self.membership_
        n = U.shape[0]
        return float(np.sum(U**2) / n)

    def classification_entropy(self) -> float:
        """Classification Entropy (CE). Lower is better (crisper)."""
        if self.membership_ is None:
            raise RuntimeError("Fit the model before calling classification_entropy().")
        U = np.clip(self.membership_, 1e-12, 1.0)
        n = U.shape[0]
        return float(-np.sum(U * np.log(U)) / n)

    def xie_beni_index(self, X: np.ndarray) -> float:
        """Xie–Beni index (XB). Lower is better.

        XB = sum_i sum_k u_ik^m ||x_i - v_k||^2 / (n * min_{p!=q} ||v_p - v_q||^2)
        """
        if self.membership_ is None or self.cluster_centers_ is None:
            raise RuntimeError("Fit the model before calling xie_beni_index().")
        X = self._check_X(X)
        U = self.membership_
        C = self.cluster_centers_
        m = self.m
        d2 = self._pairwise_sq_dists(X, C)
        num = float(np.sum((U**m) * d2))
        # min squared distance between distinct centers
        if C.shape[0] < 2:
            return np.inf
        diffs = C[:, None, :] - C[None, :, :]
        dist2 = (diffs**2).sum(axis=2)
        dist2 += np.eye(C.shape[0]) * np.inf
        den = float(np.min(dist2))
        den = max(den, 1e-12)
        return num / (X.shape[0] * den)
