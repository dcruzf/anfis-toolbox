"""Clear, minimal visualization utilities for EDA and ANFIS regression.

This module provides a small set of focused plotting functions for:
- Exploring data before fitting (histograms, feature-vs-target, correlations)
- Assessing model results (training curve, predictions vs target, residuals)
- Inspecting ANFIS internals (membership functions, rule activations)

All functions return a Matplotlib Figure and never call plt.show(), so users and
tests can control rendering. A light wrapper class (ANFISVisualizer) offers the
same API style if you prefer an OO approach.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .membership import MembershipFunction
from .model import ANFIS

# -----------------------------
# EDA helpers (pre-model)
# -----------------------------


def plot_feature_histograms(
    X: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    bins: int = 30,
    max_cols: int = 4,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> Figure:
    """Plot histograms for each feature.

    Args:
        X: Array of shape (n_samples, n_features) with the input features.
        feature_names: Optional list with one name per feature; defaults to x1..xN.
        bins: Number of histogram bins.
        max_cols: Maximum number of columns in the subplot grid.
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure containing the histogram grid.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    n_features = X.shape[1]
    cols = min(max_cols, n_features)
    rows = int(np.ceil(n_features / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4 * cols, 3 * rows), squeeze=False)
    names = feature_names or [f"x{i + 1}" for i in range(n_features)]
    for j in range(rows * cols):
        r, c = divmod(j, cols)
        ax = axes[r, c]
        if j < n_features:
            ax.hist(X[:, j], bins=bins, color="steelblue", alpha=0.8)
            ax.set_title(f"{names[j]}")
            ax.grid(True, alpha=0.2)
        else:
            ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_vs_target(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    max_cols: int = 3,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> Figure:
    """Scatter each feature vs target to reveal relationships/outliers.

    Args:
        X: Array of shape (n_samples, n_features) with the input features.
        y: Array-like of shape (n_samples,) with the target values.
        feature_names: Optional list with one name per feature; defaults to x1..xN.
        max_cols: Maximum number of columns in the subplot grid.
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure containing the scatter grid.

    Raises:
        ValueError: If X is not 2D or X/y lengths mismatch.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    n_features = X.shape[1]
    cols = min(max_cols, n_features)
    rows = int(np.ceil(n_features / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (4 * cols, 3 * rows), squeeze=False)
    names = feature_names or [f"x{i + 1}" for i in range(n_features)]
    for j in range(rows * cols):
        r, c = divmod(j, cols)
        ax = axes[r, c]
        if j < n_features:
            ax.scatter(X[:, j], y, s=12, alpha=0.7)
            ax.set_xlabel(names[j])
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.2)
        else:
            ax.axis("off")
    fig.suptitle("Feature vs Target", y=0.98)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    feature_names: list[str] | None = None,
    include_target: bool = True,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> Figure:
    """Correlation matrix heatmap for features (optionally including target).

    Args:
        X: Array of shape (n_samples, n_features) with the input features.
        y: Optional target array of shape (n_samples,) to include in the heatmap.
        feature_names: Optional list with one name per feature; defaults to x1..xN.
        include_target: Whether to include y (when provided) in the correlation matrix.
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure with the correlation heatmap.

    Raises:
        ValueError: If X is not 2D.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    data = X
    labels = feature_names or [f"x{i + 1}" for i in range(X.shape[1])]
    if include_target and y is not None:
        y1 = np.asarray(y).reshape(-1, 1)
        data = np.hstack([X, y1])
        labels = [*labels, "y"]
    corr = np.corrcoef(data, rowvar=False)
    fig, ax = plt.subplots(figsize=figsize or (3 + len(labels) * 0.6, 3 + len(labels) * 0.6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_target_distribution(
    y: np.ndarray, *, bins: int = 30, figsize: tuple[int, int] | None = None, save_path: str | None = None
) -> Figure:
    """Histogram view of the target distribution.

    Args:
        y: Array-like of shape (n_samples,) with the target values.
        bins: Number of histogram bins.
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure containing the histogram.
    """
    y = np.asarray(y).reshape(-1)
    fig, ax = plt.subplots(figsize=figsize or (6, 4))
    ax.hist(y, bins=bins, color="slateblue", alpha=0.85)
    ax.set_title("Target Distribution")
    ax.set_xlabel("y")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ----------------------------------
# Model assessment (post-fit)
# ----------------------------------


def plot_training_curve(
    losses: list[float], *, figsize: tuple[int, int] | None = None, save_path: str | None = None
) -> Figure:
    """Plot training loss per epoch.

    Args:
        losses: Sequence of loss values, one per epoch.
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure with the loss curve.
    """
    fig, ax = plt.subplots(figsize=figsize or (7, 4))
    ax.plot(range(1, len(losses) + 1), losses, "b-", linewidth=2, label="Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_predictions_vs_target(
    y_true: np.ndarray, y_pred: np.ndarray, *, figsize: tuple[int, int] | None = None, save_path: str | None = None
) -> Figure:
    """Scatter of predictions vs true values with identity line and R^2.

    Args:
        y_true: Array-like of shape (n_samples,) with ground-truth values.
        y_pred: Array-like of shape (n_samples,) with predicted values.
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure showing y_true vs y_pred.

    Raises:
        ValueError: If y_true and y_pred lengths differ.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    fig, ax = plt.subplots(figsize=figsize or (6, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    ax.set_title("Predictions vs True")
    # R^2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.8})
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, *, figsize: tuple[int, int] | None = None, save_path: str | None = None
) -> Figure:
    """Residuals vs prediction scatter and residual histogram.

    Args:
        y_true: Array-like of shape (n_samples,) with ground-truth values.
        y_pred: Array-like of shape (n_samples,) with predicted values.
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure with residual diagnostics.

    Raises:
        ValueError: If y_true and y_pred lengths differ.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    resid = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (10, 4))
    ax1.scatter(y_pred, resid, s=20, alpha=0.6)
    ax1.axhline(0.0, color="r", linestyle="--", linewidth=1)
    ax1.set_xlabel("Pred")
    ax1.set_ylabel("Residual")
    ax1.set_title("Residuals vs Pred")
    ax1.grid(True, alpha=0.3)
    ax2.hist(resid, bins=30, color="teal", alpha=0.8)
    ax2.set_title("Residuals Histogram")
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_predictions_for_model(
    model: ANFIS,
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> Figure:
    """Compute predictions from the model then call plot_predictions_vs_target.

    Args:
        model: Trained ANFIS model used to compute predictions.
        X: Input array of shape (n_samples, n_features).
        y_true: Ground-truth array of shape (n_samples,) or (n_samples, 1).
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure showing y_true vs model predictions.
    """
    y_pred = model.predict(X)
    return plot_predictions_vs_target(y_true, y_pred, figsize=figsize, save_path=save_path)


def plot_residuals_for_model(
    model: ANFIS,
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> Figure:
    """Compute predictions from the model then call plot_residuals.

    Args:
        model: Trained ANFIS model used to compute predictions.
        X: Input array of shape (n_samples, n_features).
        y_true: Ground-truth array of shape (n_samples,) or (n_samples, 1).
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure with residual diagnostics.
    """
    y_pred = model.predict(X)
    return plot_residuals(y_true, y_pred, figsize=figsize, save_path=save_path)


# ----------------------------------
# ANFIS internals
# ----------------------------------


def plot_membership_functions(
    model: ANFIS,
    *,
    input_name: str | None = None,
    num_points: int = 500,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> Figure:
    """Plot membership functions for all inputs or a specific input.

    Args:
        model: ANFIS model providing membership functions.
        input_name: If provided, only plot MFs for this named input.
        num_points: Number of points to sample for each MF curve.
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure with one subplot per input.

    Raises:
        ValueError: If input_name is provided but not found in the model.
    """
    input_mfs = model.membership_layer.membership_functions
    if input_name is not None:
        if input_name not in input_mfs:
            raise ValueError(f"Input '{input_name}' not found in model")
        targets = {input_name: input_mfs[input_name]}
    else:
        targets = input_mfs
    n = len(targets)
    fig, axes = plt.subplots(n, 1, figsize=figsize or (8, max(2 * n, 3)), squeeze=False)
    for row, (name, mfs) in enumerate(targets.items()):
        ax = axes[row, 0]
        x_lo, x_hi = _infer_mf_range(mfs)
        x = np.linspace(x_lo, x_hi, num_points)
        for i, mf in enumerate(mfs):
            ax.plot(x, mf.forward(x), label=f"MF{i + 1}")
        ax.set_title(f"Membership: {name}")
        ax.set_xlabel("x")
        ax.set_ylabel("μ")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.legend(ncol=min(4, len(mfs)))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_rule_activations(
    model: ANFIS,
    X: np.ndarray,
    *,
    sample_idx: int = 0,
    figsize: tuple[int, int] | None = None,
    save_path: str | None = None,
) -> Figure:
    """Bar plots of raw and normalized rule activations for one sample.

    Args:
        model: ANFIS model to compute rule activations.
        X: Input array of shape (n_samples, n_features).
        sample_idx: Index of the sample to visualize.
        figsize: Optional custom figure size.
        save_path: Optional path to save the figure to disk.

    Returns:
        A Matplotlib Figure with two bar plots (raw and normalized activations).

    Raises:
        ValueError: If sample_idx is out of range for X.
    """
    if not (0 <= sample_idx < len(X)):
        raise ValueError("sample_idx out of range")
    x = X[sample_idx : sample_idx + 1]
    mem = model.membership_layer.forward(x)
    rule = model.rule_layer.forward(mem)
    norm = model.normalization_layer.forward(rule)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (10, 4))
    idx = np.arange(1, rule.shape[1] + 1)
    ax1.bar(idx, rule[0], color="skyblue", alpha=0.8)
    ax1.set_title("Firing Strengths")
    ax1.set_xlabel("Rule")
    ax1.set_ylabel("Strength")
    ax1.grid(True, alpha=0.2)
    ax2.bar(idx, norm[0], color="salmon", alpha=0.8)
    ax2.set_title("Normalized Activations")
    ax2.set_xlabel("Rule")
    ax2.set_ylabel("Activation")
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def _infer_mf_range(mfs: list[MembershipFunction]) -> tuple[float, float]:
    """Best-effort input range for plotting membership functions."""
    mins, maxs = [], []
    for mf in mfs:
        p = mf.parameters
        name = mf.__class__.__name__.lower()
        if "gaussian" in name:
            mean = float(p.get("mean", 0.0))
            sigma = float(p.get("sigma", 1.0))
            mins.append(mean - 3 * sigma)
            maxs.append(mean + 3 * sigma)
        elif "triangular" in name:
            mins.append(float(p.get("a", -1)))
            maxs.append(float(p.get("c", 1)))
        elif "trapezoidal" in name:
            mins.append(float(p.get("a", -1)))
            maxs.append(float(p.get("d", 1)))
        elif "bell" in name:
            c = float(p.get("c", 0.0))
            a = float(p.get("a", 1.0))
            mins.append(c - 3 * a)
            maxs.append(c + 3 * a)
        else:
            mins.append(-5.0)
            maxs.append(5.0)
    lo = min(mins) if mins else -5.0
    hi = max(maxs) if maxs else 5.0
    pad = (hi - lo) * 0.1
    return (lo - pad, hi + pad)


# -----------------------------
# OO convenience wrapper
# -----------------------------


class ANFISVisualizer:
    """Thin OO wrapper around the functional API."""

    def __init__(self, model: ANFIS):
        """Create a visualizer bound to a given ANFIS model.

        Args:
            model: The ANFIS instance to use for model-based plots.
        """
        self.model = model

    # EDA
    def plot_feature_histograms(self, X: np.ndarray, **kwargs) -> Figure:  # pragma: no cover - thin wrapper
        """Wrapper for plot_feature_histograms.

        Args:
            X: Input features array.
            **kwargs: Forwarded to plot_feature_histograms.
        """
        return plot_feature_histograms(X, **kwargs)

    def plot_feature_vs_target(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Figure:  # pragma: no cover
        """Wrapper for plot_feature_vs_target.

        Args:
            X: Input features array.
            y: Target array.
            **kwargs: Forwarded to plot_feature_vs_target.
        """
        return plot_feature_vs_target(X, y, **kwargs)

    def plot_correlation_heatmap(
        self, X: np.ndarray, y: np.ndarray | None = None, **kwargs
    ) -> Figure:  # pragma: no cover
        """Wrapper for plot_correlation_heatmap.

        Args:
            X: Input features array.
            y: Optional target array.
            **kwargs: Forwarded to plot_correlation_heatmap.
        """
        return plot_correlation_heatmap(X, y, **kwargs)

    def plot_target_distribution(self, y: np.ndarray, **kwargs) -> Figure:  # pragma: no cover
        """Wrapper for plot_target_distribution.

        Args:
            y: Target array.
            **kwargs: Forwarded to plot_target_distribution.
        """
        return plot_target_distribution(y, **kwargs)

    # Model assessment
    def plot_training_curves(self, losses: list[float], **kwargs) -> Figure:  # pragma: no cover
        """Wrapper for plot_training_curve.

        Args:
            losses: Sequence of losses.
            **kwargs: Forwarded to plot_training_curve.
        """
        return plot_training_curve(losses, **kwargs)

    def plot_prediction_vs_target(self, X: np.ndarray, y_true: np.ndarray, **kwargs) -> Figure:  # pragma: no cover
        """Predict with bound model and call plot_predictions_vs_target.

        Args:
            X: Input features array.
            y_true: Target array.
            **kwargs: Forwarded to plot_predictions_vs_target.
        """
        y_pred = self.model.predict(X)
        return plot_predictions_vs_target(y_true, y_pred, **kwargs)

    def plot_residuals(self, X: np.ndarray, y_true: np.ndarray, **kwargs) -> Figure:  # pragma: no cover
        """Predict with bound model and call plot_residuals.

        Args:
            X: Input features array.
            y_true: Target array.
            **kwargs: Forwarded to plot_residuals.
        """
        y_pred = self.model.predict(X)
        return plot_residuals(y_true, y_pred, **kwargs)

    def plot_1d_function_approximation(
        self, X: np.ndarray, y_true: np.ndarray, n_points: int = 200, **kwargs
    ) -> Figure:  # pragma: no cover
        """Plot 1D function approximation curve and training data using the model.

        Args:
            X: Input data with a single feature, shape (n_samples, 1).
            y_true: Target values, shape (n_samples,) or (n_samples, 1).
            n_points: Number of points for the smooth prediction curve.
            **kwargs: Forwarded to Matplotlib (e.g., figsize=(w, h)).

        Raises:
            ValueError: If X does not have exactly one feature.
        """
        if X.shape[1] != 1:
            raise ValueError("This method is only for 1D inputs")
        x_min, x_max = X.min(), X.max()
        x_smooth = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
        y_smooth = self.model.predict(x_smooth)
        fig, ax = plt.subplots(figsize=kwargs.get("figsize") or (8, 4))
        ax.scatter(X.flatten(), y_true.flatten(), alpha=0.6, s=30, color="red", label="Data")
        ax.plot(x_smooth.flatten(), y_smooth.flatten(), "b-", linewidth=2, label="ANFIS")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Function Approximation (1D)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    # Internals
    def plot_membership_functions(self, **kwargs) -> Figure:  # pragma: no cover
        """Wrapper for plot_membership_functions.

        Args:
            **kwargs: Forwarded to plot_membership_functions.
        """
        return plot_membership_functions(self.model, **kwargs)

    def plot_rule_activations(self, X: np.ndarray, **kwargs) -> Figure:  # pragma: no cover
        """Wrapper for plot_rule_activations.

        Args:
            X: Input features array.
            **kwargs: Forwarded to plot_rule_activations.
        """
        return plot_rule_activations(self.model, X, **kwargs)


# -----------------------------
# Quick helpers
# -----------------------------


def quick_plot_training(losses: list[float], save_path: str | None = None, *, show: bool = False) -> Figure:
    """Quick plot for training curve.

    Args:
        losses: Sequence of loss values, one per epoch.
        save_path: Optional path to save the figure to disk.
        show: If True and backend is interactive, call plt.show().

    Returns:
        A Matplotlib Figure with the loss curve.
    """
    fig = plot_training_curve(losses)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and "agg" not in str(mpl.get_backend()).lower():
        plt.show()
    return fig


def quick_plot_results(
    X: np.ndarray, y_true: np.ndarray, model: ANFIS, save_path: str | None = None, *, show: bool = False
) -> Figure:
    """Quick plot for predictions vs target using a model.

    Args:
        X: Input features array.
        y_true: Ground-truth target array.
        model: Trained ANFIS model used to compute predictions.
        save_path: Optional path to save the figure to disk.
        show: If True and backend is interactive, call plt.show().

    Returns:
        A Matplotlib Figure showing y_true vs model predictions.
    """
    fig = plot_predictions_for_model(model, X, y_true)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and "agg" not in str(mpl.get_backend()).lower():
        plt.show()
    return fig
