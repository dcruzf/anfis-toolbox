"""Visualization utilities for ANFIS models."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .membership import MembershipFunction
from .model import ANFIS


class ANFISVisualizer:
    """Visualization utilities for ANFIS models."""

    def __init__(self, model: ANFIS):
        """Initialize visualizer with ANFIS model."""
        self.model = model

    def plot_membership_functions(self, input_name: str = None, figsize: tuple[int, int] = (12, 8)) -> Figure:
        """Plot membership functions for all inputs or specific input.

        Parameters:
            input_name: Name of specific input to plot (None for all)
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        input_mfs = self.model.membership_layer.membership_functions

        if input_name:
            if input_name not in input_mfs:
                raise ValueError(f"Input '{input_name}' not found in model")
            inputs_to_plot = {input_name: input_mfs[input_name]}
        else:
            inputs_to_plot = input_mfs

        n_inputs = len(inputs_to_plot)
        fig, axes = plt.subplots(n_inputs, 1, figsize=figsize)

        if n_inputs == 1:
            axes = [axes]

        for idx, (name, mfs) in enumerate(inputs_to_plot.items()):
            ax = axes[idx]

            # Determine input range for plotting
            x_range = self._get_input_range(mfs)
            x = np.linspace(x_range[0], x_range[1], 500)

            # Plot each membership function
            for i, mf in enumerate(mfs):
                y = mf.forward(x)
                ax.plot(x, y, label=f"MF{i + 1}", linewidth=2)

            ax.set_title(f"Membership Functions for Input: {name}")
            ax.set_xlabel("Input Value")
            ax.set_ylabel("Membership Degree")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1.1)

        plt.tight_layout()
        return fig

    def plot_training_curves(self, losses: list[float], figsize: tuple[int, int] = (10, 6)) -> Figure:
        """Plot training loss curves.

        Parameters:
            losses: List of loss values from training
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, "b-", linewidth=2, label="Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("ANFIS Training Progress")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add final loss annotation
        final_loss = losses[-1]
        ax.annotate(
            f"Final Loss: {final_loss:.6f}",
            xy=(len(losses), final_loss),
            xytext=(len(losses) * 0.7, max(losses) * 0.8),
            arrowprops={"arrowstyle": "->", "color": "red"},
            fontsize=10,
            ha="center",
        )

        plt.tight_layout()
        return fig

    def plot_prediction_vs_target(
        self, X: np.ndarray, y_true: np.ndarray, figsize: tuple[int, int] = (10, 6)
    ) -> Figure:
        """Plot predictions vs actual values.

        Parameters:
            X: Input data
            y_true: True target values
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        y_pred = self.model.predict(X)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Prediction vs Target scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6, s=30)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
        ax1.set_xlabel("True Values")
        ax1.set_ylabel("Predictions")
        ax1.set_title("Predictions vs True Values")
        ax1.grid(True, alpha=0.3)

        # Calculate R²
        r_squared = self._calculate_r_squared(y_true, y_pred)
        ax1.text(
            0.05,
            0.95,
            f"R² = {r_squared:.4f}",
            transform=ax1.transAxes,
            fontsize=12,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax2.axhline(y=0, color="r", linestyle="--")
        ax2.set_xlabel("Predictions")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_1d_function_approximation(
        self, X: np.ndarray, y_true: np.ndarray, n_points: int = 200, figsize: tuple[int, int] = (12, 6)
    ) -> Figure:
        """Plot 1D function approximation results.

        Parameters:
            X: Input data (must be 1D)
            y_true: True target values
            n_points: Number of points for smooth curve
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        if X.shape[1] != 1:
            raise ValueError("This method is only for 1D inputs")

        # Create smooth curve for visualization
        x_min, x_max = X.min(), X.max()
        x_smooth = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
        y_smooth = self.model.predict(x_smooth)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot training data
        ax.scatter(X.flatten(), y_true.flatten(), alpha=0.6, s=50, color="red", label="Training Data")

        # Plot ANFIS approximation
        ax.plot(x_smooth.flatten(), y_smooth.flatten(), "b-", linewidth=2, label="ANFIS Approximation")

        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.set_title("ANFIS Function Approximation")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_rule_activations(self, X: np.ndarray, sample_idx: int = 0, figsize: tuple[int, int] = (10, 6)) -> Figure:
        """Plot rule activation levels for a specific sample.

        Parameters:
            X: Input data
            sample_idx: Index of sample to analyze
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} out of range")

        # Get rule activations
        x_sample = X[sample_idx : sample_idx + 1]  # Keep 2D shape

        # Forward pass to get intermediate outputs
        membership_output = self.model.membership_layer.forward(x_sample)
        rule_output = self.model.rule_layer.forward(membership_output)
        normalized_output = self.model.normalization_layer.forward(rule_output)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Rule firing strengths
        rule_strengths = rule_output[0]  # First (only) sample
        rule_numbers = range(1, len(rule_strengths) + 1)

        bars1 = ax1.bar(rule_numbers, rule_strengths, alpha=0.7, color="skyblue")
        ax1.set_xlabel("Rule Number")
        ax1.set_ylabel("Firing Strength")
        ax1.set_title("Rule Firing Strengths")
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, strength in zip(bars1, rule_strengths):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{strength:.3f}", ha="center", va="bottom")

        # Normalized rule activations
        normalized_strengths = normalized_output[0]  # First (only) sample

        bars2 = ax2.bar(rule_numbers, normalized_strengths, alpha=0.7, color="lightcoral")
        ax2.set_xlabel("Rule Number")
        ax2.set_ylabel("Normalized Activation")
        ax2.set_title("Normalized Rule Activations")
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, strength in zip(bars2, normalized_strengths):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{strength:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        return fig

    def _get_input_range(self, mfs: list[MembershipFunction]) -> tuple[float, float]:
        """Estimate appropriate range for plotting membership functions."""
        # Try to extract range information from MF parameters
        min_vals, max_vals = [], []

        for mf in mfs:
            params = mf.parameters

            if hasattr(mf, "__class__") and "Gaussian" in mf.__class__.__name__:
                # For Gaussian: mean ± 3*sigma covers ~99.7% of the function
                mean = params.get("mean", 0)
                sigma = params.get("sigma", 1)
                min_vals.append(mean - 3 * sigma)
                max_vals.append(mean + 3 * sigma)

            elif hasattr(mf, "__class__") and "Triangular" in mf.__class__.__name__:
                # For Triangular: use a, c parameters
                min_vals.append(params.get("a", -1))
                max_vals.append(params.get("c", 1))

            elif hasattr(mf, "__class__") and "Trapezoidal" in mf.__class__.__name__:
                # For Trapezoidal: use a, d parameters
                min_vals.append(params.get("a", -1))
                max_vals.append(params.get("d", 1))

            else:
                # Default range
                min_vals.append(-5)
                max_vals.append(5)

        range_min = min(min_vals) if min_vals else -5
        range_max = max(max_vals) if max_vals else 5

        # Add some margin
        margin = (range_max - range_min) * 0.1
        return (range_min - margin, range_max + margin)

    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared coefficient of determination."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return r_squared


def quick_plot_training(losses: list[float], save_path: Optional[str] = None):
    """Quick function to plot training curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, "b-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def quick_plot_results(X: np.ndarray, y_true: np.ndarray, model: ANFIS, save_path: Optional[str] = None):
    """Quick function to plot prediction results."""
    visualizer = ANFISVisualizer(model)

    if X.shape[1] == 1:
        # 1D function approximation
        _fig = visualizer.plot_1d_function_approximation(X, y_true)
    else:
        # Multi-dimensional: use prediction vs target plot
        _fig = visualizer.plot_prediction_vs_target(X, y_true)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
