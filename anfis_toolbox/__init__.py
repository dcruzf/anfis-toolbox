"""ANFIS Toolbox - A Python toolbox for Adaptive Neuro-Fuzzy Inference Systems."""

__version__ = "0.1.0"

# Import builders for easy model creation
from .builders import ANFISBuilder, QuickANFIS
from .layers import ConsequentLayer, MembershipLayer, NormalizationLayer, RuleLayer
from .logging_config import disable_training_logs, enable_training_logs, setup_logging
from .membership import (
    BellMF,
    GaussianMF,
    MembershipFunction,
    PiMF,
    SigmoidalMF,
    SShapedMF,
    TrapezoidalMF,
    TriangularMF,
    ZShapedMF,
)
from .metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
    pearson_correlation,
    r2_score,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)
from .model import ANFIS

# Import validation utilities (optional)
try:
    from .validation import ANFISMetrics, ANFISValidator, quick_evaluate

    _HAS_VALIDATION = True
except ImportError:  # pragma: no cover
    _HAS_VALIDATION = False

    # Create dummy classes for documentation
    class ANFISValidator:
        """Dummy class for validation when scikit-learn is not available."""

        def __init__(self, *args, **kwargs):  # pragma: no cover
            """Initialize dummy validator."""
            raise ImportError(  # pragma: no cover
                "Validation features require scikit-learn. Install with: pip install anfis-toolbox[validation]"
            )

    class ANFISMetrics:
        """Dummy class for metrics when scikit-learn is not available."""

        @staticmethod
        def regression_metrics(*args, **kwargs):  # pragma: no cover
            """Dummy method for regression metrics when scikit-learn is not available."""
            raise ImportError(  # pragma: no cover
                "Validation features require scikit-learn. Install with: pip install anfis-toolbox[validation]"
            )

        @staticmethod
        def model_complexity_metrics(*args, **kwargs):  # pragma: no cover
            """Dummy method for model complexity metrics when scikit-learn is not available."""
            raise ImportError(  # pragma: no cover
                "Validation features require scikit-learn. Install with: pip install anfis-toolbox[validation]"
            )

    def quick_evaluate(*args, **kwargs):  # pragma: no cover
        """Dummy function for quick evaluation when scikit-learn is not available."""
        raise ImportError(  # pragma: no cover
            "Validation features require scikit-learn. Install with: pip install anfis-toolbox[validation]"
        )


# Optional imports with graceful fallback
try:
    from .visualization import ANFISVisualizer, quick_plot_results, quick_plot_training

    _HAS_VISUALIZATION = True
except ImportError:  # pragma: no cover
    _HAS_VISUALIZATION = False

    # Create dummy classes for documentation
    class ANFISVisualizer:
        """Dummy class for visualization when matplotlib is not available."""

        def __init__(self, *args, **kwargs):  # pragma: no cover
            """Initialize dummy visualizer."""
            raise ImportError(  # pragma: no cover
                "Visualization features require matplotlib. Install with: pip install anfis-toolbox[visualization]"
            )

    def quick_plot_training(*args, **kwargs):  # pragma: no cover
        """Dummy function for plotting training curves when matplotlib is not available."""
        raise ImportError(  # pragma: no cover
            "Visualization features require matplotlib. Install with: pip install anfis-toolbox[visualization]"
        )

    def quick_plot_results(*args, **kwargs):  # pragma: no cover
        """Dummy function for plotting results when matplotlib is not available."""
        raise ImportError(  # pragma: no cover
            "Visualization features require matplotlib. Install with: pip install anfis-toolbox[visualization]"
        )


__all__ = [
    # Core components
    "ANFIS",
    "MembershipFunction",
    "GaussianMF",
    "TriangularMF",
    "TrapezoidalMF",
    "BellMF",
    "SigmoidalMF",
    "PiMF",
    "SShapedMF",
    "ZShapedMF",
    "MembershipLayer",
    "RuleLayer",
    "NormalizationLayer",
    "ConsequentLayer",
    # Logging
    "setup_logging",
    "enable_training_logs",
    "disable_training_logs",
    # Easy model creation
    "ANFISBuilder",
    "QuickANFIS",
    # Metrics
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "symmetric_mean_absolute_percentage_error",
    "r2_score",
    "pearson_correlation",
    "mean_squared_logarithmic_error",
    # Validation and metrics
    "ANFISValidator",
    "ANFISMetrics",
    "quick_evaluate",
    # Visualization (may raise ImportError if matplotlib not available)
    "ANFISVisualizer",
    "quick_plot_training",
    "quick_plot_results",
]
