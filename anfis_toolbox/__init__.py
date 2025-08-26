"""ANFIS Toolbox - A Python toolbox for Adaptive Neuro-Fuzzy Inference Systems."""

__version__ = "0.1.0"

# Import builders for easy model creation
from .builders import ANFISBuilder, QuickANFIS
from .clustering import FuzzyCMeans
from .layers import ConsequentLayer, MembershipLayer, NormalizationLayer, RuleLayer
from .logging_config import disable_training_logs, enable_training_logs, setup_logging
from .losses import cross_entropy_grad, cross_entropy_loss, mse_grad, mse_loss
from .membership import (
    BellMF,
    Gaussian2MF,
    GaussianMF,
    LinSShapedMF,
    LinZShapedMF,
    MembershipFunction,
    PiMF,
    SigmoidalMF,
    SShapedMF,
    TrapezoidalMF,
    TriangularMF,
    ZShapedMF,
)
from .metrics import (
    accuracy,
    classification_entropy,
    cross_entropy,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
    partition_coefficient,
    pearson_correlation,
    r2_score,
    root_mean_squared_error,
    softmax,
    symmetric_mean_absolute_percentage_error,
    xie_beni_index,
)
from .model import ANFIS, ANFISClassifier
from .model_selection import KFold, train_test_split
from .optim import HybridTrainer, SGDTrainer
from .validation import ANFISMetrics, ANFISValidator, quick_evaluate

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
    "ANFISClassifier",
    "MembershipFunction",
    "GaussianMF",
    "Gaussian2MF",
    "TriangularMF",
    "TrapezoidalMF",
    "BellMF",
    "SigmoidalMF",
    "PiMF",
    "LinSShapedMF",
    "LinZShapedMF",
    "SShapedMF",
    "ZShapedMF",
    "MembershipLayer",
    "RuleLayer",
    "NormalizationLayer",
    "ConsequentLayer",
    # Optimizers/trainers
    "SGDTrainer",
    "HybridTrainer",
    # Logging
    "setup_logging",
    "enable_training_logs",
    "disable_training_logs",
    # Easy model creation
    "ANFISBuilder",
    "QuickANFIS",
    # Metrics
    "softmax",
    "cross_entropy",
    "log_loss",
    # Losses (training objectives)
    "mse_loss",
    "mse_grad",
    "cross_entropy_loss",
    "cross_entropy_grad",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "symmetric_mean_absolute_percentage_error",
    "r2_score",
    "pearson_correlation",
    "mean_squared_logarithmic_error",
    "accuracy",
    # Clustering metrics
    "partition_coefficient",
    "classification_entropy",
    "xie_beni_index",
    # Validation and metrics
    "ANFISValidator",
    "ANFISMetrics",
    "quick_evaluate",
    # Model selection helpers
    "KFold",
    "train_test_split",
    # Visualization (may raise ImportError if matplotlib not available)
    "ANFISVisualizer",
    "quick_plot_training",
    "quick_plot_results",
    # Clustering
    "FuzzyCMeans",
]
