"""ANFIS Toolbox - A Python toolbox for Adaptive Neuro-Fuzzy Inference Systems."""

__version__ = "0.1.0"

# Import builders for easy model creation
from .builders import ANFISBuilder, QuickANFIS
from .clustering import FuzzyCMeans
from .layers import ConsequentLayer, MembershipLayer, NormalizationLayer, RuleLayer
from .logging_config import disable_training_logs, enable_training_logs, setup_logging
from .losses import (
    CrossEntropyLoss,
    LossFunction,
    MSELoss,
    cross_entropy_grad,
    cross_entropy_loss,
    mse_grad,
    mse_loss,
    resolve_loss,
)
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
    ANFISMetrics,
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
    quick_evaluate,
    r2_score,
    root_mean_squared_error,
    softmax,
    symmetric_mean_absolute_percentage_error,
    xie_beni_index,
)
from .model import ANFIS, ANFISClassifier
from .optim import AdamTrainer, HybridTrainer, PSOTrainer, RMSPropTrainer, SGDTrainer

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
    "AdamTrainer",
    "RMSPropTrainer",
    "PSOTrainer",
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
    "LossFunction",
    "MSELoss",
    "CrossEntropyLoss",
    "resolve_loss",
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
    "ANFISMetrics",
    "quick_evaluate",
    # Clustering
    "FuzzyCMeans",
]
