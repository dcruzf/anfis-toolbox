"""ANFIS Toolbox - A Python toolbox for Adaptive Neuro-Fuzzy Inference Systems."""

__version__ = "0.1.0rc0"

# Expose high-level estimators
from .classifier import ANFISClassifier
from .regressor import ANFISRegressor

__all__ = ["ANFISClassifier", "ANFISRegressor"]
