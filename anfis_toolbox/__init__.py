"""ANFIS Toolbox - A Python toolbox for Adaptive Neuro-Fuzzy Inference Systems."""

__version__ = "0.2.0"

# Expose high-level estimators
from .classifier import ANFISClassifier
from .regressor import ANFISRegressor

__all__ = ["ANFISClassifier", "ANFISRegressor"]
