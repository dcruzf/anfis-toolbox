"""Optimization algorithms for ANFIS.

This module contains pluggable training algorithms (optimizers/trainers)
that can be used with the ANFIS model.

Design goals:
- Decouple training algorithms from the model class
- Keep a simple API similar to scikit-learn fit(X, y)
- Allow power users to instantiate and pass custom trainers

Example:
    from anfis_toolbox.optim import SGDTrainer, RMSPropTrainer, HybridTrainer
    trainer = SGDTrainer(learning_rate=0.01, epochs=200)
    losses = trainer.fit(model, X, y)

Task compatibility and guidance:
--------------------------------
- HybridTrainer implements the original Jang (1993) hybrid learning and is intended
    for regression with the regression ANFIS (single-output). It is not compatible
    with the classification head.

- SGDTrainer, RMSPropTrainer and AdamTrainer perform generic backprop updates and now
    accept pluggable loss functions (see ``anfis_toolbox.losses``). They default to mean
    squared error for regression, but can minimize other differentiable objectives such as
    categorical cross-entropy when used with ``ANFISClassifier``. Targets are adapted via the
    selected loss' ``prepare_targets`` helper, so integer labels or one-hot matrices are both
    supported seamlessly.
"""

from .adam import AdamTrainer
from .base import BaseTrainer
from .hybrid import HybridTrainer
from .hybrid_adam import HybridAdamTrainer
from .pso import PSOTrainer
from .rmsprop import RMSPropTrainer
from .sgd import SGDTrainer

__all__ = [
    "BaseTrainer",
    "SGDTrainer",
    "HybridTrainer",
    "AdamTrainer",
    "HybridAdamTrainer",
    "RMSPropTrainer",
    "PSOTrainer",
]
