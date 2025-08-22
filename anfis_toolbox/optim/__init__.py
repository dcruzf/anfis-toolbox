"""Optimization algorithms for ANFIS.

This module contains pluggable training algorithms (optimizers/trainers)
that can be used with the ANFIS model.

Design goals:
- Decouple training algorithms from the model class
- Keep a simple API similar to scikit-learn fit(X, y)
- Allow power users to instantiate and pass custom trainers

Example:
    from anfis_toolbox.optim import SGDTrainer, HybridTrainer
    trainer = SGDTrainer(learning_rate=0.01, epochs=200)
    losses = trainer.fit(model, X, y)
"""

from .hybrid import HybridTrainer
from .sgd import SGDTrainer

__all__ = [
    "SGDTrainer",
    "HybridTrainer",
]
