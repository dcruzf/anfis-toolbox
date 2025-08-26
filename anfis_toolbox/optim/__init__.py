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

- SGDTrainer, RMSPropTrainer and AdamTrainer perform generic backprop updates minimizing mean
    squared error (MSE) between the model output returned by ``model.forward`` and
    the provided target ``y``. For regression models, this is the intended usage.

    For classification models (ANFISClassifier), these trainers will still run, but
    they will optimize MSE on the classifier logits/probabilities rather than a
    proper classification loss. If ``y`` is 1D integer labels, it will be reshaped
    to ``(n, 1)`` and broadcast against logits ``(n, K)`` during the MSE; if ``y``
    is one‑hot ``(n, K)``, the MSE will be computed element‑wise. This can be used
    for quick experiments, but for principled classification training prefer
    ``ANFISClassifier.fit(...)``, which uses cross‑entropy with softmax.
"""

from .adam import AdamTrainer
from .base import BaseTrainer
from .hybrid import HybridTrainer
from .pso import PSOTrainer
from .rmsprop import RMSPropTrainer
from .sgd import SGDTrainer

__all__ = [
    "BaseTrainer",
    "SGDTrainer",
    "HybridTrainer",
    "AdamTrainer",
    "RMSPropTrainer",
    "PSOTrainer",
]
