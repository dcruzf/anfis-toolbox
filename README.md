# ANFIS Toolbox

Adaptive Neuro-Fuzzy Inference Systems (ANFIS) in pure Python (TSK architecture) with simple APIs, local model selection utilities, robust validation, and clear visualization.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Highlights

- 4-layer ANFIS (Membership → Rules → Normalization → Consequent)
- Training: SGD and Hybrid trainer (least-squares + gradient) decoupled from the model
- Membership functions: Gaussian, Bell, Sigmoidal, Triangular, Trapezoidal, S-shaped (S), Z-shaped (Z), Pi-shaped (Π)
- Builders: ANFISBuilder and QuickANFIS for fast setup from data
- Validation: ANFISValidator with deterministic CV, metrics, learning curves
- Model selection: Local `KFold` and `train_test_split` (no scikit-learn dependency)
- Visualization: EDA plots, training curves, residuals, membership and rule-activation plots
- Tested across Python 3.10–3.13 with 100% coverage; linted with Ruff; docs via MkDocs

## Installation

```bash
pip install anfis-toolbox
```

## Quick start

### High-level regressor facade

```python
import numpy as np
from anfis_toolbox import ANFISRegressor

X = np.random.uniform(-2, 2, (200, 2))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1]

reg = ANFISRegressor(
    optimizer="adam",
    epochs=40,
    learning_rate=0.01,
    inputs_config={"x1": {"mf_type": "triangular", "n_mfs": 4}},
)
reg.fit(X, y)
preds = reg.predict([[0.2, -1.5]])
metrics = reg.evaluate(X, y)
```

Minimal example using QuickANFIS:

```python
import numpy as np
from anfis_toolbox import QuickANFIS

X = np.random.uniform(-2, 2, (200, 2))
y = (X[:, 0] ** 2 + X[:, 1] ** 2)

model = QuickANFIS.for_regression(X, n_mfs=3)
losses = model.fit_hybrid(X, y, epochs=50)
preds = model.predict([[1.0, -0.5], [0.5, 1.2]])
```

Or with explicit membership functions:

```python
import numpy as np
from anfis_toolbox import ANFIS, GaussianMF

input_mfs = {
    "x1": [GaussianMF(-1.0, 1.0), GaussianMF(1.0, 1.0)],
    "x2": [GaussianMF(-1.0, 1.0), GaussianMF(1.0, 1.0)],
}

model = ANFIS(input_mfs)
X = np.random.randn(100, 2)
y = X.sum(axis=1)
losses = model.fit_hybrid(X, y, epochs=50)
pred = model.predict([[0.5, -0.5]])
```

## Visualization (optional)

```python
from anfis_toolbox import ANFISVisualizer, quick_plot_training, quick_plot_results

quick_plot_training(losses)
quick_plot_results(X, y, model)

viz = ANFISVisualizer(model)
viz.plot_membership_functions()
```

## Validation and model selection

```python
from anfis_toolbox import ANFISValidator
from anfis_toolbox.model_selection import train_test_split, KFold

Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)

validator = ANFISValidator(model)
cv = validator.cross_validate(Xtr, ytr, cv=KFold(n_splits=5, shuffle=True, random_state=42))
```

## License

MIT – see [LICENSE](LICENSE).

## References

1. Jang, J. S. (1993). ANFIS: adaptive-network-based fuzzy inference system. IEEE TSMC, 23(3), 665–685.
