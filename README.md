<div align="center">
  <a href="https://dcruzf.github.io/anfis-toolbox">
  <h1>ANFIS Toolbox</h1>
  <img src="docs/assets/logo.svg" alt="ANFIS Toolbox">
  </a>
</div>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![coverage](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fdcruzf.github.io%2Fanfis-toolbox%2Fassets%2Fcov%2Findex.html&search=%3Cspan%20class%3D%22pc_cov%22%3E(%3F%3Ccov%3E%5Cd%2B%25)%3C%2Fspan%3E&replace=%24%3Ccov%3E&style=flat&logo=pytest&logoColor=white&label=coverage&color=green)](https://dcruzf.github.io/anfis-toolbox/assets/cov/)



A batteries-included Adaptive Neuro-Fuzzy Inference System (ANFIS) toolkit built in pure Python. It exposes high-level regression and classification APIs, modern trainers, and a rich catalog of membership functions.

## 🚀 Overview

- Takagi–Sugeno–Kang (TSK) ANFIS with the classic four-layer architecture (Membership → Rules → Normalization → Consequent).
- Regressor and classifier facades with a familiar scikit-learn style (`fit`, `predict`, `score`).
- Trainers (Hybrid, SGD, Adam, RMSProp, PSO) decoupled from the model for easy experimentation.
- 10+ membership function families with convenient builders, aliases, and data-driven initialization (grid, FCM, random).
- Thorough test coverage (100%+) across Python 3.10–3.13.

## 📦 Installation

Install from PyPI:

```bash
pip install anfis-toolbox
```


```bash
make install
```

## 🧠 Quick start

### Regression

```python
import numpy as np
from anfis_toolbox import ANFISRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(200, 2))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1]

reg = ANFISRegressor(
    optimizer="adam",
    epochs=40,
    learning_rate=0.01,
    inputs_config={"x1": {"mf_type": "triangular", "n_mfs": 4}},
)

reg.fit(X, y)
prediction = reg.predict([[0.2, -1.5]])
metrics = reg.evaluate(X, y)
```

### Classification

```python
import numpy as np
from anfis_toolbox import ANFISClassifier

rng = np.random.default_rng(1)
X = rng.uniform(-1.0, 1.0, size=(150, 2))
y = (1.2 * X[:, 0] - 0.8 * X[:, 1] > 0).astype(int)

clf = ANFISClassifier(
    n_classes=2,
    optimizer="sgd",
    epochs=25,
    learning_rate=0.05,
    random_state=0,
)

clf.fit(X, y)
preds = clf.predict([[0.4, -0.2]])
proba = clf.predict_proba([[0.4, -0.2]])
report = clf.evaluate(X, y)
```

## 🧩 Membership functions at a glance

| Family | Aliases | Notes |
| --- | --- | --- |
| `gaussian`, `gaussian2` | – | Single or dual-sided Gaussians with automatic width control |
| `triangular`, `trapezoidal` | – | Piecewise-linear shapes with edge clamping |
| `bell`, `gbell` | `gbell` | Generalized bell with configurable slope |
| `sigmoidal`, `sigmoid` | `sigmoid` | Smooth step functions with width-derived slope |
| `sshape`, `linsshape`, `s` | `ls` | Linear S transitions (grid & FCM support) |
| `zshape`, `linzshape`, `z` | `lz` | Linear Z transitions |
| `prodsigmoidal`, `prodsigmoid` | `prodsigmoid` | Product of opposing sigmoids forming bumps |
| `diffsigmoidal`, `diffsigmoid` | `diffsigmoid` | Difference of sigmoids, ideal for band-pass behavior |
| `pi`, `pimf` | `pimf` | Pi-shaped with configurable plateau |

Builders support grid, fuzzy C-means (FCM), and random initialization strategies—combine them per input via `inputs_config`.

## 🛠️ Training options

```python
from anfis_toolbox import ANFISRegressor
from anfis_toolbox.optim import SGDTrainer

reg = ANFISRegressor(optimizer=SGDTrainer, epochs=200, learning_rate=0.02)
```

- **Hybrid**: Jang-style least-squares + gradient descent (default for regression).
- **Backprop trainers**: SGD, Adam, RMSProp, PSO – all expose `learning_rate`, `epochs`, `batch_size`, and optional custom losses.
- **Losses**: Choose by name (`"mse"`, `"cross_entropy"`, …) or provide a custom `LossFunction` implementation.

## 📚 Documentation

- Comprehensive guides, API reference, and examples: [docs/](docs/) (built with MkDocs).
- Example notebooks: `docs/examples/*.ipynb` showcase regression, classification, and visualization recipes.

## 🧪 Testing & quality

Run the full suite (pytest + coverage + lint):

```bash
make test
```

Additional targets:

- `make lint` — Run Ruff linting
- `make docs` — Build the MkDocs site locally
- `make help` — Show all available targets with their help messages

## 🤝 Contributing

Issues and pull requests are welcome! Please open a discussion if you’d like to propose larger changes. See the [docs/guide](docs/guide/) section for architecture notes and examples.

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## 📚 References

1. Jang, J. S. (1993). ANFIS: adaptive-network-based fuzzy inference system. IEEE transactions on systems, man, and cybernetics, 23(3), 665-685. https://doi.org/10.1109/21.256541
