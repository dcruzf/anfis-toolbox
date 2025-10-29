<div align="center">
  <a href="https://dcruzf.github.io/anfis-toolbox">
  <h1>ANFIS Toolbox</h1>
  <img src="https://dcruzf.github.io/anfis-toolbox/assets/logo.svg" alt="ANFIS Toolbox">
  </a>
</div>

[![CI](https://github.com/dcruzf/anfis-toolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/dcruzf/anfis-toolbox/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://dcruzf.github.io/anfis-toolbox/)
[![Coverage](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fdcruzf.github.io%2Fanfis-toolbox%2Fassets%2Fcov%2Findex.html&search=%3Cspan%20class%3D%22pc_cov%22%3E(%3F%3Ccov%3E%5Cd%2B%25)%3C%2Fspan%3E&replace=%24%3Ccov%3E&style=flat&logo=pytest&logoColor=white&label=coverage&color=brightgreen)](https://dcruzf.github.io/anfis-toolbox/assets/cov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-indigo.svg)](https://github.com/dcruzf/anfis-toolbox/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/anfis-toolbox)](https://pypi.org/project/anfis-toolbox/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anfis-toolbox)](https://pypi.org/project/anfis-toolbox/)
[![PyPI - Format](https://img.shields.io/pypi/format/anfis-toolbox)](https://www.piwheels.org/project/anfis-toolbox/)
[![security: bandit](https://img.shields.io/badge/security-bandit-black.svg)](https://github.com/PyCQA/bandit)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Hatch](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pypa/hatch/master/docs/assets/badge/v0.json)](https://github.com/pypa/hatch)
[![Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17437178.svg)](https://doi.org/10.5281/zenodo.17437178)

ANFIS Toolbox is a comprehensive Python library for creating, training, and deploying Adaptive Neuro-Fuzzy Inference Systems (ANFIS). It provides an intuitive API that makes fuzzy neural networks accessible to both beginners and experts.

## 🚀 Overview

- Takagi–Sugeno–Kang (TSK) ANFIS with the classic four-layer architecture (Membership → Rules → Normalization → Consequent).
- Regressor and classifier facades with a familiar scikit-learn style (`fit`, `predict`, `score`).
- Trainers (Hybrid, SGD, Adam, RMSProp, PSO) decoupled from the model for easy experimentation.
- 10+ membership function families. The primary public interfaces are `ANFISRegressor` and `ANFISClassifier`.
- Thorough test coverage (100%+).

## 📦 Installation

Install from PyPI:

```bash
pip install anfis-toolbox
```

## 🧠 Quick start

### Regression

```python
import numpy as np
from anfis_toolbox import ANFISRegressor

X = np.random.uniform(-2, 2, (100, 2))
y = X[:, 0]**2 + X[:, 1]**2

model = ANFISRegressor()
model.fit(X, y)
metrics = model.evaluate(X, y)
```

### Classification

```python
import numpy as np
from anfis_toolbox import ANFISClassifier

X = np.r_[np.random.normal(-1, .3, (50, 2)), np.random.normal(1, .3, (50, 2))]
y = np.r_[np.zeros(50, int), np.ones(50, int)]

model = ANFISClassifier()
model.fit(X, y)
metrics = model.evaluate(X, y)
```

## 🧩 Membership functions at a glance

- **Gaussian** (`GaussianMF`) - Smooth bell curves
- **Gaussian2** (`Gaussian2MF`) - Two-sided Gaussian with flat region
- **Triangular** (`TriangularMF`) - Simple triangular shapes
- **Trapezoidal** (`TrapezoidalMF`) - Plateau regions
- **Bell-shaped** (`BellMF`) - Generalized bell curves
- **Sigmoidal** (`SigmoidalMF`) - S-shaped transitions
- **Diff-Sigmoidal** (`DiffSigmoidalMF`) - Difference of two sigmoids
- **Prod-Sigmoidal** (`ProdSigmoidalMF`) - Product of two sigmoids
- **S-shaped** (`SShapedMF`) - Smooth S-curve transitions
- **Linear S-shaped** (`LinSShapedMF`) - Piecewise linear S-curve
- **Z-shaped** (`ZShapedMF`) - Smooth Z-curve transitions
- **Linear Z-shaped** (`LinZShapedMF`) - Piecewise linear Z-curve
- **Pi-shaped** (`PiMF`) - Bell with flat top



## 🛠️ Training options

* **SGD (Stochastic Gradient Descent)** – Classic gradient-based optimization with incremental updates
* **Adam** – Adaptive learning rates with momentum for faster convergence
* **RMSProp** – Scales learning rates by recent gradient magnitudes for stable training
* **PSO (Particle Swarm Optimization)** – Population-based global search strategy
* **Hybrid SGD + OLS** – Combines gradient descent with least-squares parameter refinement
* **Hybrid Adam + OLS** – Integrates adaptive optimization with analytical least-squares adjustment

## 📚 Documentation

- Comprehensive guides, API reference, and examples: [docs/](https://dcruzf.github.io/anfis-toolbox/) (built with MkDocs).

## 🧪 Testing & quality

Run the full suite (pytest + coverage):

```bash
make test
```

Additional targets:

- `make lint` — Run Ruff linting
- `make docs` — Build the MkDocs site locally
- `make help` — Show all available targets with their help messages

This project is tested on Python 3.10 | 3.11 | 3.12 | 3.13 | 3.14 across Linux, Windows and macOS.

## 🤝 Contributing

Issues and pull requests are welcome! Please open a discussion if you’d like to propose larger changes. See the [docs/guide](docs/guide.md) section for architecture notes and examples.

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## 📚 References

1. Jang, J. S. (1993). ANFIS: adaptive-network-based fuzzy inference system. IEEE transactions on systems, man, and cybernetics, 23(3), 665-685. https://doi.org/10.1109/21.256541
