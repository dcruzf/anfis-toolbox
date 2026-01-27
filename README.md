<div align="center"><h1>ANFIS Toolbox</h1><img src="https://dcruzf.github.io/anfis-toolbox/assets/logo.svg" alt="ANFIS Toolbox"></div>

[![anfis-toolbox](https://img.shields.io/endpoint?url=https%3A%2F%2Fdcruzf.github.io%2Fanfis-toolbox%2Fassets%2Fbadge%2Fv0.json&style=flat-square)](https://dcruzf.github.io/anfis-toolbox/)
[![version](https://img.shields.io/pypi/v/anfis-toolbox?style=flat-square&label=&color=303fa1)](https://pypi.org/project/anfis-toolbox/)
[![doi](https://img.shields.io/badge/-10.5281%2Fzenodo.17437178-blue?style=flat-square&logo=DOI&color=303fa1)](https://doi.org/10.5281/zenodo.17437178)
[![License: MIT](https://img.shields.io/badge/_-MIT-303fa1.svg?style=flat-square)](https://github.com/dcruzf/anfis-toolbox/blob/main/LICENSE)
[![docs](https://img.shields.io/badge/_-docs-303fa1.svg?style=flat-square)](https://dcruzf.github.io/anfis-toolbox/)
[![Python versions](https://img.shields.io/pypi/pyversions/anfis-toolbox?style=flat-square&logo=python&logoColor=white&label=%20&labelColor=303fa1&color=303fa1)](https://pypi.org/project/anfis-toolbox/)
[![CI](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fgithub.com%2Fdcruzf%2Fanfis-toolbox%2Factions%2Fworkflows%2Fci.yml%2Fbadge.svg&search=%3Ctspan%20.*%3E(%3F%3Cstatus%3Epassing%7Cfailing)%3C%2Ftspan%3E&replace=ci%20%24%3Cstatus%3E&style=flat-square&logo=github&label=%20&color=303fa1)](https://github.com/dcruzf/anfis-toolbox/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/dynamic/regex?url=https%3A%2F%2Fdcruzf.github.io%2Fanfis-toolbox%2Fassets%2Fcov%2Findex.html&search=%3Cspan%20class%3D%22pc_cov%22%3E(%3F%3Ccov%3E%5Cd%2B%25)%3C%2Fspan%3E&replace=%24%3Ccov%3E&style=flat-square&logo=pytest&logoColor=white&label=cov&color=303fa1&labelColor=303fa1)](https://dcruzf.github.io/anfis-toolbox/assets/cov/)


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

### Local setup

Clone the repository:

```bash
git clone https://github.com/dcruzf/anfis-toolbox.git
cd anfis-toolbox
```

Create a virtual environment:

```bash
python -m venv .venv
```
Activate it:

**Linux / macOS**
```
source .venv/bin/activate
```
**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

Install the project in editable mode with development dependencies
(this includes Hatch and all test tools):

```bash
pip install -e .[dev]
```
### Running tests

Run the full test suite with coverage:
```bash
hatch test -c --all
```
This project is tested on Python 3.10 | 3.11 | 3.12 | 3.13 | 3.14 across Linux, Windows and macOS.

### Linting & Formatting

Run the linter and format the codebase:
```bash
hatch fmt
```

### Typing

Run static type checks:
```bash
hatch run typing
```

### Security

Run security checks with Bandit:
```bash
hatch run security
```


## 🤝 Contributing

Issues and pull requests are welcome! Please open a discussion if you’d like to propose larger changes. See the [docs/guide](https://dcruzf.github.io/anfis-toolbox/guide/) section for architecture notes and examples.

## 📄 License

Distributed under the MIT License. See [LICENSE](https://github.com/dcruzf/anfis-toolbox/blob/main/LICENSE) for details.

## 📚 References

1. Jang, J. S. (1993). ANFIS: adaptive-network-based fuzzy inference system. IEEE transactions on systems, man, and cybernetics, 23(3), 665-685. https://doi.org/10.1109/21.256541
