# API Reference

Complete reference documentation for all ANFIS Toolbox classes, functions, and modules.

## 🏗️ Core Architecture

### Main Classes

| Class | Description | Documentation |
|-------|-------------|---------------|
| **`ANFIS`** | Core ANFIS implementation for regression | [🔗 Details](../models/anfis.md#anfis-class) |
| **`ANFISClassifier`** | ANFIS for multi-class classification | [🔗 Details](../models/anfis-classifier.md#anfisclassifier-class) |
| **`ANFISBuilder`** | Fluent API for custom model construction | [🔗 Details](builders.md#anfisbuilder) |
| **`QuickANFIS`** | Simplified API for common use cases | [🔗 Details](builders.md#quickanfis) |
| **`FuzzyCMeans`** | Fuzzy C-Means clustering algorithm | [🔗 Details](../models/fuzzy_c-means.md#fuzzycmeans-class) |

### Membership Functions

| Class | Type | Parameters | Documentation |
|-------|------|------------|---------------|
| **`GaussianMF`** | Gaussian | `mean`, `sigma` | [🔗 Details](membership-functions.md#gaussianmf) |
| **`Gaussian2MF`** | Two-sided Gaussian | `sigma1`, `c1`, `sigma2`, `c2` | [🔗 Details](membership-functions.md#gaussian2mf) |
| **`TriangularMF`** | Triangular | `a`, `b`, `c` | [🔗 Details](membership-functions.md#triangularmf) |
| **`TrapezoidalMF`** | Trapezoidal | `a`, `b`, `c`, `d` | [🔗 Details](membership-functions.md#trapezoidalmf) |
| **`BellMF`** | Bell-shaped | `a`, `b`, `c` | [🔗 Details](membership-functions.md#bellmf) |
| **`SigmoidalMF`** | Sigmoidal | `a`, `c` | [🔗 Details](membership-functions.md#sigmoidalmf) |
| **`DiffSigmoidalMF`** | Difference of sigmoids | `a1`, `c1`, `a2`, `c2` | [🔗 Details](membership-functions.md#diffsigmoidalmf) |
| **`ProdSigmoidalMF`** | Product of sigmoids | `a1`, `c1`, `a2`, `c2` | [🔗 Details](membership-functions.md#prodsigmoidalmf) |
| **`SShapedMF`** | S-shaped | `a`, `b` | [🔗 Details](membership-functions.md#sshapedmf) |
| **`LinSShapedMF`** | Linear S-shaped | `a`, `b` | [🔗 Details](membership-functions.md#linsshapedmf) |
| **`ZShapedMF`** | Z-shaped | `a`, `b` | [🔗 Details](membership-functions.md#zshapedmf) |
| **`LinZShapedMF`** | Linear Z-shaped | `a`, `b` | [🔗 Details](membership-functions.md#linzshapedmf) |
| **`PiMF`** | Pi-shaped | `a`, `b`, `c`, `d` | [🔗 Details](membership-functions.md#pimf) |

## 📊 Analysis & Visualization

| Class/Function | Purpose | Documentation |
|----------------|---------|---------------|
| **`ANFISVisualizer`** | Plotting and visualization | Coming soon |
| **`ANFISValidator`** | Model validation and metrics | Coming soon |
| **`ANFISMetrics`** | Performance metrics | Coming soon |
| **`quick_evaluate`** | Fast model evaluation | Coming soon |

## ⚙️ Configuration & Utilities

| Class/Function | Purpose | Documentation |
|----------------|---------|---------------|
| **`ANFISConfig`** | Configuration management | Coming soon |
| **`load_anfis`** | Model loading | Coming soon |
| **`save_anfis`** | Model saving | Coming soon |

## 📦 Module Organization

```
anfis_toolbox/
├── __init__.py         # Package initialization
├── model.py            # ANFIS and ANFISClassifier classes
├── membership.py       # Membership function classes
├── builders.py         # ANFISBuilder and QuickANFIS classes
├── layers.py           # Neural network layer implementations
├── optim/              # Training algorithms and optimizers
│   ├── __init__.py
│   ├── adam.py
│   ├── pso.py
│   └── rmsprop.py
├── clustering.py       # Fuzzy C-Means clustering
├── metrics.py          # Performance metrics and evaluation
├── losses.py           # Loss function implementations
├── validation.py       # Model validation utilities
├── visualization.py    # Plotting and visualization tools
├── model_selection.py  # Model selection and comparison
├── config.py           # Configuration management
└── logging_config.py   # Logging configuration
```

## 🚀 Quick Reference

### Essential Imports

```python
# Core functionality
from anfis_toolbox import ANFIS, ANFISClassifier
from anfis_toolbox.builders import ANFISBuilder, QuickANFIS

# Membership functions
from anfis_toolbox.membership import (
    GaussianMF, Gaussian2MF, TriangularMF, TrapezoidalMF,
    BellMF, SigmoidalMF, DiffSigmoidalMF, ProdSigmoidalMF,
    SShapedMF, LinSShapedMF, ZShapedMF, LinZShapedMF, PiMF
)

# Clustering
from anfis_toolbox.clustering import FuzzyCMeans

# Metrics and validation
from anfis_toolbox import metrics, validation

# Visualization (optional dependencies)
from anfis_toolbox import visualization
```

### Common Patterns

#### 🎯 Quick Start (Regression)
```python
from anfis_toolbox.builders import QuickANFIS

model = QuickANFIS.for_regression(X, n_mfs=3)
losses = model.fit(X, y, epochs=100)
predictions = model.predict(X)
```

#### 🏗️ Custom Building
```python
from anfis_toolbox.builders import ANFISBuilder

builder = ANFISBuilder()
builder.add_input('x1', 0, 10, 3, 'gaussian')
builder.add_input('x2', -5, 5, 4, 'bell')
model = builder.build()
```

#### � Clustering for Initialization
```python
from anfis_toolbox.clustering import FuzzyCMeans

fcm = FuzzyCMeans(n_clusters=3)
fcm.fit(X)
centers = fcm.cluster_centers_
```

## 📚 Detailed Documentation

### By Category

- **[ANFIS Models](../models/anfis.md)** - Regression and classification models
- **[Builders](builders.md)** - ANFISBuilder and QuickANFIS
- **[Membership Functions](membership-functions.md)** - All MF types
- **[Models](models.md)** - ANFIS and ANFISClassifier classes
- **[Layers](layers.md)** - Neural network layers
- **[Fuzzy C-Means](../models/fuzzy_c-means.md)** - Clustering algorithm
- **[Clustering](clustering.md)** - FuzzyCMeans class
- **[Losses](losses.md)** - Training loss functions
- **[Metrics](metrics.md)** - Performance evaluation
- **[Visualization](visualization.md)** - Plotting utilities
- **[Validation](validation.md)** - Model validation tools
- **[Optimization](optim.md)** - Training algorithms

### By Use Case

- **[Getting Started](../getting-started/quickstart.md)** - Basic usage patterns
- **[Function Approximation](../examples/function-approximation.md)** - Learning mathematical functions
- **[Regression Analysis](../examples/regression.md)** - Continuous value prediction
- **[Classification](../examples/classification.md)** - Multi-class problems
- **[Time Series](../examples/time-series.md)** - Forecasting applications

## 🔍 Search and Navigation

### Find by Functionality

| I want to... | Look at... |
|-------------|------------|
| Create a simple model | `QuickANFIS` in [Builders](builders.md) |
| Build custom architecture | `ANFISBuilder` in [Builders](builders.md) |
| Choose membership functions | [Membership Functions](membership-functions.md) |
| Choose loss functions | [Losses](losses.md) |
| Train my model | `fit()` method in [Models](models.md) |
| Evaluate performance | [Metrics](metrics.md) |
| Visualize results | [Visualization](visualization.md) |
| Validate models | [Validation](validation.md) |
| Cluster data | `FuzzyCMeans` in [Clustering](clustering.md) |
| Configure training | [Optimization](optim.md) |

### Find by Data Type

| Data Type | Relevant Classes/Functions |
|-----------|----------------------------|
| **Numpy arrays** | All core functionality |
| **Regression targets** | `ANFIS` with `fit()` |
| **Classification labels** | `ANFISClassifier` with `fit()` |
| **Unlabeled data** | `FuzzyCMeans` for clustering |
| **Time series** | Custom input configuration |
| **Control signals** | Domain-specific MF setup |

## Navigation

**Start here for specific needs:**

- 🚀 **New user?** → [ANFIS Models](../models/anfis.md)
- 🏗️ **Building models?** → [Builders](builders.md)
- 📊 **Analyzing results?** → [Metrics](metrics.md)
- 🎨 **Visualizing?** → [Visualization](visualization.md)
- 🔍 **Clustering?** → [Clustering](clustering.md)
- ⚙️ **Training?** → [Optimization](optim.md)

**Or browse by alphabetical order:**

[A](#) | [B](#) | [C](#) | [D](#) | [E](#) | [F](#) | [G](#) | [H](#) | [I](#) | [J](#) | [K](#) | [L](#) | [M](#) | [N](#) | [O](#) | [P](#) | [Q](#) | [R](#) | [S](#) | [T](#) | [U](#) | [V](#) | [W](#) | [X](#) | [Y](#) | [Z](#)
