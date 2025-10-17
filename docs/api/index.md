# API Reference

Complete reference documentation for all ANFIS Toolbox classes, functions, and modules.

## 🏗️ Core Architecture

### Estimators

| Estimator | Purpose | Module |
|-----------|---------|--------|
| **`ANFISRegressor`** | Scikit-learn style regression interface | [regressor](regressor.md) |
| **`ANFISClassifier`** | Scikit-learn style classification interface | [classifier](classifier.md) |


### Neural Network Layers

| Class | Purpose | Module |
|-------|---------|--------|
| **`MembershipLayer`** | Fuzzy membership function layer | [layers](layers.md) |
| **`RuleLayer`** | Fuzzy rule firing strength computation | [layers](layers.md) |
| **`NormalizationLayer`** | Rule activation normalization | [layers](layers.md) |
| **`ConsequentLayer`** | Takagi-Sugeno consequent computation | [layers](layers.md) |
| **`ClassificationConsequentLayer`** | Classification-specific consequent layer | [layers](layers.md) |

## 🎯 Membership Functions

Complete set of 13 fuzzy membership function implementations:

| Function | Type | Parameters | Module |
|----------|------|------------|--------|
| **`GaussianMF`** | Gaussian | `mean`, `sigma` | [membership-functions](membership-functions.md) |
| **`Gaussian2MF`** | Two-sided Gaussian | `sigma1`, `c1`, `sigma2`, `c2` | [membership-functions](membership-functions.md) |
| **`TriangularMF`** | Triangular | `a`, `b`, `c` | [membership-functions](membership-functions.md) |
| **`TrapezoidalMF`** | Trapezoidal | `a`, `b`, `c`, `d` | [membership-functions](membership-functions.md) |
| **`BellMF`** | Bell-shaped | `a`, `b`, `c` | [membership-functions](membership-functions.md) |
| **`SigmoidalMF`** | Sigmoidal | `a`, `c` | [membership-functions](membership-functions.md) |
| **`DiffSigmoidalMF`** | Difference of sigmoids | `a1`, `c1`, `a2`, `c2` | [membership-functions](membership-functions.md) |
| **`ProdSigmoidalMF`** | Product of sigmoids | `a1`, `c1`, `a2`, `c2` | [membership-functions](membership-functions.md) |
| **`SShapedMF`** | S-shaped | `a`, `b` | [membership-functions](membership-functions.md) |
| **`LinSShapedMF`** | Linear S-shaped | `a`, `b` | [membership-functions](membership-functions.md) |
| **`ZShapedMF`** | Z-shaped | `a`, `b` | [membership-functions](membership-functions.md) |
| **`LinZShapedMF`** | Linear Z-shaped | `a`, `b` | [membership-functions](membership-functions.md) |
| **`PiMF`** | Pi-shaped | `a`, `b`, `c`, `d` | [membership-functions](membership-functions.md) |

## 🔧 Training & Optimization

### Training Algorithms

| Trainer | Method | Module |
|---------|--------|--------|
| **`HybridTrainer`** | Least squares + backpropagation (recommended) | [optim](optim.md) |
| **`HybridAdamTrainer`** | Least squares + Adam with adaptive moments | [optim](optim.md) |
| **`SGDTrainer`** | Stochastic gradient descent | [optim](optim.md) |
| **`AdamTrainer`** | Adaptive moment estimation | [optim](optim.md) |
| **`RMSPropTrainer`** | Root mean square propagation | [optim](optim.md) |
| **`PSOTrainer`** | Particle swarm optimization | [optim](optim.md) |

### Loss Functions

| Function | Purpose | Module |
|----------|---------|--------|
| **`mse_loss`** | Mean squared error for regression | [losses](losses.md) |
| **`mse_grad`** | MSE gradient computation | [losses](losses.md) |
| **`cross_entropy_loss`** | Cross-entropy for classification | [losses](losses.md) |
| **`cross_entropy_grad`** | Cross-entropy gradient computation | [losses](losses.md) |

## 📊 Evaluation & Validation

### Metrics

Comprehensive metrics for model evaluation:

| Category | Functions | Module |
|----------|-----------|--------|
| **Regression** | MSE, RMSE, MAE, MAPE, SMAPE, R², explained variance, median AE, bias, Pearson, MSLE | [metrics](metrics.md) |
| **Classification** | Accuracy, balanced accuracy, precision/recall/F1, log loss, cross-entropy | [metrics](metrics.md) |
| **Clustering** | Partition coefficient, classification entropy, Xie-Beni index | [metrics](metrics.md) |


## 🔍 Clustering

| Class | Purpose | Module |
|-------|---------|--------|
| **`FuzzyCMeans`** | Fuzzy C-Means clustering algorithm | [clustering](clustering.md) |

## 📚 Detailed Documentation

### By Category

- **[ANFIS Models](../models/anfis.md)** - High-level model documentation
- **[Regressor](regressor.md)** - High-level regression estimator API
- **[Classifier](classifier.md)** - High-level classification estimator API
- **[Builders](builders.md)** - Model construction utilities
- **[Membership Functions](membership-functions.md)** - All 13 MF implementations
- **[Models](models.md)** - Core ANFIS and ANFISClassifier classes
- **[Layers](layers.md)** - Neural network layer implementations
- **[Clustering](clustering.md)** - FuzzyCMeans clustering
- **[Losses](losses.md)** - Training loss functions and gradients
- **[Metrics](metrics.md)** - Performance evaluation metrics
- **[Configuration](config.md)** - Persisting setups and presets
- **[Logging](logging.md)** - Training log helpers
- **[Optimization](optim.md)** - Training algorithm implementations

## 🔍 Search and Navigation

### Find by Functionality

| I want to... | Look at... |
|-------------|------------|
| Use a regression estimator | [ANFISRegressor](regressor.md) |
| Use a classification estimator | [ANFISClassifier](classifier.md) |
| Choose membership functions | [Membership Functions](membership-functions.md) |
| Train my model | `fit()` method in [Models](models.md) |
| Evaluate performance | [Metrics](metrics.md) |
| Configure training | [Optimization](optim.md) |
| Save configs or presets | [Configuration](config.md) |
| Enable training logs | [Logging](logging.md) |

## Navigation

**Start here for specific needs:**

- 🚀 **New user?** → [ANFIS Models](../models/anfis.md)
- 🏗️ **Building models?** → [Builders](builders.md)
- 📊 **Analyzing results?** → [Metrics](metrics.md)
- 🔍 **Clustering?** → [Clustering](clustering.md)
- ⚙️ **Training?** → [Optimization](optim.md)
- 🧪 **Using estimators?** → [Regressor](regressor.md) / [Classifier](classifier.md)
- 📝 **Saving configs or logs?** → [Configuration](config.md) & [Logging](logging.md)
