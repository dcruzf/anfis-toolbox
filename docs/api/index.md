# API Reference

Complete reference documentation for all ANFIS Toolbox classes, functions, and modules.

## 🏗️ Core Architecture

### Main Model Classes

| Class | Purpose | Module |
|-------|---------|--------|
| **`ANFIS`** | Core ANFIS implementation for regression tasks | [models](models) |
| **`ANFISClassifier`** | ANFIS for multi-class classification | [models](models) |

### Builder Classes

| Class | Purpose | Module |
|-------|---------|--------|
| **`QuickANFIS`** | Simplified API for common use cases | [builders](builders) |
| **`ANFISBuilder`** | Fluent API for custom model construction | [builders](builders) |

### Neural Network Layers

| Class | Purpose | Module |
|-------|---------|--------|
| **`MembershipLayer`** | Fuzzy membership function layer | [layers](layers) |
| **`RuleLayer`** | Fuzzy rule firing strength computation | [layers](layers) |
| **`NormalizationLayer`** | Rule activation normalization | [layers](layers) |
| **`ConsequentLayer`** | Takagi-Sugeno consequent computation | [layers](layers) |
| **`ClassificationConsequentLayer`** | Classification-specific consequent layer | [layers](layers) |

## 🎯 Membership Functions

Complete set of 13 fuzzy membership function implementations:

| Function | Type | Parameters | Module |
|----------|------|------------|--------|
| **`GaussianMF`** | Gaussian | `mean`, `sigma` | [membership-functions](membership-functions) |
| **`Gaussian2MF`** | Two-sided Gaussian | `sigma1`, `c1`, `sigma2`, `c2` | [membership-functions](membership-functions) |
| **`TriangularMF`** | Triangular | `a`, `b`, `c` | [membership-functions](membership-functions) |
| **`TrapezoidalMF`** | Trapezoidal | `a`, `b`, `c`, `d` | [membership-functions](membership-functions) |
| **`BellMF`** | Bell-shaped | `a`, `b`, `c` | [membership-functions](membership-functions) |
| **`SigmoidalMF`** | Sigmoidal | `a`, `c` | [membership-functions](membership-functions) |
| **`DiffSigmoidalMF`** | Difference of sigmoids | `a1`, `c1`, `a2`, `c2` | [membership-functions](membership-functions) |
| **`ProdSigmoidalMF`** | Product of sigmoids | `a1`, `c1`, `a2`, `c2` | [membership-functions](membership-functions) |
| **`SShapedMF`** | S-shaped | `a`, `b` | [membership-functions](membership-functions) |
| **`LinSShapedMF`** | Linear S-shaped | `a`, `b` | [membership-functions](membership-functions) |
| **`ZShapedMF`** | Z-shaped | `a`, `b` | [membership-functions](membership-functions) |
| **`LinZShapedMF`** | Linear Z-shaped | `a`, `b` | [membership-functions](membership-functions) |
| **`PiMF`** | Pi-shaped | `a`, `b`, `c`, `d` | [membership-functions](membership-functions) |

## 🔧 Training & Optimization

### Training Algorithms

| Trainer | Method | Module |
|---------|--------|--------|
| **`HybridTrainer`** | Least squares + backpropagation (recommended) | [optim](optim) |
| **`SGDTrainer`** | Stochastic gradient descent | [optim](optim) |
| **`AdamTrainer`** | Adaptive moment estimation | [optim](optim) |
| **`RMSPropTrainer`** | Root mean square propagation | [optim](optim) |
| **`PSOTrainer`** | Particle swarm optimization | [optim](optim) |

### Loss Functions

| Function | Purpose | Module |
|----------|---------|--------|
| **`mse_loss`** | Mean squared error for regression | [losses](losses) |
| **`mse_grad`** | MSE gradient computation | [losses](losses) |
| **`cross_entropy_loss`** | Cross-entropy for classification | [losses](losses) |
| **`cross_entropy_grad`** | Cross-entropy gradient computation | [losses](losses) |

## 📊 Evaluation & Validation

### Metrics

Comprehensive metrics for model evaluation:

| Category | Functions | Module |
|----------|-----------|--------|
| **Regression** | MSE, RMSE, MAE, MAPE, SMAPE, R², explained variance, median AE, bias, Pearson, MSLE | [metrics](metrics) |
| **Classification** | Accuracy, balanced accuracy, precision/recall/F1, log loss, cross-entropy | [metrics](metrics) |
| **Clustering** | Partition coefficient, classification entropy, Xie-Beni index | [metrics](metrics) |

> 💡 Use [`compute_metrics`](metrics/#metric-reports--automation) to auto-detect the task type and retrieve a full [`MetricReport`](metrics/#metric-reports--automation).

## 🔍 Clustering

| Class | Purpose | Module |
|-------|---------|--------|
| **`FuzzyCMeans`** | Fuzzy C-Means clustering algorithm | [clustering](clustering) |

## 📚 Detailed Documentation

### By Category

- **[ANFIS Models](../models/anfis.md)** - High-level model documentation
- **[Builders](builders)** - Model construction utilities
- **[Membership Functions](membership-functions)** - All 13 MF implementations
- **[Models](models)** - Core ANFIS and ANFISClassifier classes
- **[Layers](layers)** - Neural network layer implementations
- **[Clustering](clustering)** - FuzzyCMeans clustering
- **[Losses](losses)** - Training loss functions and gradients
- **[Metrics](metrics)** - Performance evaluation metrics
- **[Optimization](optim)** - Training algorithm implementations

## 🔍 Search and Navigation

### Find by Functionality

| I want to... | Look at... |
|-------------|------------|
| Create a simple model | `QuickANFIS` in [Builders](builders) |
| Build custom architecture | `ANFISBuilder` in [Builders](builders) |
| Choose membership functions | [Membership Functions](membership-functions) |
| Choose loss functions | [Losses](losses) |
| Train my model | `fit()` method in [Models](models) |
| Evaluate performance | [Metrics](metrics) |
| Cluster data | `FuzzyCMeans` in [Clustering](clustering) |
| Configure training | [Optimization](optim) |

## Navigation

**Start here for specific needs:**

- 🚀 **New user?** → [ANFIS Models](../models/anfis.md)
- 🏗️ **Building models?** → [Builders](builders)
- 📊 **Analyzing results?** → [Metrics](metrics)
- 🔍 **Clustering?** → [Clustering](clustering)
- ⚙️ **Training?** → [Optimization](optim)
