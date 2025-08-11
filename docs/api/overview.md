# API Reference

Complete reference documentation for all ANFIS Toolbox classes, functions, and modules.

## 🏗️ Core Architecture

### Main Classes

| Class | Description | Documentation |
|-------|-------------|---------------|
| **`ANFIS`** | Core ANFIS implementation | [🔗 Details](core.md#anfis) |
| **`QuickANFIS`** | Simplified API for common use cases | [🔗 Details](builders.md#quickanfis) |
| **`ANFISBuilder`** | Fluent API for custom model construction | [🔗 Details](builders.md#anfisbuilder) |

### Membership Functions

| Class | Type | Parameters | Documentation |
|-------|------|------------|---------------|
| **`GaussianMF`** | Gaussian | `center`, `width` | [🔗 Details](membership-functions.md#gaussianmf) |
| **`TriangularMF`** | Triangular | `left`, `center`, `right` | [🔗 Details](membership-functions.md#triangularmf) |
| **`TrapezoidalMF`** | Trapezoidal | `left`, `left_peak`, `right_peak`, `right` | [🔗 Details](membership-functions.md#trapezoidalmf) |
| **`BellMF`** | Bell-shaped | `center`, `width`, `slope` | [🔗 Details](membership-functions.md#bellmf) |
| **`SigmoidalMF`** | Sigmoidal | `center`, `slope` | [🔗 Details](membership-functions.md#sigmoidalmf) |
| **`PiMF`** | Pi-shaped | `left`, `left_peak`, `right_peak`, `right` | [🔗 Details](membership-functions.md#pimf) |

## 📊 Analysis & Visualization

| Class/Function | Purpose | Documentation |
|----------------|---------|---------------|
| **`ANFISVisualizer`** | Plotting and visualization | [🔗 Details](visualization.md) |
| **`ANFISValidator`** | Model validation and metrics | [🔗 Details](validation.md#anfisvalidator) |
| **`ANFISMetrics`** | Performance metrics | [🔗 Details](validation.md#anfismetrics) |
| **`quick_evaluate`** | Fast model evaluation | [🔗 Details](validation.md#quick-evaluate) |

## ⚙️ Configuration & Utilities

| Class/Function | Purpose | Documentation |
|----------------|---------|---------------|
| **`ANFISConfig`** | Configuration management | [🔗 Details](configuration.md) |
| **`load_anfis`** | Model loading | [🔗 Details](persistence.md#load-anfis) |
| **`save_anfis`** | Model saving | [🔗 Details](persistence.md#save-anfis) |

## 📦 Module Organization

```
anfis_toolbox/
├── core.py              # ANFIS main class
├── membership.py        # Membership function classes
├── builders.py          # QuickANFIS and ANFISBuilder
├── training.py          # Training algorithms
├── validation.py        # Validation and metrics
├── visualization.py     # Plotting utilities
├── config.py           # Configuration management
├── utils.py            # Helper functions
└── __init__.py         # Package initialization
```

## 🚀 Quick Reference

### Essential Imports

```python
# Core functionality
from anfis_toolbox import ANFIS, QuickANFIS, ANFISBuilder

# Membership functions
from anfis_toolbox.membership import (
    GaussianMF, TriangularMF, TrapezoidalMF,
    BellMF, SigmoidalMF, PiMF
)

# Validation and visualization (optional dependencies)
from anfis_toolbox import (
    ANFISValidator, ANFISVisualizer, ANFISMetrics,
    quick_evaluate, ANFISConfig
)

# Persistence
from anfis_toolbox import load_anfis
```

### Common Patterns

#### 🎯 Quick Start
```python
model = QuickANFIS.for_regression(X, n_mfs=3)
losses = model.fit_hybrid(X, y, epochs=100)
metrics = quick_evaluate(model, X, y)
```

#### 🏗️ Custom Building
```python
builder = (ANFISBuilder()
    .add_input('x1', 'gaussian', 3)
    .add_input('x2', 'bell', 4)
    .set_config(ANFISConfig(learning_rate=0.01))
)
model = builder.build()
```

#### 📊 Comprehensive Analysis
```python
validator = ANFISValidator(model)
cv_results = validator.cross_validate(X, y, cv=5)

visualizer = ANFISVisualizer(model)
visualizer.plot_membership_functions()
visualizer.plot_training_curves(losses)
```

## 📚 Detailed Documentation

### By Category

- **[Core Classes](core.md)** - ANFIS, basic functionality
- **[Builders](builders.md)** - QuickANFIS, ANFISBuilder
- **[Membership Functions](membership-functions.md)** - All 6 MF types
- **[Training](training.md)** - Learning algorithms and parameters
- **[Validation](validation.md)** - Metrics, cross-validation, evaluation
- **[Visualization](visualization.md)** - Plotting and analysis tools
- **[Configuration](configuration.md)** - ANFISConfig and settings
- **[Persistence](persistence.md)** - Saving and loading models
- **[Utilities](utilities.md)** - Helper functions and tools

### By Use Case

- **[Getting Started](../getting-started/quickstart.md)** - Basic usage patterns
- **[Function Approximation](function-approximation.md)** - Modeling mathematical functions
- **[Regression Analysis](regression-analysis.md)** - Predictive modeling
- **[Control Systems](control-systems.md)** - Fuzzy control applications
- **[Time Series](time-series.md)** - Temporal data modeling

## 🔍 Search and Navigation

### Find by Functionality

| I want to... | Look at... |
|-------------|------------|
| Create a simple model | `QuickANFIS` in [Builders](builders.md) |
| Build custom architecture | `ANFISBuilder` in [Builders](builders.md) |
| Choose membership functions | [Membership Functions](membership-functions.md) |
| Train my model | `fit_hybrid()`, `fit()` in [Core](core.md) |
| Evaluate performance | `quick_evaluate()` in [Validation](validation.md) |
| Visualize results | `ANFISVisualizer` in [Visualization](visualization.md) |
| Save/load models | [Persistence](persistence.md) |
| Configure training | `ANFISConfig` in [Configuration](configuration.md) |

### Find by Data Type

| Data Type | Relevant Classes/Functions |
|-----------|----------------------------|
| **Numpy arrays** | All core functionality |
| **Pandas DataFrames** | `QuickANFIS.from_dataframe()` |
| **Time series** | `QuickANFIS.for_time_series()` |
| **Images** | Custom MF setup, reshape utilities |
| **Control signals** | `ANFISBuilder` with domain-specific MFs |

## 📊 Parameter Reference

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | `int` | `100` | Number of training epochs |
| `learning_rate` | `float` | `0.01` | Learning rate for gradient descent |
| `tolerance` | `float` | `1e-6` | Convergence tolerance |
| `patience` | `int` | `10` | Early stopping patience |
| `validation_split` | `float` | `0.0` | Fraction of data for validation |

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_mfs` | `int` | `3` | Number of membership functions per input |
| `mf_type` | `str` | `'gaussian'` | Type of membership function |
| `random_state` | `int` | `None` | Random seed for reproducibility |

### Complete parameter lists available in each class documentation.

## 🧪 Testing and Validation

### Unit Tests
```bash
# Run all tests
pytest tests/

# Test specific module
pytest tests/test_membership.py

# Test with coverage
pytest --cov=anfis_toolbox tests/
```

### Type Checking
```bash
# Check types
mypy anfis_toolbox/

# Check specific file
mypy anfis_toolbox/core.py
```

## 🎯 Examples and Tutorials

Each API component includes:

- **💡 Usage examples** - Basic usage patterns
- **🔧 Advanced examples** - Complex configurations
- **⚠️ Common pitfalls** - What to avoid
- **📖 Related functions** - Cross-references
- **🧪 Test cases** - Validation examples

## 🤝 Contributing to Documentation

Documentation is generated using **mkdocstrings** from docstrings in the source code. To improve documentation:

1. **Edit docstrings** in source files
2. **Follow Google style** docstring format
3. **Include examples** in docstrings
4. **Add type hints** to all functions
5. **Run tests** to ensure accuracy

### Docstring Format
```python
def example_function(param1: int, param2: str = "default") -> float:
    """Brief description of the function.

    Longer description with more details about what the function does,
    when to use it, and any important considerations.

    Args:
        param1: Description of param1.
        param2: Description of param2 with default value.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.
        TypeError: When param2 is not a string.

    Examples:
        Basic usage:
        ```python
        result = example_function(5, "test")
        print(result)  # Output: 42.0
        ```

        Advanced usage:
        ```python
        result = example_function(10)  # Uses default param2
        ```
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return float(param1 * len(param2))
```

---

## Navigation

**Start here for specific needs:**

- 🚀 **New user?** → [Core Classes](core.md)
- 🏗️ **Building models?** → [Builders](builders.md)
- 📊 **Analyzing results?** → [Validation](validation.md)
- 🎨 **Visualizing?** → [Visualization](visualization.md)
- ⚙️ **Configuring?** → [Configuration](configuration.md)

**Or browse by alphabetical order:**

[A](#) | [B](#) | [C](#) | [D](#) | [E](#) | [F](#) | [G](#) | [H](#) | [I](#) | [J](#) | [K](#) | [L](#) | [M](#) | [N](#) | [O](#) | [P](#) | [Q](#) | [R](#) | [S](#) | [T](#) | [U](#) | [V](#) | [W](#) | [X](#) | [Y](#) | [Z](#)
