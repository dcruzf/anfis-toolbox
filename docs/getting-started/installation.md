# Installation

ANFIS Toolbox is available on PyPI and can be installed with `pip`. Choose the installation method that best fits your needs.

## System Requirements

- **Python**: 3.9+
- **Operating System**: Windows, macOS, Linux
- **Memory**: 4GB RAM minimum (8GB+ recommended for large models)
- **Dependencies**: NumPy (required), SciPy (required)

## Installation Options

### Basic Installation

The core package with minimal dependencies:

```bash
pip install anfis-toolbox
```

This gives you:

- ✅ Core ANFIS functionality
- ✅ All 6 membership functions
- ✅ Hybrid and backpropagation training
- ✅ Model persistence and configuration
- ✅ Basic utilities

**Dependencies**: `numpy`

### Full Installation (Recommended)

Install with all optional features:

```bash
pip install anfis-toolbox[all]
```

This includes everything from the basic installation plus:

- ✅ **Visualization**: Membership functions, training curves, predictions
- ✅ **Validation**: Cross-validation, metrics, model comparison
- ✅ **Examples**: Complete example datasets and notebooks

**Additional Dependencies**: `matplotlib`

### Feature-Specific Installation

Install only the features you need:

#### Visualization Features

```bash
pip install anfis-toolbox[visualization]
```

**Adds**: `matplotlib` for plotting capabilities

- `ANFISVisualizer` class
- `plot_membership_functions()`
- `plot_training_curves()`
- `plot_prediction_vs_target()`

Validation features are now built-in and do not require scikit-learn. You can still install scikit-learn optionally; if present, compatible utilities will delegate to it.

### Development Installation

For contributing to the project:

```bash
git clone https://github.com/dcruzf/anfis-toolbox.git
cd anfis-toolbox
pip install -e .[all,dev]
```

This includes all features plus development tools:

- `pytest` for testing
- `ruff` for linting and formatting
- `mypy` for type checking
- `mkdocs` for documentation

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade anfis-toolbox
```

To upgrade with all features:

```bash
pip install --upgrade anfis-toolbox[all]
```

## Verification

Verify your installation works correctly:

```python
import anfis_toolbox as anfis
import numpy as np

print(f"ANFIS Toolbox version: {anfis.__version__}")

# Test basic functionality
X = np.random.uniform(-1, 1, (50, 2))
y = X[:, 0]**2 + X[:, 1]**2

model = anfis.QuickANFIS.for_regression(X, n_mfs=2)
losses = model.fit_hybrid(X, y, epochs=10)

print("✅ Installation successful!")
print(f"Final training loss: {losses[-1]:.4f}")
```

Expected output:
```
ANFIS Toolbox version: 1.0.0
✅ Installation successful!
Final training loss: 0.0234
```

## Troubleshooting

### Common Issues

#### Import Error: No module named 'sklearn'

Validation no longer requires scikit-learn. If you want to use scikit-learn alongside, install it with:

```bash
pip install scikit-learn
```

#### Import Error: No module named 'matplotlib'

If you get this error when using visualization:

```bash
pip install anfis-toolbox[visualization]
# or
pip install matplotlib
```

#### Slow Training Performance

For better performance with large models:

```bash
# Install optimized NumPy (if available)
pip install numpy[mkl]

# Or consider switching to conda
conda install numpy matplotlib scikit-learn
pip install anfis-toolbox
```

#### Memory Issues

For large datasets or models:

1. Reduce batch size in training
2. Use fewer membership functions
3. Consider data preprocessing (normalization, dimensionality reduction)

### Platform-Specific Notes

#### Windows

No additional requirements. Make sure you have a recent Python version.

#### macOS

If you encounter compilation issues:

```bash
# Install Xcode command line tools
xcode-select --install

# Then retry installation
pip install anfis-toolbox[all]
```

#### Linux

Most distributions work out-of-the-box. For older systems:

```bash
# Update system packages first
sudo apt update && sudo apt upgrade

# Install Python development headers (Ubuntu/Debian)
sudo apt install python3-dev

# Install Python development headers (CentOS/RHEL)
sudo yum install python3-devel
```

## Alternative Installation Methods

### Using conda

While not officially distributed on conda-forge yet, you can still use conda for dependency management:

```bash
# Create environment
conda create -n anfis python=3.11
conda activate anfis

# Install dependencies via conda
conda install numpy scipy matplotlib scikit-learn

# Install ANFIS Toolbox via pip
pip install anfis-toolbox
```

### Using Poetry

If you're using Poetry for dependency management:

```toml
[tool.poetry.dependencies]
python = "^3.9"
anfis-toolbox = {extras = ["all"], version = "^1.0.0"}
```

Then run:
```bash
poetry install
```

### Using Docker

A Docker image with everything pre-installed:

```bash
docker pull dcruzf/anfis-toolbox:latest
docker run -it --rm -v $(pwd):/workspace dcruzf/anfis-toolbox:latest
```

## What's Next?

Now that you have ANFIS Toolbox installed, check out:

- 🚀 **[Quick Start](quickstart.md)** - Build your first model in 5 minutes
- 📖 **[Basic Tutorial](../guide/basic-usage.md)** - Learn the fundamentals
- 💡 **[Examples](../examples/basic.md)** - See practical applications

---

Having installation issues? [Open an issue](https://github.com/dcruzf/anfis-toolbox/issues) and we'll help you get started!
