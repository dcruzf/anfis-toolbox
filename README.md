# ANFIS Toolbox

A comprehensive Python implementation of Adaptive Neuro-Fuzzy Inference Systems (ANFIS) with Takagi-Sugeno-Kang (TSK) architecture.

[![Tests](https://github.com/username/anfis-toolbox/workflows/Tests/badge.svg)](https://github.com/username/anfis-toolbox/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Complete ANFIS Implementation**: Full 4-layer neural-fuzzy architecture
- **Hybrid Learning Algorithm**: Original Jang (1993) algorithm with least squares + backpropagation
- **Pure Backpropagation**: Modern gradient-based approach for comparison
- **Membership Functions**: Gaussian membership functions with automatic differentiation
- **Multi-Platform**: Supports Python 3.9-3.13 across different platforms
- **Comprehensive Testing**: 27+ tests with numerical gradient verification
- **Easy to Use**: Simple, intuitive API for creating and training ANFIS models

## Quick Start

```python
import numpy as np
from anfis_toolbox import ANFIS, GaussianMF

# Define input membership functions
input_mfs = {
    'x1': [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    'x2': [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
}

# Create and train ANFIS model
model = ANFIS(input_mfs)

# Generate training data
x_train = np.random.randn(100, 2)
y_train = np.sum(x_train, axis=1, keepdims=True)

# Train the model (hybrid algorithm - recommended)
losses = model.fit_hybrid(x_train, y_train, epochs=50, learning_rate=0.01)

# Alternative: pure backpropagation
# losses = model.fit(x_train, y_train, epochs=50, learning_rate=0.01)

# Make predictions
x_test = np.array([[0.5, -0.5], [1.0, 1.0]])
y_pred = model.predict(x_test)
```

## Installation

```bash
# From PyPI (when available)
pip install anfis-toolbox

# From source
git clone https://github.com/username/anfis-toolbox.git
cd anfis-toolbox
pip install -e .
```

## Architecture

The ANFIS toolbox implements a complete 4-layer neural-fuzzy architecture:

1. **Membership Layer**: Converts crisp inputs to fuzzy membership degrees
2. **Rule Layer**: Computes fuzzy rule firing strengths
3. **Normalization Layer**: Normalizes firing strengths
4. **Consequent Layer**: Applies TSK consequent functions

## Examples

See the `examples/` directory for comprehensive usage examples:
- 1D function approximation
- 2D function approximation
- Parameter inspection and modification

Run the examples:
```bash
uv run python examples/usage_examples.py
```

## Development

```bash
# Install development dependencies
uv install --dev

# Run tests across all Python versions
uv tool run hatch run test:test

# Run specific tests
uv run pytest tests/test_model.py -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

1. Jang, J. S. (1993). ANFIS: adaptive-network-based fuzzy inference system. IEEE transactions on systems, man, and cybernetics, 23(3), 665-685.
