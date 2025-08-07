# ANFIS Toolbox

A comprehensive Python implementation of Adaptive Neuro-Fuzzy Inference Systems (ANFIS) with Takagi-Sugeno-Kang (TSK) architecture.

[![Tests](https://github.com/username/anfis-toolbox/workflows/Tests/badge.svg)](https://github.com/username/anfis-toolbox/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Complete ANFIS Implementation**: Full 4-layer neural-fuzzy architecture
- **Membership Functions**: Gaussian membership functions with automatic differentiation
- **Training Algorithms**: Gradient-based parameter optimization
- **Multi-Platform**: Supports Python 3.9-3.13 across different platforms
- **Comprehensive Testing**: 25+ tests with numerical gradient verification
- **Easy to Use**: Simple, intuitive API for creating and training ANFIS models

## Installation

### Using pip

```bash
pip install anfis-toolbox
```

### From source

```bash
git clone https://github.com/username/anfis-toolbox.git
cd anfis-toolbox
pip install -e .
```

### Development installation

```bash
git clone https://github.com/username/anfis-toolbox.git
cd anfis-toolbox
uv install --dev  # or pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from anfis_toolbox import ANFIS, GaussianMF

# Define input membership functions
input_mfs = {
    'x1': [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    'x2': [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
}

# Create ANFIS model
model = ANFIS(input_mfs)

# Generate training data
x_train = np.random.randn(100, 2)
y_train = np.sum(x_train, axis=1, keepdims=True)  # Simple additive function

# Train the model
losses = model.fit(x_train, y_train, epochs=50, learning_rate=0.01)

# Make predictions
x_test = np.array([[0.5, -0.5], [1.0, 1.0]])
y_pred = model.predict(x_test)

print(f"Predictions: {y_pred.flatten()}")
```

## Architecture

The ANFIS toolbox implements a complete 4-layer neural-fuzzy architecture:

### Layer 1: Membership Layer
- Converts crisp inputs to fuzzy membership degrees
- Supports Gaussian membership functions
- Automatic gradient computation for backpropagation

### Layer 2: Rule Layer
- Implements fuzzy rule firing strengths
- Computes product of membership degrees for each rule
- Handles multiple input variables seamlessly

### Layer 3: Normalization Layer
- Normalizes rule firing strengths
- Ensures output consistency across different input ranges
- Gradient flow preservation for training

### Layer 4: Consequent Layer
- Implements TSK consequent functions
- Linear combination of input variables plus bias
- Adaptive parameters updated during training

## API Reference

### ANFIS Class

The main class for creating and training ANFIS models.

```python
class ANFIS:
    def __init__(self, input_mfs: Dict[str, List[MembershipFunction]])
    def forward(self, x: np.ndarray) -> np.ndarray
    def backward(self, dL_dy: np.ndarray) -> None
    def predict(self, x: np.ndarray) -> np.ndarray
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 100,
            learning_rate: float = 0.01, verbose: bool = True) -> List[float]
    def train_step(self, x: np.ndarray, y: np.ndarray,
                   learning_rate: float = 0.01) -> float
    def get_parameters(self) -> Dict
    def set_parameters(self, params: Dict) -> None
    def get_gradients(self) -> Dict
    def reset_gradients(self) -> None
```

#### Parameters

- **input_mfs**: Dictionary mapping input variable names to lists of membership functions
- **x**: Input data array of shape (n_samples, n_features)
- **y**: Target data array of shape (n_samples, 1)
- **epochs**: Number of training epochs
- **learning_rate**: Learning rate for gradient descent
- **verbose**: Whether to print training progress

### Membership Functions

#### GaussianMF

Gaussian membership function with automatic differentiation.

```python
class GaussianMF:
    def __init__(self, mean: float = 0.0, sigma: float = 1.0)
    def evaluate(self, x: np.ndarray) -> np.ndarray
    def gradient(self, x: np.ndarray, param: str) -> np.ndarray
```

#### Parameters

- **mean**: Center of the Gaussian function
- **sigma**: Standard deviation (width) of the Gaussian function
- **param**: Parameter name for gradient computation ('mean' or 'sigma')

## Examples

### 1D Function Approximation

```python
import numpy as np
from anfis_toolbox import ANFIS, GaussianMF

# Define target function
def target_function(x):
    return np.sin(x) + 0.1 * np.cos(5 * x)

# Create training data
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
y_train = target_function(x_train)

# Define membership functions
input_mfs = {
    'x': [
        GaussianMF(mean=-2*np.pi, sigma=1.5),
        GaussianMF(mean=-np.pi, sigma=1.5),
        GaussianMF(mean=0.0, sigma=1.5),
        GaussianMF(mean=np.pi, sigma=1.5),
        GaussianMF(mean=2*np.pi, sigma=1.5)
    ]
}

# Train model
model = ANFIS(input_mfs)
losses = model.fit(x_train, y_train, epochs=100, learning_rate=0.01)

# Make predictions
x_test = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y_pred = model.predict(x_test)
```

### 2D Function Approximation

```python
import numpy as np
from anfis_toolbox import ANFIS, GaussianMF

# Define 2D target function
def target_function(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.sin(r) / (r + 0.1)

# Create training data
np.random.seed(42)
n_samples = 200
x1_train = np.random.uniform(-3, 3, n_samples)
x2_train = np.random.uniform(-3, 3, n_samples)
x_train = np.column_stack([x1_train, x2_train])
y_train = target_function(x1_train, x2_train).reshape(-1, 1)

# Define membership functions for both inputs
input_mfs = {
    'x1': [GaussianMF(mean=-2.0, sigma=1.5), GaussianMF(mean=0.0, sigma=1.5), GaussianMF(mean=2.0, sigma=1.5)],
    'x2': [GaussianMF(mean=-2.0, sigma=1.5), GaussianMF(mean=0.0, sigma=1.5), GaussianMF(mean=2.0, sigma=1.5)]
}

# Train model
model = ANFIS(input_mfs)
losses = model.fit(x_train, y_train, epochs=150, learning_rate=0.01)
```

### Parameter Inspection and Modification

```python
# Get model parameters
params = model.get_parameters()
print("Membership function parameters:", params['membership'])
print("Consequent parameters:", params['consequent'])

# Modify parameters manually
modified_params = params.copy()
modified_params['membership']['x1'][0]['mean'] = -3.0
modified_params['consequent'] = np.ones_like(params['consequent'])

# Set modified parameters
model.set_parameters(modified_params)
```

## Development

### Running Tests

```bash
# Run all tests
uv tool run hatch run test:test

# Run tests for specific Python version
uv tool run hatch run test.py3.11:test

# Run with coverage
uv tool run hatch run test:cov
```

### Code Quality

```bash
# Format code
uv tool run hatch run lint:fmt

# Check code quality
uv tool run hatch run lint:check
```

### Building Documentation

```bash
# Build docs
uv tool run hatch run docs:build

# Serve docs locally
uv tool run hatch run docs:serve
```

## Mathematical Background

### ANFIS Architecture

ANFIS combines the learning capabilities of neural networks with the interpretability of fuzzy logic systems. The architecture implements a Takagi-Sugeno-Kang (TSK) fuzzy inference system with the following structure:

1. **Fuzzification**: Convert crisp inputs to fuzzy membership degrees
2. **Rule Activation**: Compute firing strength of each fuzzy rule
3. **Normalization**: Normalize firing strengths
4. **Consequent**: Apply TSK consequent functions
5. **Aggregation**: Combine outputs using weighted average

### Training Algorithm

The model uses gradient-based optimization to adjust:
- **Antecedent parameters**: Membership function centers and widths
- **Consequent parameters**: TSK function coefficients

Gradients are computed using automatic differentiation throughout all layers, ensuring mathematically correct parameter updates.

### Membership Functions

Currently supports Gaussian membership functions:

```
μ(x) = exp(-(x - c)² / (2σ²))
```

Where:
- `c` is the center (mean)
- `σ` is the standard deviation (width)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Clone the repository
2. Install development dependencies: `uv install --dev`
3. Run tests: `uv tool run hatch run test:test`
4. Make your changes
5. Add tests for new functionality
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolbox in your research, please cite:

```bibtex
@software{anfis_toolbox,
  title = {ANFIS Toolbox: A Python Implementation of Adaptive Neuro-Fuzzy Inference Systems},
  author = {Author Name},
  year = {2024},
  url = {https://github.com/username/anfis-toolbox},
  version = {0.1.0}
}
```

## Acknowledgments

- Inspired by the original ANFIS architecture by Jang (1993)
- Built with NumPy for efficient numerical computations
- Tested across multiple Python versions using Hatch and UV

## References

1. Jang, J. S. (1993). ANFIS: adaptive-network-based fuzzy inference system. IEEE transactions on systems, man, and cybernetics, 23(3), 665-685.
2. Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. IEEE transactions on systems, man, and cybernetics, (1), 116-132.
