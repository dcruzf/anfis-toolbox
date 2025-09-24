# ANFIS Model

The ANFIS (Adaptive Neuro-Fuzzy Inference System) model is the core component of the ANFIS Toolbox. It implements a fuzzy inference system that combines the benefits of fuzzy logic and neural networks for function approximation tasks.

## Overview

The ANFIS architecture consists of four layers:

1. **Membership Layer**: Fuzzifies crisp inputs using membership functions.
2. **Rule Layer**: Computes rule strengths using T-norm operations.
3. **Normalization Layer**: Normalizes rule weights to ensure they sum to 1.
4. **Consequent Layer**: Computes the final output using Takagi-Sugeno-Kang (TSK) models.

## Mathematical Foundation

ANFIS is based on the Takagi-Sugeno fuzzy model. Each rule has the form:

**If** $x_1$ is $A_1^i$ **and** $x_2$ is $A_2^i$ **... and** $x_n$ is $A_n^i$ **then** $y^i = p_0^i + p_1^i x_1 + \dots + p_n^i x_n$

Where $A_j^i$ are fuzzy sets, and $p_k^i$ are consequent parameters.

The overall output is a weighted average of rule outputs:

$y = \sum_{i=1}^R w^i y^i / \sum_{i=1}^R w^i$

Where $w^i$ is the firing strength of rule $i$.

## ANFIS Class

The `ANFIS` class implements the regression variant of the ANFIS model.

### Initialization

```python
from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.model import ANFIS

input_mfs = {
    'x1': [GaussianMF(0, 1), GaussianMF(1, 1)],
    'x2': [GaussianMF(0, 1), GaussianMF(1, 1)]
}
model = ANFIS(input_mfs)
```

### Key Methods

- `forward(x)`: Performs a forward pass through the network.
- `backward(dL_dy)`: Computes gradients via backpropagation.
- `predict(x)`: Makes predictions on input data.
- `fit(x, y, epochs=100, learning_rate=0.01)`: Trains the model using hybrid learning.
- `get_parameters()` / `set_parameters()`: For parameter management.
- `update_parameters(learning_rate)`: Applies gradient descent updates.

### Example Usage

```python
import numpy as np

# Generate sample data
X = np.random.randn(100, 2)
y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

# Train the model
losses = model.fit(X, y, epochs=50, learning_rate=0.01)

# Make predictions
predictions = model.predict(X)
```

## Detailed Architecture

### Layer 1: Membership Layer

For each input $x_j$, computes membership degrees $\mu_{A_j^i}(x_j)$ for each fuzzy set $A_j^i$.

### Layer 2: Rule Layer

Computes firing strengths $w^i = \prod_{j=1}^n \mu_{A_j^i}(x_j)$ using product T-norm.

### Layer 3: Normalization Layer

Normalizes weights: $\bar{w}^i = w^i / \sum_{k=1}^R w^k$

### Layer 4: Consequent Layer

Computes rule outputs $y^i = \sum_{k=0}^n p_k^i x_k$ (with $x_0 = 1$), then final output $y = \sum_{i=1}^R \bar{w}^i y^i$

## Training Process

ANFIS uses hybrid learning:

1. **Forward Pass**: Compute outputs using current premise parameters.
2. **Consequent Parameter Estimation**: Use least squares to find optimal $p_k^i$.
3. **Backward Pass**: Compute gradients for premise parameters.
4. **Parameter Update**: Update membership function parameters via gradient descent.

This typically converges faster than pure backpropagation.

## Choosing Membership Functions

- **Number of MFs per input**: Start with 2-3, increase for complex functions.
- **MF Types**: Gaussian for smooth functions, triangular for piecewise linear.
- **Initialization**: Place MFs to cover the input range evenly.

## Evaluation and Metrics

Use regression metrics like MSE, RMSE, R²:

```python
from anfis_toolbox.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_true, predictions)
r2 = r2_score(y_true, predictions)
```

## Advanced Usage

### Custom Trainers

```python
from anfis_toolbox.optim import HybridTrainer

trainer = HybridTrainer(learning_rate=0.01, epochs=200)
losses = trainer.fit(model, X, y)
```

### Parameter Management

```python
# Save parameters
params = model.get_parameters()

# Load parameters
model.set_parameters(params)
```

## Performance Considerations

- **Scalability**: The number of rules grows exponentially with the number of membership functions per input.
- **Training Time**: Hybrid training typically converges faster than pure backpropagation.
- **Memory Usage**: Parameter storage scales with the number of rules and inputs.

## Troubleshooting

- **Poor Convergence**: Try more epochs, adjust learning rate, or add more membership functions.
- **Overfitting**: Use fewer rules or add regularization (via custom trainers).
- **Numerical Issues**: Ensure inputs are scaled to similar ranges.

## References

- Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.
