# ANFIS Classifier

The ANFIS Classifier is a variant of the Adaptive Neuro-Fuzzy Inference System designed for multi-class classification tasks. It extends the ANFIS architecture with a softmax output layer and trains using cross-entropy loss.

## Overview

The ANFIS Classifier uses the same four-layer architecture as the regression model:

1. **Membership Layer**: Fuzzifies crisp inputs using membership functions.
2. **Rule Layer**: Computes rule strengths using T-norm operations.
3. **Normalization Layer**: Normalizes rule weights to ensure they sum to 1.
4. **Consequent Layer**: Computes class logits using Takagi-Sugeno-Kang (TSK) models with multiple outputs.

## Mathematical Foundation

Each rule produces logits for all classes:

**If** $x_1$ is $A_1^i$ **and** $x_2$ is $A_2^i$ **... and** $x_n$ is $A_n^i$ **then** $y^i_k = p_{0,k}^i + p_{1,k}^i x_1 + \dots + p_{n,k}^i x_n$ for $k = 1, \dots, K$

Where $K$ is the number of classes.

The final logits are weighted sums: $z_k = \sum_{i=1}^R w^i y^i_k / \sum_{i=1}^R w^i$

Probabilities are computed via softmax: $p_k = \exp(z_k) / \sum_{j=1}^K \exp(z_j)$

Training minimizes cross-entropy loss.

## ANFISClassifier Class

The `ANFISClassifier` class implements the classification variant of the ANFIS model.

### Initialization

```python
from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.model import ANFISClassifier

input_mfs = {
    'x1': [GaussianMF(0, 1), GaussianMF(1, 1)],
    'x2': [GaussianMF(0, 1), GaussianMF(1, 1)]
}
classifier = ANFISClassifier(input_mfs, n_classes=3)
```

### Key Methods

- `forward(x)`: Returns logits for classification.
- `predict_proba(x)`: Returns class probabilities.
- `predict(x)`: Returns predicted class labels.
- `fit(X, y, epochs=100, learning_rate=0.01)`: Trains the classifier.
- `get_parameters()` / `set_parameters()`: For parameter management.
- `update_parameters(learning_rate)`: Applies gradient descent updates.

### Example Usage

```python
import numpy as np

# Generate classification data
X = np.random.randn(100, 2)
y = np.random.randint(0, 3, 100)

# Train the classifier
losses = classifier.fit(X, y, epochs=50, learning_rate=0.01)

# Make predictions
probabilities = classifier.predict_proba(X)
labels = classifier.predict(X)
```

## Detailed Architecture

### Layer 1: Membership Layer

For each input $x_j$, computes membership degrees $\mu_{A_j^i}(x_j)$ for each fuzzy set $A_j^i$.

### Layer 2: Rule Layer

Computes firing strengths $w^i = \prod_{j=1}^n \mu_{A_j^i}(x_j)$ using product T-norm.

### Layer 3: Normalization Layer

Normalizes weights: $\bar{w}^i = w^i / \sum_{k=1}^R w^k$

### Layer 4: Consequent Layer

Computes rule logits $y^i_k = \sum_{m=0}^n p_{m,k}^i x_m$, then final logits $z_k = \sum_{i=1}^R \bar{w}^i y^i_k$

## Training Process

The classifier uses gradient descent on cross-entropy loss:

1. **Forward Pass**: Compute logits and probabilities.
2. **Loss Computation**: Cross-entropy between true labels and predicted probabilities.
3. **Backward Pass**: Compute gradients through softmax and all layers.
4. **Parameter Update**: Update membership and consequent parameters.

## Choosing Membership Functions

- **Number of MFs per input**: Start with 2-3, increase for complex decision boundaries.
- **MF Types**: Gaussian for smooth boundaries, triangular for piecewise linear.
- **Initialization**: Place MFs to cover the input range evenly.

## Evaluation and Metrics

Use classification metrics like accuracy, cross-entropy:

```python
from anfis_toolbox.metrics import accuracy, cross_entropy

acc = accuracy(y_true, predictions)
loss = cross_entropy(y_true_onehot, logits)
```

## Advanced Usage

### Custom Training

```python
# Manual training loop
for epoch in range(100):
    classifier.reset_gradients()
    logits = classifier.forward(X)
    # Compute loss and gradients...
    classifier.update_parameters(0.01)
```

### Parameter Management

```python
# Save parameters
params = classifier.get_parameters()

# Load parameters
classifier.set_parameters(params)
```

## Performance Considerations

- **Scalability**: Rule count grows exponentially; suitable for low-dimensional data.
- **Training Time**: May require more epochs than regression due to cross-entropy.
- **Memory Usage**: Scales with rules × classes × inputs.

## Troubleshooting

- **Poor Accuracy**: Increase membership functions or epochs.
- **Overfitting**: Reduce rules or add regularization.
- **Class Imbalance**: Ensure balanced training data.

## References

- Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.
