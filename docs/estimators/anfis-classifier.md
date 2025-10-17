# ANFISClassifier

`ANFISClassifier` is the high-level entry point for training Adaptive
Neuro-Fuzzy Inference Systems on multi-class tasks. It hides the low-level
membership construction, rule synthesis, and trainer wiring behind a familiar
scikit-learn style API (`fit`, `predict`, `predict_proba`, `evaluate`, `save`,
`load`).

## At a Glance

- Works with NumPy arrays, array-like objects, or pandas DataFrames.
- Automatically generates membership functions per input (grid, FCM, or random).
- Supports custom membership definitions and rule subsets.
- Provides optimizers tailored for classification: `"adam"`, `"rmsprop"`,
    `"sgd"`, `"pso"`.
- Ships with built-in evaluation (`evaluate`) and persistence (`save`, `load`).

## Quick Start

```python
import numpy as np
from anfis_toolbox import ANFISClassifier

# Synthetic binary classification data
rng = np.random.default_rng(42)
X = rng.normal(size=(240, 2))
y = (X[:, 0] - 0.75 * X[:, 1] > 0).astype(int)

clf = ANFISClassifier(epochs=50, learning_rate=0.01, verbose=False)
clf.fit(X, y)

proba = clf.predict_proba([[0.2, -0.4]])
pred = clf.predict([[0.2, -0.4]])
report = clf.evaluate(X, y)
```

## Core Workflow

1. **Configure** – Set global defaults (`n_classes`, `n_mfs`, `mf_type`, `optimizer`).
2. **Fit** – Call `fit(X, y)` with optional validation data.
3. **Predict** – Use `predict` or `predict_proba` for inference.
4. **Evaluate** – Call `evaluate` to obtain accuracy, precision/recall/F1, and confusion matrix.
5. **Persist** – Store or restore trained estimators via `save` / `load`.

## Model Equations

Each fuzzy rule emits a Takagi–Sugeno–Kang consequent for every class:

$$
    \text{Rule}_i:\;\text{if } x_1 \text{ is } A_1^i \land \dots \land x_n \text{ is } A_n^i
\;\text{then}\; y_{i,k} = p_{0,k}^i + \sum_{j=1}^n p_{j,k}^i x_j.
$$

The firing strength of rule $i$ is the product of the memberships
$w_i = \prod_{j=1}^n \mu_{A_j^i}(x_j)$ and the normalised weights are
$\bar{w}_i = w_i / \sum_{r=1}^R w_r$. Class logits are the weighted sums
$z_k = \sum_{i=1}^R \bar{w}_i y_{i,k}$ and probabilities follow from the
softmax: $p_k = \exp(z_k) / \sum_{j=1}^K \exp(z_j)$. Training minimises
cross-entropy between the predicted probabilities and the target distribution.

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `n_classes` | Number of classes (optional, inferred on first `fit`). |
| `n_mfs` | Default membership count per input (int). |
| `mf_type` | Membership family (`"gaussian"`, `"triangular"`, `"bell"`, etc.). |
| `init` | Membership initialization (`"grid"`, `"fcm"`, `"random"`, or `None`). |
| `inputs_config` | Per-input overrides (dict, list of membership functions, or `None`). |
| `optimizer` | Trainer identifier, subclass, or instance (`"adam"`, `"sgd"`, `"rmsprop"`, `"pso"`). |
| `optimizer_params` | Extra keyword arguments forwarded to the trainer constructor. |
| `learning_rate`, `epochs`, `batch_size`, `shuffle`, `verbose` | Convenience overrides passed to compatible trainers. |
| `loss` | Optional custom loss (string key or callable). |
| `rules` | Optional list of rule tuples limiting the rule set. |

## Customizing Membership Functions

Use `inputs_config` to tailor membership families, counts, or ranges on a
per-input basis. Keys may be column names (for pandas DataFrames), integer
indices, or `"x{i}"` aliases.

```python
import numpy as np
from anfis_toolbox import ANFISClassifier
from anfis_toolbox.membership import GaussianMF

rng = np.random.default_rng(7)
X_multi = rng.normal(size=(300, 2))
y_multi = np.digitize(X_multi[:, 0] + 0.5 * X_multi[:, 1], bins=[-0.5, 0.5])

inputs_config = {
    0: {
        "mf_type": "triangular",
        "n_mfs": 3,
        "overlap": 0.55,
    },
    1: {
        "membership_functions": [
            GaussianMF(mean=-1.0, sigma=0.4),
            GaussianMF(mean=0.0, sigma=0.35),
            GaussianMF(mean=1.2, sigma=0.45),
        ]
    },
}

clf = ANFISClassifier(n_classes=3, inputs_config=inputs_config, epochs=60, learning_rate=0.01)
clf.fit(X_multi, y_multi)
```

!!! note
    Keep the number of membership functions consistent across inputs when
    mixing dictionary overrides and explicit membership lists. The example
    above configures three functions for each feature.

The `X_multi` and `y_multi` arrays from the example are reused in the sections
below.

## Choosing an Optimizer

Pass a string alias, trainer class, or trainer instance:

```python
clf = ANFISClassifier(optimizer="adam", epochs=80, learning_rate=0.005)
clf.fit(X, y)

from anfis_toolbox.optim import RMSPropTrainer

clf = ANFISClassifier(optimizer=RMSPropTrainer(learning_rate=0.001, epochs=120))
clf.fit(X, y)
```

- `"adam"` (default): Adaptive gradient-based training.
- `"rmsprop"`: Root-mean-square propagation.
- `"sgd"`: Mini-batch stochastic gradient descent.
- `"pso"`: Particle Swarm Optimisation for derivative-free updates.

Hybrid optimisers that rely on least-squares refinements are limited to
regression and are rejected by `ANFISClassifier`.

## Restricting the Rule Base

Supply `rules` to freeze the rule combinations used during training.

```python
selected_rules = [(0, 0), (1, 1), (2, 2)]
clf = ANFISClassifier(n_classes=3, rules=selected_rules, epochs=40, learning_rate=0.01)
clf.fit(X_multi, y_multi)
assert tuple(clf.get_rules()) == tuple(selected_rules)
```

If `rules` is omitted, the full Cartesian product of membership indices is used.

## Evaluating Performance

`evaluate` reports accuracy, precision/recall/F1 averages, balanced accuracy,
and the confusion matrix. Disable printing with `print_results=False`.

```python
metrics = clf.evaluate(X_test, y_test, print_results=False)
print(metrics["accuracy"], metrics["macro_f1"])

proba = clf.predict_proba(X_test[:3])
labels = clf.predict(X_test[:3])
```

## Saving and Loading Models

```python
clf.fit(X, y)
clf.save("artifacts/anfis-classifier.pkl")

from anfis_toolbox import ANFISClassifier

loaded = ANFISClassifier.load("artifacts/anfis-classifier.pkl")
pred = loaded.predict(X[:3])
```

The pickled artefact stores fitted membership functions, rule definitions, and
training history, enabling reproducible deployments.

## Tips & Troubleshooting

- **Input scale** – Normalize or standardize features for smoother membership learning.
- **Underfitting** – Increase `n_mfs`, provide richer `inputs_config`, or allow more epochs.
- **Overfitting** – Reduce rule count, add validation data, or lower `epochs`.
- **Imbalanced labels** – Use class-balanced datasets or resampling strategies.
- **Verbose logging** – Set `verbose=True` during fitting to stream trainer progress.

## Further Reading

- [API Reference – Classifier](../api/classifier.md)
- [Membership Functions catalog](../api/membership-functions.md)
- [Optimizer reference](../api/optim.md)
- Jang, J.-S. R. (1993). *ANFIS: Adaptive-network-based fuzzy inference system*.
