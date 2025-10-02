# Fuzzy C-Means Clustering

The Fuzzy C-Means (FCM) clustering algorithm is a soft clustering method that assigns each data point to multiple clusters with varying degrees of membership. It's particularly useful for fuzzy logic applications and can be used to initialize membership functions for ANFIS models.

## Overview

Unlike hard clustering methods like K-Means that assign each point to exactly one cluster, FCM allows points to belong to multiple clusters simultaneously. This makes it suitable for applications where data points have ambiguous cluster memberships.

## Mathematical Foundation

FCM minimizes the objective function:

$J_m(U, V) = \sum_{i=1}^n \sum_{k=1}^c u_{ik}^m \|x_i - v_k\|^2$

Subject to:
- $\sum_{k=1}^c u_{ik} = 1$ for all $i$
- $u_{ik} \in [0, 1]$ for all $i, k$

Where:
- $u_{ik}$ is the membership degree of point $i$ to cluster $k$
- $v_k$ is the center of cluster $k$
- $m > 1$ is the fuzzifier parameter
- $c$ is the number of clusters

The algorithm iteratively updates memberships and centers using:

$u_{ik} = \frac{1}{\sum_{j=1}^c \left(\frac{\|x_i - v_k\|}{\|x_i - v_j\|}\right)^{\frac{2}{m-1}}}$

$v_k = \frac{\sum_{i=1}^n u_{ik}^m x_i}{\sum_{i=1}^n u_{ik}^m}$

## FuzzyCMeans Class

The `FuzzyCMeans` class implements the Fuzzy C-Means algorithm.

### Initialization

```python
from anfis_toolbox.clustering import FuzzyCMeans

# Basic usage
fcm = FuzzyCMeans(n_clusters=3, m=2.0)

# With custom parameters
fcm = FuzzyCMeans(
    n_clusters=4,
    m=1.5,
    max_iter=200,
    tol=1e-5,
    random_state=42
)
```

### Key Parameters

- `n_clusters`: Number of clusters (≥ 2)
- `m`: Fuzzifier parameter (> 1, default 2.0)
- `max_iter`: Maximum iterations (default 300)
- `tol`: Convergence tolerance (default 1e-4)
- `random_state`: Random seed for reproducibility

### Key Methods

- `fit(X)`: Fit the FCM model to data
- `fit_predict(X)`: Fit and return hard cluster labels
- `predict(X)`: Return hard labels for new data
- `predict_proba(X)`: Return fuzzy memberships for new data
- `transform(X)`: Alias for predict_proba

### Example Usage

```python
import numpy as np
from anfis_toolbox.clustering import FuzzyCMeans

# Generate sample data
X = np.random.randn(150, 2)
X[:50] += [2, 2]   # Cluster 1
X[50:100] += [-2, 2]  # Cluster 2
X[100:] += [0, -2]    # Cluster 3

# Fit FCM
fcm = FuzzyCMeans(n_clusters=3, random_state=42)
fcm.fit(X)

# Get results
centers = fcm.cluster_centers_
memberships = fcm.membership_
hard_labels = fcm.predict(X)

print(f"Cluster centers shape: {centers.shape}")
print(f"Membership matrix shape: {memberships.shape}")
```

## Evaluation Metrics

FCM provides several metrics to evaluate clustering quality:

### Partition Coefficient (PC)

Measures the amount of fuzziness in the partition:

$PC = \frac{1}{n} \sum_{i=1}^n \sum_{k=1}^c u_{ik}^2$

Higher values (closer to 1) indicate crisper partitions.

```python
pc = fcm.partition_coefficient()
print(f"Partition Coefficient: {pc:.4f}")
```

### Classification Entropy (CE)

Measures the entropy of the membership distribution:

$CE = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^c u_{ik} \log u_{ik}$

Lower values indicate better clustering.

```python
ce = fcm.classification_entropy()
print(f"Classification Entropy: {ce:.4f}")
```

### Xie-Beni Index (XB)

Combines compactness and separation:

$XB = \frac{\sum_{i=1}^n \sum_{k=1}^c u_{ik}^m \|x_i - v_k\|^2}{n \cdot \min_{p \neq q} \|v_p - v_q\|^2}$

Lower values indicate better clustering.

```python
xb = fcm.xie_beni_index(X)
print(f"Xie-Beni Index: {xb:.4f}")
```

## Advanced Usage

### Using FCM for ANFIS Initialization

FCM can be used to initialize membership functions for ANFIS models:

```python
from anfis_toolbox.membership import GaussianMF

# Cluster data to find centers
fcm = FuzzyCMeans(n_clusters=3)
fcm.fit(X)

# Use cluster centers to initialize Gaussians
centers = fcm.cluster_centers_[:, 0]  # Assuming 1D input
input_mfs = {
    'x1': [GaussianMF(center, 1.0) for center in centers]
}
```

### Custom Fuzzifier

The fuzzifier `m` controls the fuzziness of the clustering:

- `m → 1`: Approaches hard clustering (K-Means)
- `m → ∞`: Maximum fuzziness, all memberships equal

```python
# Hard clustering approximation
hard_fcm = FuzzyCMeans(n_clusters=3, m=1.01)

# Very fuzzy clustering
fuzzy_fcm = FuzzyCMeans(n_clusters=3, m=5.0)
```

## Performance Considerations

- **Convergence**: FCM typically converges in fewer iterations than expected
- **Computational Complexity**: O(n × c × d × iter), where n=samples, c=clusters, d=dimensions
- **Memory Usage**: Stores membership matrix (n × c) and centers (c × d)
- **Scalability**: Suitable for small to medium datasets (< 10,000 samples)

## Troubleshooting

- **Poor Convergence**: Increase `max_iter` or decrease `tol`
- **All Points in One Cluster**: Try different `random_state` or increase `m`
- **Numerical Issues**: Ensure data is scaled to similar ranges

## References

- Bezdek, J. C. (1981). Pattern Recognition with Fuzzy Objective Function Algorithms. Springer.
- Xie, X. L., & Beni, G. (1991). A validity measure for fuzzy clustering. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(8), 841-847.
