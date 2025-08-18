# Examples

Welcome to the ANFIS Toolbox examples! This section provides practical, real-world applications demonstrating how to use ANFIS for various tasks.

## 📂 Example Categories

### 🎯 Basic Examples
Perfect for getting started and understanding core concepts.

- **[Basic Function Approximation](basic.md)** - Learn nonlinear functions

### 🏭 Real-World Applications
More examples coming soon.

### 🔬 Advanced Techniques
More examples coming soon.

### 🛠️ Integration Examples
More examples coming soon.

## 🚀 Quick Start Examples

### 30-Second Example
```python
import numpy as np
from anfis_toolbox import QuickANFIS

X = np.random.uniform(-2, 2, (100, 2))
y = X[:, 0]**2 + X[:, 1]**2

model = QuickANFIS.for_regression(X)
model.fit_hybrid(X, y, epochs=50)
print(f"Prediction: {model.predict([[1, 1]])}")
```

### 2-Minute Example
```python
import numpy as np
from anfis_toolbox import QuickANFIS

# Generate data
X = np.random.uniform(-3, 3, (200, 2))
y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(200)

# Create and train model
model = QuickANFIS.for_regression(X, n_mfs=4, mf_type='gaussian')
losses = model.fit_hybrid(X, y, epochs=100, learning_rate=0.01)

# Evaluate and visualize
# Evaluate (placeholder)
pred = model.predict(X[:5])
print(pred[:3])
```

### 5-Minute Example
```python
import numpy as np
from anfis_toolbox import ANFISBuilder

# Prepare data
X = np.random.uniform(-1, 1, (300, 3))
y = X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + 0.05 * np.random.randn(300)

# Configure model
config = dict(n_epochs=150, learning_rate=0.02)

# Build model with custom architecture
builder = (ANFISBuilder()
    .add_input('x1', 'gaussian', n_mfs=3)
    .add_input('x2', 'bell', n_mfs=4)
    .add_input('x3', 'triangular', n_mfs=2)
    .set_config(config)
)

model = builder.build()

# Train with validation
losses = model.fit_hybrid(X, y, **config)

# Comprehensive evaluation
# Placeholder evaluation
print(f"Loss history (len): {len(losses)}")

# Save model
# Placeholder persistence
# model.save('complex_model.pkl')
```

## 📊 Example Datasets

All examples use the following built-in datasets:

### Synthetic Functions
```python
from anfis_toolbox.datasets import (
    polynomial_2d,      # f(x,y) = x² + xy + y²
    sinusoidal_2d,      # f(x,y) = sin(x)cos(y)
    nonlinear_3d,       # f(x,y,z) = xyz + sin(x+y+z)
    control_surface     # Control system response surface
)

# Usage
X, y = polynomial_2d(n_samples=200, noise=0.1)
```

### Real-World Data
```python
from anfis_toolbox.datasets import (
    boston_housing,     # Modified Boston housing (regression)
    wine_quality,       # Wine quality prediction
    air_quality,        # Air pollution levels
    stock_returns       # Financial time series
)

# Usage
X, y = wine_quality(return_X_y=True)
```

## 🎯 Example Difficulty Levels

### 🟢 Beginner (Green)
- **Prerequisites**: Basic Python, NumPy
- **Concepts**: Simple ANFIS usage, QuickANFIS
- **Time**: 15-30 minutes each
- **Examples**: [Basic Function Approximation](basic.md), [Simple Regression](regression.md)

### 🟡 Intermediate (Yellow)
- **Prerequisites**: ML basics, some fuzzy logic knowledge
- **Concepts**: Model tuning, validation, visualization
- **Time**: 30-60 minutes each
- **Examples**: [Stock Prediction](stock-prediction.md), [Temperature Control](temperature-control.md)

### 🔴 Advanced (Red)
- **Prerequisites**: Strong ML background, ANFIS theory
- **Concepts**: Custom implementations, advanced techniques
- **Time**: 1-2 hours each
- **Examples**: [Ensemble ANFIS](ensemble.md), [Online Learning](online-learning.md)

## 💻 Running Examples

### Online (Recommended)
All examples are available as interactive Jupyter notebooks:

- **Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dcruzf/anfis-toolbox/examples)
- **Binder**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dcruzf/anfis-toolbox/main?filepath=examples)

### Local Installation
```bash
# Clone repository
git clone https://github.com/dcruzf/anfis-toolbox.git
cd anfis-toolbox

# Install with examples
pip install -e .[all,examples]

# Run Jupyter
jupyter notebook examples/
```

### Dependencies for Examples
```bash
# Core examples
pip install anfis-toolbox[all]

# Advanced examples (additional dependencies)
pip install seaborn plotly scikit-optimize
```

## 📁 Example Structure

Each example follows a consistent structure:

```
example-name/
├── README.md              # Overview and learning objectives
├── example-name.ipynb     # Main Jupyter notebook
├── example-name.py        # Standalone Python script
├── data/                  # Example-specific datasets
├── results/               # Pre-computed results
└── figures/               # Generated plots and visualizations
```

### Notebook Sections

1. **📖 Introduction** - Problem description and objectives
2. **📊 Data Exploration** - Understanding the dataset
3. **🏗️ Model Building** - ANFIS construction and configuration
4. **🎯 Training** - Model training with progress monitoring
5. **📈 Evaluation** - Performance analysis and metrics
6. **🔍 Interpretation** - Rule extraction and analysis
7. **💡 Insights** - Key takeaways and next steps

## 🛠️ Code Style and Best Practices

All examples follow these conventions:

### Imports
```python
# Standard library
import numpy as np
import matplotlib.pyplot as plt

# ANFIS Toolbox
from anfis_toolbox import QuickANFIS, ANFISBuilder
from anfis_toolbox import QuickANFIS

# Optional dependencies (with fallbacks)
import warnings
```

### Configuration
```python
# Reproducible results
np.random.seed(42)

# Plot settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
```

### Error Handling
```python
# Graceful degradation for optional features
try:
    # Visualization code
    viz = ANFISVisualizer(model)
    viz.plot_membership_functions()
except ImportError as e:
    print(f"Visualization unavailable: {e}")
    print("Install with: pip install anfis-toolbox[visualization]")
```

## 🏆 Featured Examples

### 🥇 Most Popular
1. **[Stock Price Prediction](stock-prediction.md)** - Real financial data analysis
2. **[Temperature Control](temperature-control.md)** - Classic control system
3. **[Medical Diagnosis](medical-diagnosis.md)** - Healthcare decision support

### 🆕 Recently Added
1. **[Ensemble ANFIS](ensemble.md)** - Multiple model combination
2. **[Online Learning](online-learning.md)** - Adaptive, streaming models
3. **[Custom MF Example](custom-mf-example.md)** - Extending the library

### 🎯 Most Educational
1. **[Basic Function Approximation](basic.md)** - Perfect introduction
2. **[Model Architecture Deep Dive](../guide/model-architecture.md)** - Understanding internals
3. **[Custom Membership Functions](custom-mf-example.md)** - Advanced customization

## 🤝 Contributing Examples

Want to add your own example? Great! Please:

1. **Follow the structure** described above
2. **Include comprehensive documentation**
3. **Add learning objectives** and difficulty level
4. **Test thoroughly** across different Python versions
5. **Submit a pull request** with description

### Example Contribution Template
```python
"""
Example: Your Example Name

Learning Objectives:
- Objective 1
- Objective 2
- Objective 3

Difficulty: Beginner/Intermediate/Advanced
Time: X minutes
Prerequisites: List prerequisites

Tags: tag1, tag2, tag3
"""

# Your example code here...
```

## 📞 Getting Help

- **Issues with examples**: [GitHub Issues](https://github.com/dcruzf/anfis-toolbox/issues)
- **General questions**: [GitHub Discussions](https://github.com/dcruzf/anfis-toolbox/discussions)
- **Feature requests**: [Feature Request Template](https://github.com/dcruzf/anfis-toolbox/issues/new?template=feature_request.md)

---

Ready to explore? **[Start with Basic Function Approximation →](basic.md)**
