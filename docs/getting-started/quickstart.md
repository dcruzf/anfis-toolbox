# Quick Start

Get up and running with ANFIS Toolbox in just 5 minutes! This tutorial will walk you through creating your first neuro-fuzzy model.

## What You'll Learn

By the end of this tutorial, you'll know how to:

- ✅ Load or generate data for training
- ✅ Create an ANFIS model with one line of code
- ✅ Train the model using hybrid learning
- ✅ Make predictions and evaluate performance
- ✅ Visualize results and membership functions

## Prerequisites

Make sure you have ANFIS Toolbox installed:

```bash
pip install anfis-toolbox[all]
```

## Your First ANFIS Model

Let's create a model to approximate the function `f(x, y) = x² + y²`:

### Step 1: Import and Prepare Data

```python
import numpy as np
from anfis_toolbox import QuickANFIS, quick_evaluate

# Generate training data
np.random.seed(42)  # For reproducible results
n_samples = 200

# Create 2D input space: x and y from -2 to 2
X = np.random.uniform(-2, 2, (n_samples, 2))

# Target function: f(x, y) = x² + y²
y = X[:, 0]**2 + X[:, 1]**2

# Add some noise to make it realistic
y += np.random.normal(0, 0.1, n_samples)

print(f"Training data shape: X={X.shape}, y={y.shape}")
print(f"Input range: [{X.min():.2f}, {X.max():.2f}]")
print(f"Output range: [{y.min():.2f}, {y.max():.2f}]")
```

### Step 2: Create the Model

```python
# Create ANFIS model with QuickANFIS (one line!)
model = QuickANFIS.for_regression(
    X=X,
    n_mfs=3,  # 3 membership functions per input
    mf_type='gaussian'  # Gaussian membership functions
)

print(f"Model created with {model.n_rules} rules")
print(f"Total parameters: {len(model.get_parameters())}")
```

### Step 3: Train the Model

```python
# Train using hybrid learning (LSE + backpropagation)
print("Training model...")
losses = model.fit_hybrid(
    X=X,
    y=y,
    epochs=100,
    learning_rate=0.01,
    verbose=True
)

print(f"✅ Training complete!")
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
```

### Step 4: Evaluate Performance

```python
# Quick evaluation with common metrics
metrics = quick_evaluate(model, X, y)

print("\n📊 Model Performance:")
print(f"R² Score: {metrics['r2']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAPE: {metrics['mape']:.2f}%")
```

### Step 5: Make Predictions

```python
# Test on new data points
test_points = np.array([
    [1.0, 1.0],    # Should predict ~2.0
    [0.0, 0.0],    # Should predict ~0.0
    [-1.5, 0.5],   # Should predict ~2.5
    [2.0, -2.0]    # Should predict ~8.0
])

predictions = model.predict(test_points)

print("\n🎯 Predictions vs Expected:")
for i, (point, pred) in enumerate(zip(test_points, predictions)):
    expected = point[0]**2 + point[1]**2
    error = abs(pred - expected)
    print(f"Point {point}: predicted={pred:.3f}, expected={expected:.3f}, error={error:.3f}")
```

### Step 6: Visualize Results (Optional)

If you installed with visualization features:

```python
from anfis_toolbox import ANFISVisualizer

# Create visualizer
viz = ANFISVisualizer(model)

# Plot membership functions for each input
viz.plot_membership_functions()

# Plot training progress
viz.plot_training_curves(losses)

# Plot predictions vs targets
viz.plot_prediction_vs_target(X, y)

print("📈 Plots displayed! Check the figures.")
```

## Complete Example

Here's the complete working example:

```python
import numpy as np
from anfis_toolbox import QuickANFIS, quick_evaluate

# 1. Generate data
np.random.seed(42)
X = np.random.uniform(-2, 2, (200, 2))
y = X[:, 0]**2 + X[:, 1]**2 + np.random.normal(0, 0.1, 200)

# 2. Create and train model
model = QuickANFIS.for_regression(X, n_mfs=3)
losses = model.fit_hybrid(X, y, epochs=100)

# 3. Evaluate
metrics = quick_evaluate(model, X, y)
print(f"R² Score: {metrics['r2']:.4f}")

# 4. Predict
predictions = model.predict([[1.0, 1.0], [0.0, 0.0]])
print(f"Predictions: {predictions}")

# 5. Visualize (if matplotlib available)
try:
    from anfis_toolbox import ANFISVisualizer
    viz = ANFISVisualizer(model)
    viz.plot_membership_functions()
except ImportError:
    print("Install matplotlib for visualization: pip install anfis-toolbox[visualization]")
```

Expected output:
```
R² Score: 0.9876
Predictions: [1.987 0.023]
```

## Understanding the Results

### Training Loss Curve

A good ANFIS model should show:
- **Decreasing loss**: Steady improvement over epochs
- **Convergence**: Loss stabilizes after some epochs
- **No overfitting**: Validation loss (if used) follows training loss

### Performance Metrics

- **R² Score**: Closer to 1.0 is better (0.95+ is excellent)
- **MAE**: Mean Absolute Error - lower is better
- **RMSE**: Root Mean Square Error - penalizes large errors
- **MAPE**: Mean Absolute Percentage Error - good for comparing across scales

### Membership Functions

After training, the membership functions adapt to the data:
- **Centers shift** to important regions
- **Widths adjust** for optimal coverage
- **Shapes optimize** for the target function

## Next Steps

🎉 Congratulations! You've created your first ANFIS model. Now you can:

### Try Different Configurations

```python
# Different membership function types
model_bell = QuickANFIS.for_regression(X, n_mfs=4, mf_type='bell')
model_triangular = QuickANFIS.for_regression(X, n_mfs=2, mf_type='triangular')

# Different training methods
losses_bp = model.fit(X, y, epochs=50)  # Pure backpropagation
losses_hybrid = model.fit_hybrid(X, y, epochs=50)  # Hybrid (recommended)
```

### Use Advanced Features

```python
# Cross-validation (needs sklearn)
from anfis_toolbox import ANFISValidator

validator = ANFISValidator(model)
cv_scores = validator.cross_validate(X, y, cv=5)
print(f"CV R²: {cv_scores['r2_mean']:.3f} ± {cv_scores['r2_std']:.3f}")

# Learning curves
learning_data = validator.learning_curve(X, y, train_sizes=[0.2, 0.5, 0.8])
```

### Save and Load Models

```python
# Save trained model
model.save('my_anfis_model.pkl')

# Load later
from anfis_toolbox import load_anfis
loaded_model = load_anfis('my_anfis_model.pkl')
```

## Common Issues & Solutions

### "Training loss not decreasing"

- Try different learning rates: `0.001`, `0.01`, `0.1`
- Reduce number of membership functions
- Check if data is normalized
- Use hybrid learning instead of pure backpropagation

### "Poor prediction accuracy"

- Increase number of membership functions
- Try different MF types (`'gaussian'`, `'bell'`, `'triangular'`)
- Check for sufficient training data
- Consider data preprocessing (normalization, feature engineering)

### "Training too slow"

- Reduce number of epochs or membership functions
- Use smaller datasets for initial testing
- Consider using pure analytical methods for simple problems

## What's Next?

Now that you've mastered the basics, explore:

- 📚 **[User Guide](../guide/introduction.md)** - Deep dive into ANFIS theory and advanced usage
- 💡 **[Examples](../examples/basic.md)** - Real-world applications and use cases
- 🔧 **[API Reference](../api/overview.md)** - Complete documentation of all classes and methods
- 🎯 **[Advanced Tutorial](../guide/advanced-usage.md)** - Custom membership functions, manual model building

---

Ready to build more sophisticated models? [Continue with the User Guide →](../guide/introduction.md)
