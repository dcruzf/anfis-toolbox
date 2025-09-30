# User Guide

Welcome to the comprehensive ANFIS Toolbox User Guide! This section provides in-depth coverage of all features, concepts, and best practices for building effective neuro-fuzzy systems.

## 📖 What's in This Guide

### 🎯 Core Concepts
- **[Introduction](introduction.md)** - ANFIS theory and architecture
- Membership Functions - Types, properties, and selection
- Training Methods - Hybrid learning vs. backpropagation
- Model Architecture - Understanding the 5-layer structure

### 🛠️ Practical Usage
- Basic Usage - Essential patterns and workflows
- Advanced Usage - Custom configurations and fine-tuning
- Builder Patterns - Using ANFISRegressor and ANFISBuilder
- Model Persistence - Saving, loading, and deployment

### 📊 Analysis & Visualization
- Validation - Cross-validation, metrics, and model comparison
- Visualization - Plotting membership functions and results
- Performance Tuning - Optimization strategies

### 🔧 Advanced Topics
- Configuration Management - ANFISConfig and settings
- Custom Membership Functions - Creating your own MF types
- Integration - Using with scikit-learn and other libraries

## 🚀 Quick Navigation

### I want to...

| **Goal** | **Start Here** | **Level** |
|----------|----------------|-----------|
| Understand ANFIS theory | Introduction | Beginner |
| Build my first model | Basic Usage | Beginner |
| Compare different models | Validation | Intermediate |
| Optimize model performance | Performance Tuning | Intermediate |
| Create custom MF types | Custom Membership Functions | Advanced |
| Deploy models in production | Model Persistence | Intermediate |

### By Application Type

| **Application** | **Recommended Reading** |
|----------------|-------------------------|
| **Function Approximation** | Basic Usage → Training Methods |
| **Regression Analysis** | Builder Patterns → Validation |
| **Control Systems** | Membership Functions → Advanced Usage |
| **Time Series Forecasting** | Model Architecture → Performance Tuning |

## 🎯 Learning Path

### 👶 Beginner Path (Essential Knowledge)

1. **[Introduction](introduction.md)** - Learn what ANFIS is and how it works
2. **[Basic Usage](basic-usage.md)** - Build and train your first models
3. **[Builder Patterns](builder-patterns.md)** - Use ANFISRegressor for rapid prototyping
4. **[Visualization](visualization.md)** - Understand your models visually

**Time Investment**: ~2 hours
**Outcome**: Can build and evaluate basic ANFIS models

### 🎓 Intermediate Path (Practical Proficiency)

1. **[Membership Functions](membership-functions.md)** - Choose the right MF types
2. **[Training Methods](training-methods.md)** - Understand hybrid vs. backprop learning
3. **[Validation](validation.md)** - Properly evaluate model performance
4. **[Model Persistence](model-persistence.md)** - Save and deploy models
5. **[Performance Tuning](performance-tuning.md)** - Optimize for speed and accuracy

**Time Investment**: ~4 hours
**Outcome**: Can design, tune, and deploy production-ready models

### 🚀 Advanced Path (Expert Level)

1. **[Model Architecture](model-architecture.md)** - Deep understanding of ANFIS internals
2. **[Advanced Usage](advanced-usage.md)** - Manual model construction and fine-tuning
3. **[Custom Membership Functions](custom-membership-functions.md)** - Extend the library
4. **[Configuration Management](configuration-management.md)** - Manage complex workflows
5. **[Integration](integration.md)** - Combine with other ML tools

**Time Investment**: ~6 hours
**Outcome**: Can extend the library and handle any ANFIS use case

## 🛠️ Common Workflows

### Quick Prototyping
```python
from anfis_toolbox import ANFISRegressor

# Scikit-style estimator for rapid experiments
model = ANFISRegressor(n_mfs=3, random_state=42)
model.fit(X, y)
metrics = model.evaluate(X, y)
```
📖 **See**: [Basic Usage](basic-usage.md), [Builder Patterns](builder-patterns.md)

### Production Pipeline
```python
from anfis_toolbox import ANFISBuilder, ANFISValidator, ANFISConfig

# Structured model building
config = ANFISConfig(n_epochs=100, learning_rate=0.01)
builder = ANFISBuilder().add_input('x1', 'gaussian', 3)
model = builder.build()

# Robust validation
validator = ANFISValidator(model)
results = validator.cross_validate(X, y, cv=5)

# Model deployment
model.save('production_model.pkl')
```
📖 **See**: [Advanced Usage](advanced-usage.md), [Validation](validation.md), [Model Persistence](model-persistence.md)

### Research & Analysis
```python
from anfis_toolbox import ANFISVisualizer, ANFISMetrics

# Detailed analysis
viz = ANFISVisualizer(model)
viz.plot_membership_functions()
viz.plot_surface_3d(X, y)  # For 2D inputs

metrics = ANFISMetrics(model)
detailed_results = metrics.comprehensive_evaluation(X, y)
```
📖 **See**: [Visualization](visualization.md), [Validation](validation.md)

## 📚 Reference Materials

### Quick Reference Cards
- **[MF Types Cheat Sheet](../api/membership-functions.md)** - All 6 membership function types
- **[Training Parameters](../api/training.md)** - Complete parameter reference
- **[Metrics Guide](../api/validation.md)** - All evaluation metrics explained

### Theoretical Background
- **[ANFIS Paper](../theory/original-paper.md)** - Jang (1993) original paper summary
- **[Mathematical Foundations](../theory/mathematics.md)** - Detailed mathematical formulation
- **[Comparison with Other Methods](../theory/comparisons.md)** - ANFIS vs. neural networks, fuzzy systems

## 🤝 Getting Help

### Within Each Page
- **💡 Tips**: Practical advice and best practices
- **⚠️ Warnings**: Common pitfalls to avoid
- **🔍 Examples**: Concrete code examples
- **📖 Related**: Links to relevant sections

### External Resources
- **[API Reference](../api/overview.md)** - Complete function documentation
- **[Examples Gallery](../examples/basic.md)** - Real-world use cases
- **[FAQ](../development/faq.md)** - Frequently asked questions
- **[GitHub Issues](https://github.com/dcruzf/anfis-toolbox/issues)** - Report problems or ask questions

---

## Where to Start?

**New to ANFIS?** → Begin with Introduction
**Need to solve a problem quickly?** → Jump to Basic Usage
**Want to understand everything?** → Follow the Beginner Path

Ready to become an ANFIS expert? Let's dive in! 🚀
