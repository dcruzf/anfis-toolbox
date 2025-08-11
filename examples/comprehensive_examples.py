"""Comprehensive examples demonstrating easy ANFIS usage."""

import numpy as np
from anfis_toolbox import ANFIS
from anfis_toolbox.builders import ANFISBuilder, QuickANFIS
from anfis_toolbox.validation import quick_evaluate
# from anfis_toolbox.visualization import quick_plot_results, quick_plot_training


def example_basic_usage():
    """Example 1: Most basic ANFIS usage with automatic setup."""
    print("🚀 Example 1: Basic ANFIS Usage")
    print("=" * 50)

    # Generate sample data: y = sin(x1) + cos(x2)
    np.random.seed(42)
    X = np.random.uniform(-np.pi, np.pi, (200, 2))
    y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])

    # Create ANFIS model automatically
    model = QuickANFIS.for_regression(X, n_mfs=3, mf_type='gaussian')

    print(f"📊 Created ANFIS with {model.n_inputs} inputs and {model.n_rules} rules")

    # Train model
    print("🏋️ Training model...")
    losses = model.fit_hybrid(X, y, epochs=50, learning_rate=0.01, verbose=True)

    # Evaluate
    metrics = quick_evaluate(model, X, y)

    print(f"✅ Training completed! Final R² = {metrics['r2']:.4f}")
    return model, X, y, losses


def example_custom_builder():
    """Example 2: Using the builder pattern for custom configuration."""
    print("\n🔧 Example 2: Custom Builder Pattern")
    print("=" * 50)

    # Generate 1D function data
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = X**3 - 2*X**2 + X + np.random.normal(0, 0.1, X.shape)

    # Build custom ANFIS model
    model = (ANFISBuilder()
             .add_input('x', -3, 3, n_mfs=5, mf_type='triangular', overlap=0.3)
             .build())

    print(f"🏗️ Built custom ANFIS with {model.n_rules} rules using triangular MFs")

    # Train and evaluate
    losses = model.fit_hybrid(X, y, epochs=100, learning_rate=0.02)
    metrics = quick_evaluate(model, X, y)

    return model, X, y, losses


def example_function_approximation():
    """Example 3: Function approximation with visualization."""
    print("\n📈 Example 3: Function Approximation")
    print("=" * 50)

    # Define a complex 1D function
    def complex_function(x):
        return np.sin(2*x) * np.exp(-x/3) + 0.1*x

    # Generate data
    X = np.linspace(0, 6, 150).reshape(-1, 1)
    y = complex_function(X)

    # Create model using quick setup
    model = QuickANFIS.for_function_approximation([(-1, 7)], n_mfs=7)

    print("🎯 Approximating: f(x) = sin(2x) * exp(-x/3) + 0.1x")

    # Train
    losses = model.fit_hybrid(X, y, epochs=80, learning_rate=0.01, verbose=True)

    # Evaluate
    metrics = quick_evaluate(model, X, y)

    # Uncomment for visualization (requires matplotlib)
    # quick_plot_results(X, y, model)
    # quick_plot_training(losses)

    return model, X, y, losses


def example_multi_input_system():
    """Example 4: Multi-input system with cross-validation."""
    print("\n🎛️ Example 4: Multi-Input System")
    print("=" * 50)

    # Generate multi-dimensional data: complex surface
    np.random.seed(123)
    n_samples = 300
    X1 = np.random.uniform(-2, 2, n_samples)
    X2 = np.random.uniform(-2, 2, n_samples)
    X3 = np.random.uniform(-1, 1, n_samples)

    X = np.column_stack([X1, X2, X3])
    y = (X1**2 + X2*X3 + np.sin(X1*X2) + 0.1*np.random.randn(n_samples)).reshape(-1, 1)

    # Create model with custom configuration
    builder = ANFISBuilder()
    model = (builder
             .add_input('x1', -2.5, 2.5, n_mfs=4, mf_type='gaussian')
             .add_input('x2', -2.5, 2.5, n_mfs=4, mf_type='gaussian')
             .add_input('x3', -1.5, 1.5, n_mfs=3, mf_type='gaussian')
             .build())

    print(f"🏗️ Created {model.n_inputs}-input ANFIS with {model.n_rules} rules")

    # Train
    losses = model.fit_hybrid(X, y, epochs=60, learning_rate=0.015)

    # Cross-validation (uncomment if sklearn available)
    # from anfis_toolbox.validation import ANFISValidator
    # validator = ANFISValidator(model)
    # cv_results = validator.cross_validate(X, y, cv=3, epochs=30)
    # print(f"📊 Cross-validation R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")

    # Simple evaluation
    metrics = quick_evaluate(model, X, y)

    return model, X, y, losses


def example_comparison_study():
    """Example 5: Compare different ANFIS configurations."""
    print("\n⚖️ Example 5: Configuration Comparison")
    print("=" * 50)

    # Generate test data
    np.random.seed(456)
    X = np.random.uniform(-2, 2, (200, 2))
    y = X[:, 0:1]**2 + X[:, 1:2]**2 + 0.1*np.random.randn(200, 1)

    configurations = [
        {'name': 'Few Rules (2 MFs/input)', 'n_mfs': 2, 'mf_type': 'gaussian'},
        {'name': 'Medium Rules (3 MFs/input)', 'n_mfs': 3, 'mf_type': 'gaussian'},
        {'name': 'Many Rules (4 MFs/input)', 'n_mfs': 4, 'mf_type': 'gaussian'},
        {'name': 'Triangular MFs (3/input)', 'n_mfs': 3, 'mf_type': 'triangular'},
    ]

    results = []

    for config in configurations:
        print(f"\n🔍 Testing: {config['name']}")

        # Create model
        model = QuickANFIS.for_regression(X, n_mfs=config['n_mfs'],
                                        mf_type=config['mf_type'])

        # Train
        losses = model.fit_hybrid(X, y, epochs=50, learning_rate=0.01, verbose=False)

        # Evaluate
        metrics = quick_evaluate(model, X, y, print_results=False)

        result = {
            'name': config['name'],
            'n_rules': model.n_rules,
            'r2': metrics['r2'],
            'mse': metrics['mse'],
            'final_loss': losses[-1]
        }
        results.append(result)

        print(f"   Rules: {result['n_rules']}, R²: {result['r2']:.4f}, MSE: {result['mse']:.6f}")

    # Find best configuration
    best_config = max(results, key=lambda x: x['r2'])
    print(f"\n🏆 Best configuration: {best_config['name']} (R² = {best_config['r2']:.4f})")

    return results


def example_step_by_step_tutorial():
    """Example 6: Step-by-step tutorial for beginners."""
    print("\n📚 Example 6: Step-by-Step Tutorial")
    print("=" * 50)

    print("Step 1: Generate or load your data")
    # Your data should be in numpy arrays: X (inputs), y (targets)
    X = np.random.uniform(-1, 1, (100, 2))  # 100 samples, 2 inputs
    y = X[:, 0:1] + X[:, 1:2]  # Simple sum function
    print(f"   Data shape: X={X.shape}, y={y.shape}")

    print("\nStep 2: Choose your model configuration")
    # Option A: Automatic (recommended for beginners)
    model_auto = QuickANFIS.for_regression(X)
    print(f"   Automatic config: {model_auto.n_rules} rules")

    # Option B: Custom configuration
    model_custom = (ANFISBuilder()
                   .add_input('x1', -1.2, 1.2, n_mfs=3)
                   .add_input('x2', -1.2, 1.2, n_mfs=3)
                   .build())
    print(f"   Custom config: {model_custom.n_rules} rules")

    print("\nStep 3: Train your model")
    # Hybrid method (recommended) - combines least squares + backpropagation
    losses = model_auto.fit_hybrid(X, y, epochs=50, learning_rate=0.01)
    print(f"   Training completed. Final loss: {losses[-1]:.6f}")

    print("\nStep 4: Evaluate your model")
    metrics = quick_evaluate(model_auto, X, y, print_results=False)
    print(f"   R-squared: {metrics['r2']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")

    print("\nStep 5: Make predictions on new data")
    X_new = np.array([[0.5, -0.3], [0.1, 0.8]])
    predictions = model_auto.predict(X_new)
    print(f"   Predictions: {predictions.flatten()}")

    print("\n✅ Tutorial completed! You now know the basics of using ANFIS.")

    return model_auto, X, y, losses


def run_all_examples():
    """Run all examples to demonstrate the toolbox capabilities."""
    print("🎉 ANFIS Toolbox - Complete Examples")
    print("=" * 70)

    try:
        # Example 1: Basic usage
        model1, X1, y1, losses1 = example_basic_usage()

        # Example 2: Custom builder
        model2, X2, y2, losses2 = example_custom_builder()

        # Example 3: Function approximation
        model3, X3, y3, losses3 = example_function_approximation()

        # Example 4: Multi-input system
        model4, X4, y4, losses4 = example_multi_input_system()

        # Example 5: Configuration comparison
        comparison_results = example_comparison_study()

        # Example 6: Tutorial
        tutorial_model, tutorial_X, tutorial_y, tutorial_losses = example_step_by_step_tutorial()

        print("\n🎊 All examples completed successfully!")
        print("📖 Check the documentation for more advanced features.")

        return {
            'basic': (model1, X1, y1, losses1),
            'custom': (model2, X2, y2, losses2),
            'function_approx': (model3, X3, y3, losses3),
            'multi_input': (model4, X4, y4, losses4),
            'comparison': comparison_results,
            'tutorial': (tutorial_model, tutorial_X, tutorial_y, tutorial_losses)
        }

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("🔧 Make sure all dependencies are installed and try again.")
        return None


if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
