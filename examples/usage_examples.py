#!/usr/bin/env python3
"""Example usage of the ANFIS toolbox.

This script demonstrates how to use the ANFIS class to create and train
a fuzzy neural network for function approximation.
"""

import numpy as np
from anfis_toolbox import ANFIS, GaussianMF, enable_training_logs


def example_1d_function():
    """Example: Approximate a 1D nonlinear function."""
    print("=" * 60)
    print("Example 1: 1D Function Approximation")
    print("=" * 60)

    # Define the target function: y = sin(x) + 0.1*cos(5*x)
    def target_function(x):
        return np.sin(x) + 0.1 * np.cos(5 * x)

    # Create training data
    x_train = np.linspace(-2 * np.pi, 2 * np.pi, 100).reshape(-1, 1)
    y_train = target_function(x_train)

    # Define membership functions for input variable
    # Using 5 Gaussian membership functions across the input range
    input_mfs = {
        'x': [
            GaussianMF(mean=-2*np.pi, sigma=1.5),
            GaussianMF(mean=-np.pi, sigma=1.5),
            GaussianMF(mean=0.0, sigma=1.5),
            GaussianMF(mean=np.pi, sigma=1.5),
            GaussianMF(mean=2*np.pi, sigma=1.5)
        ]
    }

    # Create ANFIS model
    model = ANFIS(input_mfs)
    print(f"Created ANFIS model with {model.n_inputs} input(s) and {model.n_rules} rules")

    # Enable training logs for this example
    enable_training_logs()

    # Train the model
    print("Training the model...")
    losses = model.fit(x_train, y_train, epochs=100, learning_rate=0.01, verbose=True)

    # Make predictions
    x_test = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
    y_test_true = target_function(x_test)
    y_test_pred = model.predict(x_test)

    # Calculate error metrics
    mse = np.mean((y_test_pred - y_test_true) ** 2)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Final training loss: {losses[-1]:.6f}")
    print(f"Training loss reduction: {losses[0]:.6f} -> {losses[-1]:.6f}")

    return model


def example_2d_function():
    """Example: Approximate a 2D nonlinear function."""
    print("=" * 60)
    print("Example 2: 2D Function Approximation")
    print("=" * 60)

    # Define the target function: z = sin(sqrt(x^2 + y^2))
    def target_function(x, y):
        r = np.sqrt(x**2 + y**2)
        return np.sin(r) / (r + 0.1)  # Add small epsilon to avoid division by zero

    # Create training data
    np.random.seed(42)
    n_samples = 200
    x1_train = np.random.uniform(-3, 3, n_samples)
    x2_train = np.random.uniform(-3, 3, n_samples)
    x_train = np.column_stack([x1_train, x2_train])
    y_train = target_function(x1_train, x2_train).reshape(-1, 1)

    # Define membership functions for both input variables
    input_mfs = {
        'x1': [
            GaussianMF(mean=-2.0, sigma=1.5),
            GaussianMF(mean=0.0, sigma=1.5),
            GaussianMF(mean=2.0, sigma=1.5)
        ],
        'x2': [
            GaussianMF(mean=-2.0, sigma=1.5),
            GaussianMF(mean=0.0, sigma=1.5),
            GaussianMF(mean=2.0, sigma=1.5)
        ]
    }

    # Create ANFIS model
    model = ANFIS(input_mfs)
    print(f"Created ANFIS model with {model.n_inputs} input(s) and {model.n_rules} rules")

    # Enable training logs for this example
    enable_training_logs()

    # Train the model
    print("Training the model...")
    losses = model.fit(x_train, y_train, epochs=150, learning_rate=0.01, verbose=True)

    # Create test data (subset for evaluation)
    np.random.seed(123)
    n_test = 100
    x1_test = np.random.uniform(-3, 3, n_test)
    x2_test = np.random.uniform(-3, 3, n_test)
    x_test = np.column_stack([x1_test, x2_test])

    # Make predictions
    y_test_true = target_function(x_test[:, 0], x_test[:, 1]).reshape(-1, 1)
    y_test_pred = model.predict(x_test)

    # Calculate error metrics
    mse = np.mean((y_test_pred - y_test_true) ** 2)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Final training loss: {losses[-1]:.6f}")
    print(f"Training loss reduction: {losses[0]:.6f} -> {losses[-1]:.6f}")

    return model


def example_parameter_inspection():
    """Example: Inspect and modify ANFIS parameters."""
    print("=" * 60)
    print("Example 3: Parameter Inspection and Modification")
    print("=" * 60)

    # Create a simple ANFIS model
    input_mfs = {
        'x': [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    }
    model = ANFIS(input_mfs)

    # Get initial parameters
    print("Initial parameters:")
    params = model.get_parameters()

    print("\\nMembership function parameters:")
    for input_name, mf_params in params['membership'].items():
        print(f"  {input_name}:")
        for i, mf_param in enumerate(mf_params):
            print(f"    MF {i+1}: mean={mf_param['mean']:.3f}, sigma={mf_param['sigma']:.3f}")

    print(f"\\nConsequent parameters shape: {params['consequent'].shape}")
    print(f"Consequent parameters:\\n{params['consequent']}")

    # Train on some data to see parameter changes
    x_train = np.array([[-2], [-1], [0], [1], [2]], dtype=float)
    y_train = x_train ** 2  # Simple quadratic function

    print("\\nTraining on quadratic function...")
    losses = model.fit(x_train, y_train, epochs=50, learning_rate=0.1, verbose=False)

    # Get updated parameters
    print("\\nParameters after training:")
    params_after = model.get_parameters()

    print("\\nMembership function parameters:")
    for input_name, mf_params in params_after['membership'].items():
        print(f"  {input_name}:")
        for i, mf_param in enumerate(mf_params):
            print(f"    MF {i+1}: mean={mf_param['mean']:.3f}, sigma={mf_param['sigma']:.3f}")

    print(f"\\nConsequent parameters:\\n{params_after['consequent']}")

    # Demonstrate parameter modification
    print("\\n" + "="*40)
    print("Modifying parameters manually...")

    # Modify membership function parameters
    modified_params = params_after.copy()
    modified_params['membership']['x'][0]['mean'] = -2.0
    modified_params['membership']['x'][1]['mean'] = 2.0
    modified_params['consequent'] = np.ones_like(modified_params['consequent'])

    # Set the modified parameters
    model.set_parameters(modified_params)

    # Verify the changes
    current_params = model.get_parameters()
    print("\\nModified parameters:")
    for input_name, mf_params in current_params['membership'].items():
        print(f"  {input_name}:")
        for i, mf_param in enumerate(mf_params):
            print(f"    MF {i+1}: mean={mf_param['mean']:.3f}, sigma={mf_param['sigma']:.3f}")

    print(f"\\nConsequent parameters:\\n{current_params['consequent']}")

    return model


def example_hybrid_vs_backprop():
    """Example: Compare original hybrid algorithm vs pure backpropagation."""
    print("=" * 60)
    print("Example 4: Hybrid Algorithm vs Pure Backpropagation")
    print("=" * 60)

    # Create a simple 2D test function
    def target_function(x1, x2):
        return np.sin(x1) * np.cos(x2) + 0.5 * x1 * x2

    # Generate training data
    np.random.seed(42)
    n_samples = 100
    x1_train = np.random.uniform(-2, 2, n_samples)
    x2_train = np.random.uniform(-2, 2, n_samples)
    x_train = np.column_stack([x1_train, x2_train])
    y_train = target_function(x1_train, x2_train).reshape(-1, 1)

    # Test data
    x1_test = np.linspace(-2, 2, 20)
    x2_test = np.linspace(-2, 2, 20)
    X1, X2 = np.meshgrid(x1_test, x2_test)
    x_test = np.column_stack([X1.ravel(), X2.ravel()])
    y_test = target_function(x_test[:, 0], x_test[:, 1]).reshape(-1, 1)

    # Define identical membership functions for both models
    input_mfs = {
        'x1': [GaussianMF(mean=-1.0, sigma=0.8), GaussianMF(mean=0.0, sigma=0.8), GaussianMF(mean=1.0, sigma=0.8)],
        'x2': [GaussianMF(mean=-1.0, sigma=0.8), GaussianMF(mean=0.0, sigma=0.8), GaussianMF(mean=1.0, sigma=0.8)]
    }

    print("\\n" + "="*40)
    print("Testing Original Hybrid Algorithm (Jang 1993)")
    print("="*40)

    # Train with hybrid algorithm
    model_hybrid = ANFIS(input_mfs)
    import time

    enable_training_logs()
    start_time = time.time()
    losses_hybrid = model_hybrid.fit_hybrid(x_train, y_train, epochs=50, learning_rate=0.01, verbose=True)
    hybrid_time = time.time() - start_time

    # Test hybrid model
    y_pred_hybrid = model_hybrid.predict(x_test)
    rmse_hybrid = np.sqrt(np.mean((y_pred_hybrid - y_test) ** 2))

    print("\\n" + "="*40)
    print("Testing Pure Backpropagation (Modern)")
    print("="*40)

    # Reset membership functions for fair comparison
    input_mfs_bp = {
        'x1': [GaussianMF(mean=-1.0, sigma=0.8), GaussianMF(mean=0.0, sigma=0.8), GaussianMF(mean=1.0, sigma=0.8)],
        'x2': [GaussianMF(mean=-1.0, sigma=0.8), GaussianMF(mean=0.0, sigma=0.8), GaussianMF(mean=1.0, sigma=0.8)]
    }

    # Train with pure backpropagation
    model_backprop = ANFIS(input_mfs_bp)
    start_time = time.time()
    losses_backprop = model_backprop.fit(x_train, y_train, epochs=50, learning_rate=0.01, verbose=True)
    backprop_time = time.time() - start_time

    # Test backpropagation model
    y_pred_backprop = model_backprop.predict(x_test)
    rmse_backprop = np.sqrt(np.mean((y_pred_backprop - y_test) ** 2))

    print("\\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Hybrid Algorithm (Original ANFIS):")
    print(f"  - Final Loss: {losses_hybrid[-1]:.6f}")
    print(f"  - Test RMSE: {rmse_hybrid:.6f}")
    print(f"  - Training Time: {hybrid_time:.3f} seconds")
    print(f"  - Loss Reduction: {losses_hybrid[0]:.6f} -> {losses_hybrid[-1]:.6f}")

    print(f"\\nPure Backpropagation (Modern):")
    print(f"  - Final Loss: {losses_backprop[-1]:.6f}")
    print(f"  - Test RMSE: {rmse_backprop:.6f}")
    print(f"  - Training Time: {backprop_time:.3f} seconds")
    print(f"  - Loss Reduction: {losses_backprop[0]:.6f} -> {losses_backprop[-1]:.6f}")

    print(f"\\nComparison:")
    if rmse_hybrid < rmse_backprop:
        print(f"  ✅ Hybrid algorithm achieved better accuracy ({rmse_hybrid:.6f} vs {rmse_backprop:.6f})")
    else:
        print(f"  ✅ Backpropagation achieved better accuracy ({rmse_backprop:.6f} vs {rmse_hybrid:.6f})")

    if hybrid_time < backprop_time:
        print(f"  ⚡ Hybrid algorithm was faster ({hybrid_time:.3f}s vs {backprop_time:.3f}s)")
    else:
        print(f"  ⚡ Backpropagation was faster ({backprop_time:.3f}s vs {hybrid_time:.3f}s)")

    convergence_hybrid = abs(losses_hybrid[-1] - losses_hybrid[-10]) if len(losses_hybrid) >= 10 else abs(losses_hybrid[-1] - losses_hybrid[0])
    convergence_backprop = abs(losses_backprop[-1] - losses_backprop[-10]) if len(losses_backprop) >= 10 else abs(losses_backprop[-1] - losses_backprop[0])

    if convergence_hybrid < convergence_backprop:
        print(f"  🎯 Hybrid algorithm converged better (change: {convergence_hybrid:.6f} vs {convergence_backprop:.6f})")
    else:
        print(f"  🎯 Backpropagation converged better (change: {convergence_backprop:.6f} vs {convergence_hybrid:.6f})")

    return model_hybrid, model_backprop, losses_hybrid, losses_backprop


def main():
    """Run all examples."""
    print("ANFIS Toolbox Examples")
    print("=" * 60)

    # Run examples
    try:
        model1 = example_1d_function()
        model2 = example_2d_function()
        model3 = example_parameter_inspection()
        model_hybrid, model_backprop, losses_hybrid, losses_backprop = example_hybrid_vs_backprop()

        print("\\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
