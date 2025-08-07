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


def main():
    """Run all examples."""
    print("ANFIS Toolbox Examples")
    print("=" * 60)

    # Run examples
    try:
        model1 = example_1d_function()
        model2 = example_2d_function()
        model3 = example_parameter_inspection()

        print("\\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
