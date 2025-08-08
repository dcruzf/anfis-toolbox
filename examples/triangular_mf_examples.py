#!/usr/bin/env python3
"""Example usage of TriangularMF with ANFIS.

This script demonstrates how to use the new TriangularMF membership function
with the ANFIS model and compares it with GaussianMF.
"""

import numpy as np
from anfis_toolbox import ANFIS, GaussianMF, TriangularMF, enable_training_logs


def example_triangular_membership():
    """Demonstrate triangular membership function properties."""
    print("=" * 60)
    print("Example: Triangular Membership Function Properties")
    print("=" * 60)

    # Create different triangular membership functions
    tri_narrow = TriangularMF(-1.0, 0.0, 1.0)    # Narrow triangle
    tri_wide = TriangularMF(-2.0, 0.0, 2.0)      # Wide triangle
    tri_asymmetric = TriangularMF(-1.5, 0.5, 2.0)  # Asymmetric triangle

    # Compare with Gaussian
    gaussian = GaussianMF(mean=0.0, sigma=0.5)

    # Test input range
    x = np.linspace(-3, 3, 100)

    # Compute membership values
    y_narrow = tri_narrow.forward(x)
    y_wide = tri_wide.forward(x)
    y_asymmetric = tri_asymmetric.forward(x)
    y_gaussian = gaussian.forward(x)

    print(f"Narrow Triangle: Peak at 0.0, support [-1.0, 1.0]")
    print(f"Wide Triangle: Peak at 0.0, support [-2.0, 2.0]")
    print(f"Asymmetric Triangle: Peak at 0.5, support [-1.5, 2.0]")
    print(f"Gaussian: Mean at 0.0, sigma=0.5")

    # Show some specific values
    test_points = [-1.0, -0.5, 0.0, 0.5, 1.0]
    print(f"\\nMembership values at test points {test_points}:")
    print(f"{'Point':<8} {'Narrow':<8} {'Wide':<8} {'Asymm.':<8} {'Gaussian':<8}")
    print("-" * 45)

    for point in test_points:
        idx = np.argmin(np.abs(x - point))
        print(f"{point:<8.1f} {y_narrow[idx]:<8.3f} {y_wide[idx]:<8.3f} "
              f"{y_asymmetric[idx]:<8.3f} {y_gaussian[idx]:<8.3f}")


def example_triangular_anfis():
    """Example using TriangularMF with ANFIS for function approximation."""
    print("\\n" + "=" * 60)
    print("Example: ANFIS with Triangular Membership Functions")
    print("=" * 60)

    # Create a test function: y = x1 * sin(x2) + 0.5 * x1 * x2
    def target_function(x1, x2):
        return x1 * np.sin(x2) + 0.5 * x1 * x2

    # Generate training data
    np.random.seed(42)
    n_samples = 150
    x1_train = np.random.uniform(-2, 2, n_samples)
    x2_train = np.random.uniform(-1.5, 1.5, n_samples)
    x_train = np.column_stack([x1_train, x2_train])
    y_train = target_function(x1_train, x2_train).reshape(-1, 1)

    print(f"Training data: {n_samples} samples")
    print(f"Input ranges: x1 ∈ [-2, 2], x2 ∈ [-1.5, 1.5]")

    # Define triangular membership functions
    # For x1: three overlapping triangles covering the range
    input_mfs_tri = {
        'x1': [
            TriangularMF(-3.0, -1.0, 1.0),   # Left
            TriangularMF(-1.0, 0.0, 2.0),    # Center
            TriangularMF(-1.0, 1.0, 3.0)     # Right
        ],
        'x2': [
            TriangularMF(-2.0, -0.75, 0.5),  # Left
            TriangularMF(-0.5, 0.0, 1.0),    # Center
            TriangularMF(-0.5, 0.75, 2.0)    # Right
        ]
    }

    # Create ANFIS model with triangular membership functions
    model_tri = ANFIS(input_mfs_tri)
    print(f"\\nTriangular ANFIS: {model_tri.n_inputs} inputs, {model_tri.n_rules} rules")

    # Train the model
    enable_training_logs()
    print("\\nTraining with triangular membership functions...")
    losses_tri = model_tri.fit_hybrid(x_train, y_train, epochs=50, learning_rate=0.01, verbose=True)

    # Create test data
    x1_test = np.linspace(-2, 2, 20)
    x2_test = np.linspace(-1.5, 1.5, 20)
    X1, X2 = np.meshgrid(x1_test, x2_test)
    x_test = np.column_stack([X1.ravel(), X2.ravel()])
    y_test_true = target_function(x_test[:, 0], x_test[:, 1]).reshape(-1, 1)

    # Make predictions
    y_pred_tri = model_tri.predict(x_test)
    rmse_tri = np.sqrt(np.mean((y_pred_tri - y_test_true) ** 2))

    print(f"\\nTriangular ANFIS Results:")
    print(f"  - Final training loss: {losses_tri[-1]:.6f}")
    print(f"  - Test RMSE: {rmse_tri:.6f}")
    print(f"  - Loss reduction: {losses_tri[0]:.6f} -> {losses_tri[-1]:.6f}")

    return model_tri, losses_tri, rmse_tri


def example_comparison_triangular_vs_gaussian():
    """Compare triangular vs Gaussian membership functions in ANFIS."""
    print("\\n" + "=" * 60)
    print("Example: Triangular vs Gaussian Membership Functions")
    print("=" * 60)

    # Simple 1D function for clear comparison
    def target_function(x):
        return 0.5 * x**2 + 0.3 * np.sin(4 * x)

    # Generate training data
    np.random.seed(123)
    x_train = np.random.uniform(-2, 2, 100).reshape(-1, 1)
    y_train = target_function(x_train.flatten()).reshape(-1, 1)

    print(f"Target function: y = 0.5*x² + 0.3*sin(4x)")
    print(f"Training samples: {len(x_train)}")

    # Triangular membership functions
    input_mfs_tri = {
        'x': [
            TriangularMF(-3.0, -1.5, 0.0),
            TriangularMF(-1.0, -0.5, 1.0),
            TriangularMF(-0.5, 0.0, 1.5),
            TriangularMF(0.0, 1.5, 3.0)
        ]
    }

    # Gaussian membership functions (similar coverage)
    input_mfs_gauss = {
        'x': [
            GaussianMF(mean=-1.5, sigma=0.8),
            GaussianMF(mean=-0.5, sigma=0.8),
            GaussianMF(mean=0.0, sigma=0.8),
            GaussianMF(mean=1.5, sigma=0.8)
        ]
    }

    # Train both models
    print("\\nTraining Triangular ANFIS...")
    model_tri = ANFIS(input_mfs_tri)
    losses_tri = model_tri.fit(x_train, y_train, epochs=60, learning_rate=0.01, verbose=False)

    print("Training Gaussian ANFIS...")
    model_gauss = ANFIS(input_mfs_gauss)
    losses_gauss = model_gauss.fit(x_train, y_train, epochs=60, learning_rate=0.01, verbose=False)

    # Test both models
    x_test = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
    y_test_true = target_function(x_test.flatten()).reshape(-1, 1)

    y_pred_tri = model_tri.predict(x_test)
    y_pred_gauss = model_gauss.predict(x_test)

    rmse_tri = np.sqrt(np.mean((y_pred_tri - y_test_true) ** 2))
    rmse_gauss = np.sqrt(np.mean((y_pred_gauss - y_test_true) ** 2))

    print(f"\\n{'Model':<20} {'Final Loss':<12} {'Test RMSE':<12} {'Convergence':<12}")
    print("-" * 58)
    print(f"{'Triangular':<20} {losses_tri[-1]:<12.6f} {rmse_tri:<12.6f} "
          f"{losses_tri[0]/losses_tri[-1]:<12.2f}x")
    print(f"{'Gaussian':<20} {losses_gauss[-1]:<12.6f} {rmse_gauss:<12.6f} "
          f"{losses_gauss[0]/losses_gauss[-1]:<12.2f}x")

    if rmse_tri < rmse_gauss:
        print(f"\\n✅ Triangular MF performed better (RMSE: {rmse_tri:.6f} vs {rmse_gauss:.6f})")
    else:
        print(f"\\n✅ Gaussian MF performed better (RMSE: {rmse_gauss:.6f} vs {rmse_tri:.6f})")

    print("\\nCharacteristics comparison:")
    print("  Triangular MF:")
    print("    ✓ Computationally efficient (piecewise linear)")
    print("    ✓ Good linguistic interpretability")
    print("    ✓ Sharp boundaries")
    print("    ✗ Non-smooth derivatives (may affect optimization)")

    print("  Gaussian MF:")
    print("    ✓ Smooth derivatives (good for gradient-based optimization)")
    print("    ✓ Biologically inspired")
    print("    ✓ Good generalization properties")
    print("    ✗ More computationally expensive")

    return model_tri, model_gauss, rmse_tri, rmse_gauss


def main():
    """Run all triangular membership function examples."""
    print("TriangularMF Examples")
    print("=" * 60)

    try:
        # Basic properties demonstration
        example_triangular_membership()

        # ANFIS with triangular MF
        model_tri, losses_tri, rmse_tri = example_triangular_anfis()

        # Comparison with Gaussian MF
        model_tri_comp, model_gauss_comp, rmse_tri_comp, rmse_gauss_comp = example_comparison_triangular_vs_gaussian()

        print("\\n" + "=" * 60)
        print("All TriangularMF examples completed successfully!")
        print("=" * 60)
        print("\\nSummary:")
        print(f"  - TriangularMF successfully integrated with ANFIS")
        print(f"  - Works with both fit() and fit_hybrid() methods")
        print(f"  - Provides alternative to GaussianMF with different characteristics")
        print(f"  - Test RMSE achieved: {rmse_tri:.6f} (triangular) vs comparison results")

    except Exception as e:
        print(f"Error running triangular MF examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
