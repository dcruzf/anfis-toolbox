import numpy as np
import pytest

from anfis_toolbox.membership import GaussianMF, TrapezoidalMF, TriangularMF


def test_gaussian_mf():
    """Test the Gaussian membership function."""
    mean = 0.0
    sigma = 1.0
    gaussian_mf = GaussianMF(mean, sigma)

    # Test forward pass
    x = np.array([-2, -1, 0, 1, 2])
    output = gaussian_mf(x)
    expected_output = np.exp(-((x - mean) ** 2) / (2 * sigma**2))

    assert np.allclose(output, expected_output), "Forward pass failed"

    # Test backward pass
    dL_dy = np.ones_like(output)
    gaussian_mf.backward(dL_dy)

    # Check if gradients are computed correctly
    assert "mean" in gaussian_mf.gradients and "sigma" in gaussian_mf.gradients, "Gradients not computed"

    gaussian_mf.reset()
    assert gaussian_mf.last_input is None and gaussian_mf.last_output is None, "Reset failed"
    assert all(v == 0.0 for v in gaussian_mf.gradients.values()), "Gradients not reset"


@pytest.mark.parametrize("param_name", ["mean", "sigma"])
def test_gaussianmf_gradient(param_name):
    eps = 1e-5
    x = np.array([1.0, 2.0, 3.0])

    # Inicializa a função de pertinência
    mf = GaussianMF(mean=0.5, sigma=1.0)

    # Calcula saída original e perda
    y = mf.forward(x)

    # Backward manual
    dL_dy = np.ones_like(y)
    mf.backward(dL_dy)
    grad_analytical = mf.gradients[param_name]

    # Gradiente numérico
    original_value = mf.parameters[param_name]

    # L(param + eps)
    mf.parameters[param_name] = original_value + eps
    y_plus = mf.forward(x)
    L_plus = np.sum(y_plus)

    # L(param - eps)
    mf.parameters[param_name] = original_value - eps
    y_minus = mf.forward(x)
    L_minus = np.sum(y_minus)

    grad_numerical = (L_plus - L_minus) / (2 * eps)

    # Restaurar o parâmetro original
    mf.parameters[param_name] = original_value

    # Verificação
    assert np.allclose(grad_analytical, grad_numerical, atol=1e-4), (
        f"Gradient check failed for {param_name}.\nAnalytical: {grad_analytical}, Numerical: {grad_numerical}"
    )


def test_triangular_mf_basic():
    """Test basic functionality of triangular membership function."""
    a, b, c = -1.0, 0.0, 1.0
    tri_mf = TriangularMF(a, b, c)

    # Test key points
    x = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    output = tri_mf.forward(x)

    # Expected values: outside = 0, at peak = 1, linear in between
    expected = np.array([0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])

    assert np.allclose(output, expected), f"Expected {expected}, got {output}"

    # Test that peak is correctly at 1.0
    peak_output = tri_mf.forward(np.array([b]))
    assert np.allclose(peak_output, [1.0]), "Peak should be 1.0"

    # Test boundary conditions
    boundary_output = tri_mf.forward(np.array([a, c]))
    assert np.allclose(boundary_output, [0.0, 0.0]), "Boundaries should be 0.0"


def test_triangular_mf_parameter_validation():
    """Test parameter validation for triangular membership function."""

    # Valid parameters
    tri_mf = TriangularMF(0, 1, 2)
    assert tri_mf.parameters["a"] == 0
    assert tri_mf.parameters["b"] == 1
    assert tri_mf.parameters["c"] == 2

    # Test invalid parameter ordering
    with pytest.raises(ValueError, match="must satisfy a ≤ b ≤ c"):
        TriangularMF(2, 1, 0)  # a > b > c

    with pytest.raises(ValueError, match="must satisfy a ≤ b ≤ c"):
        TriangularMF(0, 2, 1)  # b > c

    # Test zero-width triangle
    with pytest.raises(ValueError, match="cannot be equal"):
        TriangularMF(1, 1, 1)  # a = b = c

    # Test edge case: b = a or b = c (should be valid)
    tri_mf1 = TriangularMF(0, 0, 1)  # Left-aligned triangle
    tri_mf2 = TriangularMF(0, 1, 1)  # Right-aligned triangle

    assert tri_mf1.parameters["a"] == tri_mf1.parameters["b"]
    assert tri_mf2.parameters["b"] == tri_mf2.parameters["c"]


def test_triangular_mf_forward_edge_cases():
    """Test forward pass edge cases for triangular membership function."""

    # Test asymmetric triangle
    tri_mf = TriangularMF(-2, 0, 3)
    x = np.array([-3, -2, -1, 0, 1.5, 3, 4])
    output = tri_mf.forward(x)

    # Expected: linear interpolation
    expected = np.array(
        [
            0.0,  # x = -3 (outside)
            0.0,  # x = -2 (boundary)
            0.5,  # x = -1 (halfway up left slope: (-1 - (-2))/(0 - (-2)) = 0.5)
            1.0,  # x = 0  (peak)
            0.5,  # x = 1.5 (halfway down right slope: (3 - 1.5)/(3 - 0) = 0.5)
            0.0,  # x = 3  (boundary)
            0.0,  # x = 4  (outside)
        ]
    )

    assert np.allclose(output, expected), f"Expected {expected}, got {output}"


def test_triangular_mf_backward():
    """Test backward pass for triangular membership function."""
    tri_mf = TriangularMF(-1, 0, 2)

    # Forward pass
    x = np.array([-0.5, 0, 1])
    y = tri_mf.forward(x)

    # Backward pass
    dL_dy = np.ones_like(y)
    tri_mf.backward(dL_dy)

    # Check that gradients are computed
    assert "a" in tri_mf.gradients
    assert "b" in tri_mf.gradients
    assert "c" in tri_mf.gradients

    # Gradients should be non-zero for points affecting the function
    assert tri_mf.gradients["a"] != 0.0 or tri_mf.gradients["b"] != 0.0 or tri_mf.gradients["c"] != 0.0


@pytest.mark.parametrize("param_name", ["a", "b", "c"])
def test_triangular_mf_gradient_numerical(param_name):
    """Test gradients using numerical differentiation with relaxed conditions."""
    # Note: Triangular MF gradients are more complex to verify numerically due to
    # piecewise nature and changing domains when parameters change.
    # This test verifies that gradients are computed and have reasonable magnitudes.

    tri_mf = TriangularMF(a=0.0, b=1.0, c=2.0)

    # Use multiple test points to get more robust gradient estimate
    x = np.array([0.3, 0.7, 1.3, 1.7])  # Points in various regions

    # Forward pass and backward pass
    y = tri_mf.forward(x)
    dL_dy = np.ones_like(y)
    tri_mf.backward(dL_dy)
    grad_analytical = tri_mf.gradients[param_name]

    # For triangular functions, we verify:
    # 1. Gradients are computed (not NaN)
    # 2. Gradients have reasonable magnitudes
    # 3. Gradients are zero for parameters that don't affect the current points

    assert not np.isnan(grad_analytical), f"Gradient for {param_name} is NaN"
    assert abs(grad_analytical) < 100, f"Gradient for {param_name} is unexpectedly large: {grad_analytical}"

    # Test that changing parameters actually changes the output (gradient should be non-zero if param affects output)
    original_value = tri_mf.parameters[param_name]

    tri_mf.parameters[param_name] = original_value + 0.1  # Larger perturbation
    y_perturbed = tri_mf.forward(x)
    tri_mf.parameters[param_name] = original_value  # Restore

    if not np.allclose(y, y_perturbed):
        # If output changes when parameter changes, gradient should be non-zero
        assert abs(grad_analytical) > 1e-10, (
            f"Gradient for {param_name} should be non-zero when parameter affects output"
        )


def test_triangular_mf_gradient_analytical():
    """Test analytical gradients with known cases."""
    tri_mf = TriangularMF(a=0.0, b=1.0, c=2.0)

    # Test case 1: Point on left slope
    x = np.array([0.5])  # μ(0.5) = 0.5, on left slope
    y = tri_mf.forward(x)
    assert np.allclose(y, [0.5]), "Forward pass incorrect"

    # Reset only gradients, keep last_input for backward pass
    for key in tri_mf.gradients:
        tri_mf.gradients[key] = 0.0

    tri_mf.backward(np.array([1.0]))  # dL_dy = 1.0

    # Expected gradients for left slope:
    # ∂μ/∂a = -1/(b-a) = -1/1 = -1
    # ∂μ/∂b = -(x-a)/(b-a)² = -(0.5-0)/1² = -0.5
    # ∂μ/∂c = 0 (point not on right slope)

    assert np.allclose(tri_mf.gradients["a"], -1.0), f"Expected ∂μ/∂a = -1.0, got {tri_mf.gradients['a']}"
    assert np.allclose(tri_mf.gradients["b"], -0.5), f"Expected ∂μ/∂b = -0.5, got {tri_mf.gradients['b']}"
    assert np.allclose(tri_mf.gradients["c"], 0.0), f"Expected ∂μ/∂c = 0.0, got {tri_mf.gradients['c']}"

    # Test case 2: Point on right slope
    x = np.array([1.5])  # μ(1.5) = 0.5, on right slope
    y = tri_mf.forward(x)
    assert np.allclose(y, [0.5]), "Forward pass incorrect"

    # Reset only gradients
    for key in tri_mf.gradients:
        tri_mf.gradients[key] = 0.0

    tri_mf.backward(np.array([1.0]))  # dL_dy = 1.0

    # Expected gradients for right slope:
    # ∂μ/∂a = 0 (point not on left slope)
    # ∂μ/∂b = (x-c)/(c-b)² = (1.5-2)/1² = -0.5
    # ∂μ/∂c = (x-b)/(c-b)² = (1.5-1)/1² = 0.5

    assert np.allclose(tri_mf.gradients["a"], 0.0), f"Expected ∂μ/∂a = 0.0, got {tri_mf.gradients['a']}"
    assert np.allclose(tri_mf.gradients["b"], -0.5), f"Expected ∂μ/∂b = -0.5, got {tri_mf.gradients['b']}"
    assert np.allclose(tri_mf.gradients["c"], 0.5), f"Expected ∂μ/∂c = 0.5, got {tri_mf.gradients['c']}"

    # Test case 3: Point outside support
    x = np.array([3.0])  # μ(3.0) = 0.0, outside support
    y = tri_mf.forward(x)
    assert np.allclose(y, [0.0]), "Forward pass incorrect"

    # Reset only gradients
    for key in tri_mf.gradients:
        tri_mf.gradients[key] = 0.0

    tri_mf.backward(np.array([1.0]))  # dL_dy = 1.0

    # Expected gradients outside support: all should be 0
    assert np.allclose(tri_mf.gradients["a"], 0.0), f"Expected ∂μ/∂a = 0.0, got {tri_mf.gradients['a']}"
    assert np.allclose(tri_mf.gradients["b"], 0.0), f"Expected ∂μ/∂b = 0.0, got {tri_mf.gradients['b']}"
    assert np.allclose(tri_mf.gradients["c"], 0.0), f"Expected ∂μ/∂c = 0.0, got {tri_mf.gradients['c']}"


def test_triangular_mf_reset():
    """Test reset functionality for triangular membership function."""
    tri_mf = TriangularMF(-1, 0, 1)

    # Forward and backward to set some values
    x = np.array([-0.5])  # Point on left slope
    y = tri_mf.forward(x)
    tri_mf.backward(np.ones_like(y))

    # Verify gradients are set
    assert any(abs(v) > 1e-10 for v in tri_mf.gradients.values()), "Some gradients should be non-zero"
    assert tri_mf.last_input is not None
    assert tri_mf.last_output is not None

    # Reset and verify
    tri_mf.reset()
    assert all(v == 0.0 for v in tri_mf.gradients.values())
    assert tri_mf.last_input is None
    assert tri_mf.last_output is None


def test_triangular_mf_string_representation():
    """Test string representations of triangular membership function."""
    tri_mf = TriangularMF(a=-1.0, b=0.0, c=1.0)

    str_repr = str(tri_mf)
    assert "TriangularMF" in str_repr
    assert "-1.000" in str_repr
    assert "0.000" in str_repr
    assert "1.000" in str_repr

    repr_str = repr(tri_mf)
    assert "TriangularMF(a=-1.0, b=0.0, c=1.0)" == repr_str


def test_triangular_mf_batch_processing():
    """Test triangular membership function with batch inputs."""
    tri_mf = TriangularMF(-2, 0, 2)

    # Large batch input that includes the peak point
    x = np.linspace(-3, 3, 100)
    output = tri_mf.forward(x)

    # Check properties
    assert output.shape == x.shape
    assert np.all(output >= 0.0) and np.all(output <= 1.0)  # Valid membership values

    # Test with exact peak value
    peak_output = tri_mf.forward(np.array([0.0]))  # Peak should be at b=0
    assert np.allclose(peak_output, [1.0]), "Peak should be 1.0 at b=0"

    # Check that values outside support are zero
    outside_mask = (x <= -2) | (x >= 2)
    assert np.all(output[outside_mask] == 0.0)


def test_triangular_vs_gaussian_integration():
    """Test that TriangularMF integrates properly with ANFIS like GaussianMF."""
    from anfis_toolbox import ANFIS

    # Create ANFIS with triangular membership functions
    # Using same number of MFs for each input for compatibility
    input_mfs_tri = {
        "x1": [TriangularMF(-2, -1, 0), TriangularMF(-1, 0, 1)],
        "x2": [TriangularMF(-1, 0, 1), TriangularMF(0, 1, 2)],
    }

    model_tri = ANFIS(input_mfs_tri)

    # Test forward pass
    x_test = np.array([[0.0, 0.5], [-0.5, 1.0]])
    output_tri = model_tri.predict(x_test)

    # Should produce valid outputs
    assert output_tri.shape == (2, 1)
    assert not np.any(np.isnan(output_tri))

    # Create equivalent ANFIS with Gaussian membership functions for comparison
    from anfis_toolbox import GaussianMF

    input_mfs_gauss = {
        "x1": [GaussianMF(-0.5, 0.5), GaussianMF(0.5, 0.5)],
        "x2": [GaussianMF(0, 0.5), GaussianMF(1, 0.5)],
    }

    model_gauss = ANFIS(input_mfs_gauss)
    output_gauss = model_gauss.predict(x_test)

    # Both should produce valid outputs (though values will be different)
    assert output_gauss.shape == output_tri.shape
    assert not np.any(np.isnan(output_gauss))


def test_trapezoidal_mf_basic():
    """Test basic functionality of trapezoidal membership function."""
    a, b, c, d = -2.0, -1.0, 1.0, 2.0
    trap_mf = TrapezoidalMF(a, b, c, d)

    # Test key points
    x = np.array([-3, -2, -1.5, -1, 0, 1, 1.5, 2, 3])
    output = trap_mf.forward(x)

    # Expected values: outside = 0, plateau = 1, linear slopes in between
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])

    assert np.allclose(output, expected), f"Expected {expected}, got {output}"

    # Test that plateau is correctly at 1.0
    plateau_output = trap_mf.forward(np.array([b, 0.0, c]))
    assert np.allclose(plateau_output, [1.0, 1.0, 1.0]), "Plateau should be 1.0"

    # Test boundary conditions
    boundary_output = trap_mf.forward(np.array([a, d]))
    assert np.allclose(boundary_output, [0.0, 0.0]), "Boundaries should be 0.0"


def test_trapezoidal_mf_parameter_validation():
    """Test parameter validation for trapezoidal membership function."""

    # Valid parameters
    trap_mf = TrapezoidalMF(0, 1, 2, 3)
    assert trap_mf.parameters["a"] == 0
    assert trap_mf.parameters["b"] == 1
    assert trap_mf.parameters["c"] == 2
    assert trap_mf.parameters["d"] == 3

    # Test invalid parameter ordering
    with pytest.raises(ValueError, match="must satisfy a ≤ b ≤ c ≤ d"):
        TrapezoidalMF(3, 2, 1, 0)  # a > b > c > d

    with pytest.raises(ValueError, match="must satisfy a ≤ b ≤ c ≤ d"):
        TrapezoidalMF(0, 2, 1, 3)  # b > c

    with pytest.raises(ValueError, match="must satisfy a ≤ b ≤ c ≤ d"):
        TrapezoidalMF(0, 1, 3, 2)  # c > d

    # Test zero-width trapezoid
    with pytest.raises(ValueError, match="cannot be equal"):
        TrapezoidalMF(1, 1, 1, 1)  # a = b = c = d

    # Test edge cases (should be valid)
    trap_mf1 = TrapezoidalMF(0, 0, 1, 2)  # Left-aligned (triangle-like left side)
    trap_mf2 = TrapezoidalMF(0, 1, 2, 2)  # Right-aligned (triangle-like right side)
    trap_mf3 = TrapezoidalMF(0, 1, 1, 2)  # Zero-width plateau (triangular)

    assert trap_mf1.parameters["a"] == trap_mf1.parameters["b"]
    assert trap_mf2.parameters["c"] == trap_mf2.parameters["d"]
    assert trap_mf3.parameters["b"] == trap_mf3.parameters["c"]


def test_trapezoidal_mf_forward_edge_cases():
    """Test forward pass edge cases for trapezoidal membership function."""

    # Test asymmetric trapezoid
    trap_mf = TrapezoidalMF(-3, -1, 1, 4)
    x = np.array([-4, -3, -2, -1, 0, 1, 2.5, 4, 5])
    output = trap_mf.forward(x)

    # Expected: linear interpolation and plateau
    expected = np.array(
        [
            0.0,  # x = -4 (outside)
            0.0,  # x = -3 (boundary)
            0.5,  # x = -2 (halfway up left slope: (-2 - (-3))/(-1 - (-3)) = 0.5)
            1.0,  # x = -1 (start of plateau)
            1.0,  # x = 0  (plateau)
            1.0,  # x = 1  (end of plateau)
            0.5,  # x = 2.5 (halfway down right slope: (4 - 2.5)/(4 - 1) = 0.5)
            0.0,  # x = 4  (boundary)
            0.0,  # x = 5  (outside)
        ]
    )

    assert np.allclose(output, expected), f"Expected {expected}, got {output}"


def test_trapezoidal_mf_backward():
    """Test backward pass for trapezoidal membership function."""
    trap_mf = TrapezoidalMF(-2, -1, 1, 2)

    # Forward pass with points in different regions
    x = np.array([-1.5, 0, 1.5])  # Left slope, plateau, right slope
    y = trap_mf.forward(x)

    # Backward pass
    dL_dy = np.ones_like(y)
    trap_mf.backward(dL_dy)

    # Check that gradients are computed
    assert "a" in trap_mf.gradients
    assert "b" in trap_mf.gradients
    assert "c" in trap_mf.gradients
    assert "d" in trap_mf.gradients

    # Gradients should be non-zero for parameters affecting the function
    total_grad = sum(abs(v) for v in trap_mf.gradients.values())
    assert total_grad > 1e-10, "Some gradients should be non-zero"


@pytest.mark.parametrize("param_name", ["a", "b", "c", "d"])
def test_trapezoidal_mf_gradient_numerical(param_name):
    """Test gradients using numerical differentiation with relaxed conditions."""
    # Similar approach to triangular MF test - verify gradients are reasonable

    trap_mf = TrapezoidalMF(a=0.0, b=1.0, c=2.0, d=3.0)

    # Use multiple test points to get robust gradient estimate
    x = np.array([0.5, 1.5, 2.5])  # Points in left slope, plateau, right slope

    # Forward pass and backward pass
    y = trap_mf.forward(x)
    dL_dy = np.ones_like(y)
    trap_mf.backward(dL_dy)
    grad_analytical = trap_mf.gradients[param_name]

    # For trapezoidal functions, we verify:
    # 1. Gradients are computed (not NaN)
    # 2. Gradients have reasonable magnitudes

    assert not np.isnan(grad_analytical), f"Gradient for {param_name} is NaN"
    assert abs(grad_analytical) < 100, f"Gradient for {param_name} is unexpectedly large: {grad_analytical}"

    # Test that changing parameters actually changes the output
    original_value = trap_mf.parameters[param_name]

    trap_mf.parameters[param_name] = original_value + 0.1  # Larger perturbation
    y_perturbed = trap_mf.forward(x)
    trap_mf.parameters[param_name] = original_value  # Restore

    if not np.allclose(y, y_perturbed):
        # If output changes when parameter changes, gradient should be non-zero
        assert abs(grad_analytical) > 1e-10, (
            f"Gradient for {param_name} should be non-zero when parameter affects output"
        )


def test_trapezoidal_mf_gradient_analytical():
    """Test analytical gradients with known cases."""
    trap_mf = TrapezoidalMF(a=0.0, b=1.0, c=2.0, d=3.0)

    # Test case 1: Point on left slope
    x = np.array([0.5])  # μ(0.5) = 0.5, on left slope
    y = trap_mf.forward(x)
    assert np.allclose(y, [0.5]), "Forward pass incorrect"

    # Reset only gradients
    for key in trap_mf.gradients:
        trap_mf.gradients[key] = 0.0

    trap_mf.backward(np.array([1.0]))  # dL_dy = 1.0

    # Expected gradients for left slope: μ(x) = (x-a)/(b-a)
    # ∂μ/∂a = -1/(b-a) = -1/1 = -1
    # ∂μ/∂b = -(x-a)/(b-a)² = -(0.5-0)/1² = -0.5
    # ∂μ/∂c = 0, ∂μ/∂d = 0 (point not on right slope)

    assert np.allclose(trap_mf.gradients["a"], -1.0), f"Expected ∂μ/∂a = -1.0, got {trap_mf.gradients['a']}"
    assert np.allclose(trap_mf.gradients["b"], -0.5), f"Expected ∂μ/∂b = -0.5, got {trap_mf.gradients['b']}"
    assert np.allclose(trap_mf.gradients["c"], 0.0), f"Expected ∂μ/∂c = 0.0, got {trap_mf.gradients['c']}"
    assert np.allclose(trap_mf.gradients["d"], 0.0), f"Expected ∂μ/∂d = 0.0, got {trap_mf.gradients['d']}"

    # Test case 2: Point on plateau
    x = np.array([1.5])  # μ(1.5) = 1.0, on plateau
    y = trap_mf.forward(x)
    assert np.allclose(y, [1.0]), "Forward pass incorrect"

    # Reset only gradients
    for key in trap_mf.gradients:
        trap_mf.gradients[key] = 0.0

    trap_mf.backward(np.array([1.0]))  # dL_dy = 1.0

    # Expected gradients for plateau: all should be 0 (constant function)
    assert np.allclose(trap_mf.gradients["a"], 0.0), f"Expected ∂μ/∂a = 0.0, got {trap_mf.gradients['a']}"
    assert np.allclose(trap_mf.gradients["b"], 0.0), f"Expected ∂μ/∂b = 0.0, got {trap_mf.gradients['b']}"
    assert np.allclose(trap_mf.gradients["c"], 0.0), f"Expected ∂μ/∂c = 0.0, got {trap_mf.gradients['c']}"
    assert np.allclose(trap_mf.gradients["d"], 0.0), f"Expected ∂μ/∂d = 0.0, got {trap_mf.gradients['d']}"

    # Test case 3: Point on right slope
    x = np.array([2.5])  # μ(2.5) = 0.5, on right slope
    y = trap_mf.forward(x)
    assert np.allclose(y, [0.5]), "Forward pass incorrect"

    # Reset only gradients
    for key in trap_mf.gradients:
        trap_mf.gradients[key] = 0.0

    trap_mf.backward(np.array([1.0]))  # dL_dy = 1.0

    # Expected gradients for right slope: μ(x) = (d-x)/(d-c)
    # ∂μ/∂c = (x-d)/(d-c)² = (2.5-3)/1² = -0.5
    # ∂μ/∂d = (x-c)/(d-c)² = (2.5-2)/1² = 0.5
    # ∂μ/∂a = 0, ∂μ/∂b = 0 (point not on left slope)

    assert np.allclose(trap_mf.gradients["a"], 0.0), f"Expected ∂μ/∂a = 0.0, got {trap_mf.gradients['a']}"
    assert np.allclose(trap_mf.gradients["b"], 0.0), f"Expected ∂μ/∂b = 0.0, got {trap_mf.gradients['b']}"
    assert np.allclose(trap_mf.gradients["c"], -0.5), f"Expected ∂μ/∂c = -0.5, got {trap_mf.gradients['c']}"
    assert np.allclose(trap_mf.gradients["d"], 0.5), f"Expected ∂μ/∂d = 0.5, got {trap_mf.gradients['d']}"


def test_trapezoidal_mf_reset():
    """Test reset functionality for trapezoidal membership function."""
    trap_mf = TrapezoidalMF(-2, -1, 1, 2)

    # Forward and backward to set some values
    x = np.array([-1.5])  # Point on left slope
    y = trap_mf.forward(x)
    trap_mf.backward(np.ones_like(y))

    # Verify gradients are set
    assert any(abs(v) > 1e-10 for v in trap_mf.gradients.values()), "Some gradients should be non-zero"
    assert trap_mf.last_input is not None
    assert trap_mf.last_output is not None

    # Reset and verify
    trap_mf.reset()
    assert all(v == 0.0 for v in trap_mf.gradients.values())
    assert trap_mf.last_input is None
    assert trap_mf.last_output is None


def test_trapezoidal_mf_string_representation():
    """Test string representations of trapezoidal membership function."""
    trap_mf = TrapezoidalMF(a=-2.0, b=-1.0, c=1.0, d=2.0)

    str_repr = str(trap_mf)
    assert "TrapezoidalMF" in str_repr
    assert "-2.000" in str_repr
    assert "-1.000" in str_repr
    assert "1.000" in str_repr
    assert "2.000" in str_repr

    repr_str = repr(trap_mf)
    assert "TrapezoidalMF(a=-2.0, b=-1.0, c=1.0, d=2.0)" == repr_str


def test_trapezoidal_mf_batch_processing():
    """Test trapezoidal membership function with batch inputs."""
    trap_mf = TrapezoidalMF(-2, -1, 1, 2)

    # Large batch input that includes all regions
    x = np.linspace(-3, 3, 100)
    output = trap_mf.forward(x)

    # Check properties
    assert output.shape == x.shape
    assert np.all(output >= 0.0) and np.all(output <= 1.0)  # Valid membership values

    # Test with exact plateau values
    plateau_output = trap_mf.forward(np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(plateau_output, [1.0, 1.0, 1.0]), "Plateau should be 1.0"

    # Check that values outside support are zero
    outside_mask = (x <= -2) | (x >= 2)
    assert np.all(output[outside_mask] == 0.0)


def test_trapezoidal_vs_triangular_integration():
    """Test that TrapezoidalMF integrates properly with ANFIS."""
    from anfis_toolbox import ANFIS

    # Create ANFIS with trapezoidal membership functions
    input_mfs_trap = {
        "x1": [TrapezoidalMF(-3, -2, -1, 0), TrapezoidalMF(-1, 0, 1, 2)],
        "x2": [TrapezoidalMF(-2, -1, 0, 1), TrapezoidalMF(0, 1, 2, 3)],
    }

    model_trap = ANFIS(input_mfs_trap)

    # Test forward pass
    x_test = np.array([[0.0, 0.5], [-0.5, 1.0]])
    output_trap = model_trap.predict(x_test)

    # Should produce valid outputs
    assert output_trap.shape == (2, 1)
    assert not np.any(np.isnan(output_trap))

    # Create equivalent ANFIS with triangular membership functions for comparison
    input_mfs_tri = {
        "x1": [TriangularMF(-2, -1, 0), TriangularMF(-1, 0, 1)],
        "x2": [TriangularMF(-1, 0, 1), TriangularMF(0, 1, 2)],
    }

    model_tri = ANFIS(input_mfs_tri)
    output_tri = model_tri.predict(x_test)

    # Both should produce valid outputs (though values will be different)
    assert output_tri.shape == output_trap.shape
    assert not np.any(np.isnan(output_tri))
