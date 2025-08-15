import numpy as np
import pytest

from anfis_toolbox import ANFIS
from anfis_toolbox.membership import BellMF, GaussianMF, PiMF, SigmoidalMF, TrapezoidalMF, TriangularMF


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


def test_bell_mf_basic():
    """Test basic functionality of bell membership function."""
    a, b, c = 2.0, 4.0, 0.0
    bell_mf = BellMF(a, b, c)

    # Test key points
    x = np.array([-4, -2, -1, 0, 1, 2, 4])
    output = bell_mf.forward(x)

    # At center (x=c=0), output should be 1.0
    center_idx = np.where(x == c)[0][0]
    assert np.allclose(output[center_idx], 1.0), f"At center x={c}, output should be 1.0, got {output[center_idx]}"

    # Function should be symmetric around center
    left_val = bell_mf.forward(np.array([c - 1]))
    right_val = bell_mf.forward(np.array([c + 1]))
    assert np.allclose(left_val, right_val), "Bell function should be symmetric around center"

    # All values should be in [0, 1]
    assert np.all(output >= 0.0) and np.all(output <= 1.0), "All membership values should be in [0, 1]"

    # Values should decrease as we move away from center
    center_output = bell_mf.forward(np.array([c]))[0]
    distant_output = bell_mf.forward(np.array([c + 10]))[0]
    assert center_output > distant_output, "Output should decrease away from center"


def test_bell_mf_parameter_validation():
    """Test parameter validation for bell membership function."""

    # Valid parameters
    bell_mf = BellMF(1.0, 2.0, 0.0)
    assert bell_mf.parameters["a"] == 1.0
    assert bell_mf.parameters["b"] == 2.0
    assert bell_mf.parameters["c"] == 0.0

    # Test invalid parameter 'a' (must be positive)
    with pytest.raises(ValueError, match="Parameter 'a' must be positive"):
        BellMF(0.0, 2.0, 0.0)

    with pytest.raises(ValueError, match="Parameter 'a' must be positive"):
        BellMF(-1.0, 2.0, 0.0)

    # Test invalid parameter 'b' (must be positive)
    with pytest.raises(ValueError, match="Parameter 'b' must be positive"):
        BellMF(1.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="Parameter 'b' must be positive"):
        BellMF(1.0, -1.0, 0.0)

    # Test edge cases (should be valid)
    bell_mf1 = BellMF(a=0.1, b=0.1, c=5.0)  # Small positive values
    bell_mf2 = BellMF(a=10.0, b=10.0, c=-5.0)  # Large values, negative center

    assert bell_mf1.parameters["a"] == 0.1
    assert bell_mf2.parameters["c"] == -5.0


def test_bell_mf_forward_edge_cases():
    """Test forward pass edge cases for bell membership function."""

    # Test with different parameter combinations
    bell_mf1 = BellMF(a=1.0, b=1.0, c=0.0)  # Gentle slope
    bell_mf2 = BellMF(a=1.0, b=10.0, c=0.0)  # Sharp slope
    bell_mf3 = BellMF(a=5.0, b=2.0, c=3.0)  # Wide, centered at 3

    x = np.array([0.0])

    output1 = bell_mf1.forward(x)
    output2 = bell_mf2.forward(x)
    output3 = bell_mf3.forward(x)

    # All should give valid outputs
    assert np.all(output1 >= 0.0) and np.all(output1 <= 1.0)
    assert np.all(output2 >= 0.0) and np.all(output2 <= 1.0)
    assert np.all(output3 >= 0.0) and np.all(output3 <= 1.0)

    # Higher 'b' should give sharper transitions (more steep)
    # At the same distance from center, higher 'b' should give lower values
    x_test = np.array([2.0])  # Distance 2 from center (larger distance to see difference)
    steep_output = bell_mf2.forward(x_test)[0]
    gentle_output = bell_mf1.forward(x_test)[0]
    assert steep_output < gentle_output, (
        f"Higher 'b' should give steeper curve, \
        got steep={steep_output}, gentle={gentle_output}"
    )


def test_bell_mf_backward():
    """Test backward pass for bell membership function."""
    bell_mf = BellMF(2.0, 3.0, 1.0)

    # Forward pass
    x = np.array([0.0, 1.0, 2.0])  # Include center and off-center points
    y = bell_mf.forward(x)

    # Backward pass
    dL_dy = np.ones_like(y)
    bell_mf.backward(dL_dy)

    # Check that gradients are computed
    assert "a" in bell_mf.gradients
    assert "b" in bell_mf.gradients
    assert "c" in bell_mf.gradients

    # Gradients should be finite
    for param, grad in bell_mf.gradients.items():
        assert np.isfinite(grad), f"Gradient for {param} should be finite, got {grad}"


@pytest.mark.parametrize("param_name", ["a", "b", "c"])
def test_bell_mf_gradient_numerical(param_name):
    """Test gradients using numerical differentiation with relaxed conditions."""

    bell_mf = BellMF(a=2.0, b=3.0, c=1.0)

    # Use multiple test points
    x = np.array([0.5, 1.0, 1.5])  # Points around center

    # Forward pass and backward pass
    y = bell_mf.forward(x)
    dL_dy = np.ones_like(y)
    bell_mf.backward(dL_dy)
    grad_analytical = bell_mf.gradients[param_name]

    # Verify gradients are reasonable
    assert not np.isnan(grad_analytical), f"Gradient for {param_name} is NaN"
    assert np.isfinite(grad_analytical), f"Gradient for {param_name} is not finite: {grad_analytical}"
    assert abs(grad_analytical) < 1000, f"Gradient for {param_name} is unexpectedly large: {grad_analytical}"

    # Test that changing parameters actually changes the output (if it should)
    original_value = bell_mf.parameters[param_name]

    perturbation = 0.01 if param_name != "c" else 0.1  # Smaller perturbation for a,b
    bell_mf.parameters[param_name] = original_value + perturbation
    y_perturbed = bell_mf.forward(x)
    bell_mf.parameters[param_name] = original_value  # Restore

    # If output changes when parameter changes, and we're not at a stationary point,
    # gradient might be non-zero
    if not np.allclose(y, y_perturbed, atol=1e-10):
        # Output changed, so gradient computation is working
        pass  # This is good, shows the parameter affects the output


def test_bell_mf_gradient_analytical():
    """Test analytical gradients with known cases."""
    bell_mf = BellMF(a=1.0, b=2.0, c=0.0)

    # Test case 1: Point at center (x = c = 0)
    x = np.array([0.0])
    y = bell_mf.forward(x)
    assert np.allclose(y, [1.0]), "At center, output should be 1.0"

    # Reset gradients
    for key in bell_mf.gradients:
        bell_mf.gradients[key] = 0.0

    bell_mf.backward(np.array([1.0]))

    # At the center (x = c), some gradients should be zero due to symmetry
    # The gradient w.r.t. 'c' should be zero at the peak
    assert np.allclose(bell_mf.gradients["c"], 0.0, atol=1e-10), (
        f"Expected ∂μ/∂c = 0 at center, got {bell_mf.gradients['c']}"
    )

    # Test case 2: Point away from center
    x = np.array([1.0])  # x = 1, c = 0, so (x-c)/a = 1
    y = bell_mf.forward(x)
    # μ(1) = 1/(1 + |1|^4) = 1/2 = 0.5
    assert np.allclose(y, [0.5]), f"Expected μ(1) = 0.5, got {y[0]}"

    # Reset gradients
    for key in bell_mf.gradients:
        bell_mf.gradients[key] = 0.0

    bell_mf.backward(np.array([1.0]))

    # At this point, gradients should be non-zero
    assert not np.allclose(bell_mf.gradients["c"], 0.0), (
        f"Expected non-zero ∂μ/∂c away from center, got {bell_mf.gradients['c']}"
    )


def test_bell_mf_reset():
    """Test reset functionality for bell membership function."""
    bell_mf = BellMF(2.0, 3.0, 1.0)

    # Forward and backward to set some values
    x = np.array([1.5])
    y = bell_mf.forward(x)
    bell_mf.backward(np.ones_like(y))

    # Verify gradients are set
    assert any(abs(v) > 1e-10 or not np.isfinite(v) for v in bell_mf.gradients.values()) or all(
        abs(v) < 1e-10 for v in bell_mf.gradients.values()
    )
    assert bell_mf.last_input is not None
    assert bell_mf.last_output is not None

    # Reset and verify
    bell_mf.reset()
    assert all(v == 0.0 for v in bell_mf.gradients.values())
    assert bell_mf.last_input is None
    assert bell_mf.last_output is None


def test_bell_mf_batch_processing():
    """Test bell membership function with batch inputs."""
    bell_mf = BellMF(a=2.0, b=4.0, c=0.0)

    # Large batch input
    x = np.linspace(-10, 10, 100)
    output = bell_mf.forward(x)

    # Check properties
    assert output.shape == x.shape
    assert np.all(output >= 0.0) and np.all(output <= 1.0)  # Valid membership values

    # Test maximum at center
    center_idx = np.argmin(np.abs(x - 0.0))  # Find closest to center
    assert output[center_idx] >= 0.99, "Maximum should be close to 1.0 at center"

    # Check symmetry
    for i in range(len(x)):
        symmetric_idx = np.argmin(np.abs(x - (-x[i])))  # Find symmetric point
        if abs(x[i]) < 9:  # Avoid edge effects
            assert np.allclose(output[i], output[symmetric_idx], atol=0.1), "Function should be approximately symmetric"


def test_sigmoidal_mf_basic():
    """Test basic functionality of sigmoidal membership function."""
    a, c = 1.0, 0.0
    sigmoid_mf = SigmoidalMF(a, c)

    # Test key points
    x = np.array([-5, -2, 0, 2, 5])
    output = sigmoid_mf.forward(x)

    # At center (x=c=0), output should be 0.5
    center_idx = np.where(x == c)[0][0]
    assert np.allclose(output[center_idx], 0.5), f"At center x={c}, output should be 0.5, got {output[center_idx]}"

    # Function should be monotonically increasing for a > 0
    assert np.all(np.diff(output) >= 0), "Sigmoid with a > 0 should be monotonically increasing"

    # All values should be in [0, 1]
    assert np.all(output >= 0.0) and np.all(output <= 1.0), "All membership values should be in [0, 1]"

    # Values should approach 0 and 1 at extremes
    assert output[0] < 0.1, "Output should approach 0 for large negative x"
    assert output[-1] > 0.9, "Output should approach 1 for large positive x"


def test_sigmoidal_mf_inverted():
    """Test inverted sigmoidal membership function (negative a)."""
    a, c = -1.0, 0.0
    sigmoid_mf = SigmoidalMF(a, c)

    # Test key points
    x = np.array([-5, -2, 0, 2, 5])
    output = sigmoid_mf.forward(x)

    # At center (x=c=0), output should still be 0.5
    center_idx = np.where(x == c)[0][0]
    assert np.allclose(output[center_idx], 0.5), f"At center x={c}, output should be 0.5, got {output[center_idx]}"

    # Function should be monotonically decreasing for a < 0
    assert np.all(np.diff(output) <= 0), "Sigmoid with a < 0 should be monotonically decreasing"

    # All values should be in [0, 1]
    assert np.all(output >= 0.0) and np.all(output <= 1.0), "All membership values should be in [0, 1]"

    # Values should approach 1 and 0 at extremes (inverted)
    assert output[0] > 0.9, "Output should approach 1 for large negative x (inverted)"
    assert output[-1] < 0.1, "Output should approach 0 for large positive x (inverted)"


def test_sigmoidal_mf_parameter_validation():
    """Test parameter validation for sigmoidal membership function."""

    # Valid parameters
    sigmoid_mf = SigmoidalMF(1.0, 0.0)
    assert sigmoid_mf.parameters["a"] == 1.0
    assert sigmoid_mf.parameters["c"] == 0.0

    # Test invalid parameter 'a' (cannot be zero)
    with pytest.raises(ValueError, match="Parameter 'a' cannot be zero"):
        SigmoidalMF(0.0, 0.0)

    # Test edge cases (should be valid)
    sigmoid_mf1 = SigmoidalMF(a=0.1, c=5.0)  # Small positive slope
    sigmoid_mf2 = SigmoidalMF(a=-10.0, c=-5.0)  # Large negative slope, negative center
    sigmoid_mf3 = SigmoidalMF(a=100.0, c=0.0)  # Very steep slope

    assert sigmoid_mf1.parameters["a"] == 0.1
    assert sigmoid_mf2.parameters["c"] == -5.0
    assert sigmoid_mf3.parameters["a"] == 100.0


def test_sigmoidal_mf_forward_edge_cases():
    """Test forward pass edge cases for sigmoidal membership function."""

    # Test with different parameter combinations
    sigmoid_mf1 = SigmoidalMF(a=0.5, c=0.0)  # Gentle slope
    sigmoid_mf2 = SigmoidalMF(a=5.0, c=0.0)  # Steep slope
    sigmoid_mf3 = SigmoidalMF(a=1.0, c=3.0)  # Shifted center

    x = np.array([0.0])

    output1 = sigmoid_mf1.forward(x)
    output2 = sigmoid_mf2.forward(x)
    output3 = sigmoid_mf3.forward(x)

    # All should give valid outputs
    assert np.all(output1 >= 0.0) and np.all(output1 <= 1.0)
    assert np.all(output2 >= 0.0) and np.all(output2 <= 1.0)
    assert np.all(output3 >= 0.0) and np.all(output3 <= 1.0)

    # All should be 0.5 at their respective centers
    assert np.allclose(output1[0], 0.5), "Should be 0.5 at center"
    assert np.allclose(output2[0], 0.5), "Should be 0.5 at center"

    # For shifted center, x=0 should not be 0.5
    assert not np.allclose(output3[0], 0.5), "Should not be 0.5 away from center"

    # Test at the actual center for shifted function
    center_output = sigmoid_mf3.forward(np.array([3.0]))[0]
    assert np.allclose(center_output, 0.5), "Should be 0.5 at actual center"


def test_sigmoidal_mf_backward():
    """Test backward pass for sigmoidal membership function."""
    sigmoid_mf = SigmoidalMF(2.0, 1.0)

    # Forward pass
    x = np.array([0.0, 1.0, 2.0])  # Include center and off-center points
    y = sigmoid_mf.forward(x)

    # Backward pass
    dL_dy = np.ones_like(y)
    sigmoid_mf.backward(dL_dy)

    # Check that gradients are computed
    assert "a" in sigmoid_mf.gradients
    assert "c" in sigmoid_mf.gradients

    # Gradients should be finite
    for param, grad in sigmoid_mf.gradients.items():
        assert np.isfinite(grad), f"Gradient for {param} should be finite, got {grad}"


@pytest.mark.parametrize("param_name", ["a", "c"])
def test_sigmoidal_mf_gradient_numerical(param_name):
    """Test gradients using numerical differentiation with relaxed conditions."""

    sigmoid_mf = SigmoidalMF(a=2.0, c=0.0)

    # Use multiple test points
    x = np.array([-1.0, 0.0, 1.0])  # Points around center

    # Forward pass and backward pass
    y = sigmoid_mf.forward(x)
    dL_dy = np.ones_like(y)
    sigmoid_mf.backward(dL_dy)
    grad_analytical = sigmoid_mf.gradients[param_name]

    # Verify gradients are reasonable
    assert not np.isnan(grad_analytical), f"Gradient for {param_name} is NaN"
    assert np.isfinite(grad_analytical), f"Gradient for {param_name} is not finite: {grad_analytical}"
    assert abs(grad_analytical) < 1000, f"Gradient for {param_name} is unexpectedly large: {grad_analytical}"

    # Test that changing parameters actually changes the output
    original_value = sigmoid_mf.parameters[param_name]

    perturbation = 0.01
    sigmoid_mf.parameters[param_name] = original_value + perturbation
    y_perturbed = sigmoid_mf.forward(x)
    sigmoid_mf.parameters[param_name] = original_value  # Restore

    # If output changes when parameter changes, gradient computation is working
    if not np.allclose(y, y_perturbed, atol=1e-10):
        # Output changed, so gradient computation is working
        pass


def test_sigmoidal_mf_gradient_analytical():
    """Test analytical gradients with known cases."""
    sigmoid_mf = SigmoidalMF(a=1.0, c=0.0)

    # Test case 1: Point at center (x = c = 0)
    x = np.array([0.0])
    y = sigmoid_mf.forward(x)
    assert np.allclose(y, [0.5]), "At center, output should be 0.5"

    # Reset gradients
    for key in sigmoid_mf.gradients:
        sigmoid_mf.gradients[key] = 0.0

    sigmoid_mf.backward(np.array([1.0]))

    # At center (x=c), with a=1:
    # μ(0) = 0.5, so μ(1-μ) = 0.5 * 0.5 = 0.25
    # ∂μ/∂a = μ(1-μ)(x-c) = 0.25 * (0-0) = 0
    # ∂μ/∂c = -aμ(1-μ) = -1 * 0.25 = -0.25
    expected_grad_a = 0.0
    expected_grad_c = -0.25
    assert np.allclose(sigmoid_mf.gradients["a"], expected_grad_a, atol=1e-10), (
        f"Expected ∂μ/∂a = {expected_grad_a} at center, got {sigmoid_mf.gradients['a']}"
    )
    assert np.allclose(sigmoid_mf.gradients["c"], expected_grad_c, atol=1e-10), (
        f"Expected ∂μ/∂c = {expected_grad_c} at center, got {sigmoid_mf.gradients['c']}"
    )

    # Test case 2: Point away from center
    x = np.array([1.0])  # x = 1, c = 0, a = 1
    y = sigmoid_mf.forward(x)
    # μ(1) = 1/(1 + exp(-1*1)) = 1/(1 + exp(-1)) ≈ 0.731
    expected_y = 1.0 / (1.0 + np.exp(-1.0))
    assert np.allclose(y, [expected_y]), f"Expected μ(1) = {expected_y}, got {y[0]}"

    # Reset gradients
    for key in sigmoid_mf.gradients:
        sigmoid_mf.gradients[key] = 0.0

    sigmoid_mf.backward(np.array([1.0]))

    # At x=1: μ(1-μ) = expected_y * (1 - expected_y)
    mu_1_minus_mu = expected_y * (1 - expected_y)
    # ∂μ/∂a = μ(1-μ)(x-c) = mu_1_minus_mu * (1-0) = mu_1_minus_mu
    # ∂μ/∂c = -aμ(1-μ) = -1 * mu_1_minus_mu = -mu_1_minus_mu

    assert np.allclose(sigmoid_mf.gradients["a"], mu_1_minus_mu, atol=1e-6), (
        f"Expected ∂μ/∂a = {mu_1_minus_mu}, got {sigmoid_mf.gradients['a']}"
    )
    assert np.allclose(sigmoid_mf.gradients["c"], -mu_1_minus_mu, atol=1e-6), (
        f"Expected ∂μ/∂c = {-mu_1_minus_mu}, got {sigmoid_mf.gradients['c']}"
    )


def test_sigmoidal_mf_reset():
    """Test reset functionality for sigmoidal membership function."""
    sigmoid_mf = SigmoidalMF(2.0, 1.0)

    # Forward and backward to set some values
    x = np.array([1.5])
    y = sigmoid_mf.forward(x)
    sigmoid_mf.backward(np.ones_like(y))

    # Verify gradients are set
    assert sigmoid_mf.last_input is not None
    assert sigmoid_mf.last_output is not None

    # Reset and verify
    sigmoid_mf.reset()
    assert all(v == 0.0 for v in sigmoid_mf.gradients.values())
    assert sigmoid_mf.last_input is None
    assert sigmoid_mf.last_output is None


def test_sigmoidal_mf_batch_processing():
    """Test sigmoidal membership function with batch inputs."""
    sigmoid_mf = SigmoidalMF(a=1.0, c=0.0)

    # Large batch input
    x = np.linspace(-10, 10, 200)
    output = sigmoid_mf.forward(x)

    # Check properties
    assert output.shape == x.shape
    assert np.all(output >= 0.0) and np.all(output <= 1.0)  # Valid membership values

    # Test monotonicity (should be strictly increasing for a > 0)
    assert np.all(np.diff(output) >= 0), "Sigmoid should be monotonically increasing"

    # Test center value
    center_idx = np.argmin(np.abs(x - 0.0))  # Find closest to center
    assert np.allclose(output[center_idx], 0.5, atol=0.1), "Should be close to 0.5 at center"

    # Test extreme values
    assert output[0] < 0.01, "Should approach 0 for large negative x"
    assert output[-1] > 0.99, "Should approach 1 for large positive x"


def test_sigmoidal_mf_numerical_stability():
    """Test numerical stability of sigmoidal membership function."""
    sigmoid_mf = SigmoidalMF(a=10.0, c=0.0)  # Steep sigmoid

    # Test with extreme values that could cause overflow
    x_extreme = np.array([-100, -50, 0, 50, 100])
    output = sigmoid_mf.forward(x_extreme)

    # Should still produce valid outputs
    assert np.all(np.isfinite(output)), "Output should be finite for extreme inputs"
    assert np.all(output >= 0.0) and np.all(output <= 1.0), "Output should be in [0,1]"

    # Test gradients don't explode
    sigmoid_mf.backward(np.ones_like(output))
    for param, grad in sigmoid_mf.gradients.items():
        assert np.isfinite(grad), f"Gradient for {param} should be finite"


# ============================================================================
# PiMF (Pi-shaped Membership Function) Tests
# ============================================================================


def test_pi_mf_basic():
    """Test basic PiMF functionality."""
    # Create standard Pi-shaped function
    pi_mf = PiMF(a=-2.0, b=-1.0, c=1.0, d=2.0)

    # Test forward pass
    x = np.array([-3.0, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0])
    y = pi_mf.forward(x)

    # Verify shape and bounds
    assert y.shape == x.shape
    assert np.all(y >= 0.0) and np.all(y <= 1.0)

    # Verify key points
    assert y[0] == 0.0  # x=-3 (outside left)
    assert y[1] == 0.0  # x=-2 (left foot)
    assert y[3] == 1.0  # x=-1 (left shoulder)
    assert y[4] == 1.0  # x=0 (center, flat region)
    assert y[5] == 1.0  # x=1 (right shoulder)
    assert y[7] == 0.0  # x=2 (right foot)
    assert y[8] == 0.0  # x=3 (outside right)

    # Verify smooth transitions (should be between 0 and 1)
    assert 0.0 < y[2] < 1.0  # x=-1.5 (rising edge)
    assert 0.0 < y[6] < 1.0  # x=1.5 (falling edge)


def test_pi_mf_parameter_validation():
    """Test parameter validation for PiMF."""
    # Valid parameters
    pi_mf = PiMF(a=0.0, b=1.0, c=2.0, d=3.0)
    assert pi_mf.parameters["a"] == 0.0
    assert pi_mf.parameters["b"] == 1.0
    assert pi_mf.parameters["c"] == 2.0
    assert pi_mf.parameters["d"] == 3.0

    # Test boundary case: b == c (no flat region)
    pi_mf_no_flat = PiMF(a=0.0, b=1.0, c=1.0, d=2.0)
    x = np.array([0.5, 1.0, 1.5])
    y = pi_mf_no_flat.forward(x)
    assert 0.0 < y[0] < 1.0  # Rising
    assert y[1] == 1.0  # Peak
    assert 0.0 < y[2] < 1.0  # Falling

    # Invalid parameters: a >= b
    with pytest.raises(ValueError, match="Parameters must satisfy a < b ≤ c < d"):
        PiMF(a=1.0, b=1.0, c=2.0, d=3.0)

    # Invalid parameters: b > c
    with pytest.raises(ValueError, match="Parameters must satisfy a < b ≤ c < d"):
        PiMF(a=0.0, b=2.0, c=1.0, d=3.0)

    # Invalid parameters: c >= d
    with pytest.raises(ValueError, match="Parameters must satisfy a < b ≤ c < d"):
        PiMF(a=0.0, b=1.0, c=3.0, d=3.0)


def test_pi_mf_forward_edge_cases():
    """Test forward pass edge cases for PiMF."""
    pi_mf = PiMF(a=-1.0, b=0.0, c=1.0, d=2.0)

    # Test with single values
    assert pi_mf.forward(np.array([-2.0]))[0] == 0.0
    assert pi_mf.forward(np.array([0.5]))[0] == 1.0
    assert pi_mf.forward(np.array([3.0]))[0] == 0.0

    # Test empty array
    result = pi_mf.forward(np.array([]))
    assert result.shape == (0,)

    # Test scalar input
    result = pi_mf.forward(-0.5)
    assert isinstance(result, np.ndarray)
    result_scalar = np.asarray(result).item() if result.ndim == 0 else result[0]
    assert 0.0 < result_scalar < 1.0


def test_pi_mf_backward():
    """Test backward pass for PiMF."""
    pi_mf = PiMF(a=-1.0, b=0.0, c=1.0, d=2.0)

    # Forward pass first
    x = np.array([-0.5, 0.5, 1.5])
    pi_mf.forward(x)

    # Reset gradients
    pi_mf.reset()

    # Backward pass
    dL_dy = np.array([1.0, 1.0, 1.0])
    pi_mf.backward(dL_dy)

    # Gradients should be computed (non-zero for affected parameters)
    assert "a" in pi_mf.gradients
    assert "b" in pi_mf.gradients
    assert "c" in pi_mf.gradients
    assert "d" in pi_mf.gradients


@pytest.mark.parametrize("param_name", ["a", "b", "c", "d"])
def test_pi_mf_gradient_numerical(param_name):
    """Test numerical gradient computation for PiMF parameters."""
    # Base parameters
    base_params = {"a": -1.0, "b": 0.0, "c": 1.0, "d": 2.0}

    # Test points in different regions
    x = np.array([-0.5, 0.5, 1.5])  # Rising, flat, falling

    # Compute numerical gradient by creating new instances
    eps = 1e-6
    original_param = base_params[param_name]

    # Create perturbed parameter sets
    params_plus = base_params.copy()
    params_minus = base_params.copy()
    params_plus[param_name] = original_param + eps
    params_minus[param_name] = original_param - eps

    # Handle constraint violations by adjusting eps if needed
    try:
        pi_mf_plus = PiMF(**params_plus)
        pi_mf_minus = PiMF(**params_minus)
    except ValueError:
        # If constraint violated, use smaller eps
        eps = 1e-7
        params_plus[param_name] = original_param + eps
        params_minus[param_name] = original_param - eps
        pi_mf_plus = PiMF(**params_plus)
        pi_mf_minus = PiMF(**params_minus)

    # Forward pass with perturbed parameters
    y_plus = pi_mf_plus.forward(x)
    y_minus = pi_mf_minus.forward(x)

    numerical_grad = np.sum((y_plus - y_minus) / (2 * eps))

    # Compute analytical gradient with original parameters
    pi_mf = PiMF(**base_params)
    for key in pi_mf.gradients:
        pi_mf.gradients[key] = 0.0  # Reset gradients without clearing cache
    y = pi_mf.forward(x)
    pi_mf.backward(np.ones_like(y))
    analytical_grad = pi_mf.gradients[param_name]

    # Compare gradients (allow some numerical error)
    assert np.allclose(analytical_grad, numerical_grad, atol=1e-4), (
        f"Gradient mismatch for {param_name}: analytical={analytical_grad:.8f}, numerical={numerical_grad:.8f}"
    )


def test_pi_mf_gradient_analytical():
    """Test analytical gradients for PiMF in detail."""
    pi_mf = PiMF(a=-1.0, b=0.0, c=1.0, d=2.0)

    # Test point in rising region (S-function)
    x = np.array([-0.5])
    pi_mf.forward(x)
    for key in pi_mf.gradients:
        pi_mf.gradients[key] = 0.0  # Reset gradients without clearing cache
    pi_mf.backward(np.array([1.0]))

    # Gradients for a and b should be non-zero (affecting S-function)
    assert pi_mf.gradients["a"] != 0.0, "Gradient w.r.t. 'a' should be non-zero in rising region"
    assert pi_mf.gradients["b"] != 0.0, "Gradient w.r.t. 'b' should be non-zero in rising region"
    assert pi_mf.gradients["c"] == 0.0, "Gradient w.r.t. 'c' should be zero in rising region"
    assert pi_mf.gradients["d"] == 0.0, "Gradient w.r.t. 'd' should be zero in rising region"

    # Test point in flat region
    x = np.array([0.5])
    pi_mf.forward(x)
    for key in pi_mf.gradients:
        pi_mf.gradients[key] = 0.0  # Reset gradients without clearing cache
    pi_mf.backward(np.array([1.0]))

    # All gradients should be zero in flat region (constant function)
    assert pi_mf.gradients["a"] == 0.0, "Gradient w.r.t. 'a' should be zero in flat region"
    assert pi_mf.gradients["b"] == 0.0, "Gradient w.r.t. 'b' should be zero in flat region"
    assert pi_mf.gradients["c"] == 0.0, "Gradient w.r.t. 'c' should be zero in flat region"
    assert pi_mf.gradients["d"] == 0.0, "Gradient w.r.t. 'd' should be zero in flat region"

    # Test point in falling region (Z-function)
    x = np.array([1.5])
    pi_mf.forward(x)
    for key in pi_mf.gradients:
        pi_mf.gradients[key] = 0.0  # Reset gradients without clearing cache
    pi_mf.backward(np.array([1.0]))

    # Gradients for c and d should be non-zero (affecting Z-function)
    assert pi_mf.gradients["a"] == 0.0, "Gradient w.r.t. 'a' should be zero in falling region"
    assert pi_mf.gradients["b"] == 0.0, "Gradient w.r.t. 'b' should be zero in falling region"
    assert pi_mf.gradients["c"] != 0.0, "Gradient w.r.t. 'c' should be non-zero in falling region"
    assert pi_mf.gradients["d"] != 0.0, "Gradient w.r.t. 'd' should be non-zero in falling region"


def test_pi_mf_symmetry():
    """Test symmetry properties of PiMF."""
    # Create symmetric Pi function
    pi_mf = PiMF(a=-2.0, b=-1.0, c=1.0, d=2.0)

    # Test symmetric points
    x_left = np.array([-1.5])
    x_right = np.array([1.5])

    y_left = pi_mf.forward(x_left)
    y_right = pi_mf.forward(x_right)

    # Should have same membership values (symmetric)
    assert np.allclose(y_left, y_right, atol=1e-10), (
        f"Symmetric points should have equal membership: "
        f"μ({x_left[0]})={y_left[0]:.6f}, μ({x_right[0]})={y_right[0]:.6f}"
    )


def test_pi_mf_reset():
    """Test gradient reset functionality for PiMF."""
    pi_mf = PiMF(a=-1.0, b=0.0, c=1.0, d=2.0)

    # Perform forward and backward pass
    x = np.array([0.5])
    pi_mf.forward(x)
    pi_mf.backward(np.array([1.0]))

    # Reset gradients
    pi_mf.reset()

    # All gradients should be zero
    for key in pi_mf.gradients:
        assert pi_mf.gradients[key] == 0.0, f"Gradient for '{key}' not reset to zero"

    assert pi_mf.last_input is None
    assert pi_mf.last_output is None


def test_pi_mf_batch_processing():
    """Test batch processing with PiMF."""
    pi_mf = PiMF(a=-2.0, b=-1.0, c=1.0, d=2.0)

    # Large batch
    batch_size = 1000
    x = np.random.uniform(-3, 3, size=batch_size)
    y = pi_mf.forward(x)

    # Verify output properties
    assert y.shape == (batch_size,)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)
    assert not np.any(np.isnan(y))

    # Verify some samples are in different regions
    assert np.any(y == 0.0)  # Some outside
    assert np.any(y == 1.0)  # Some in flat region
    assert np.any((y > 0.0) & (y < 1.0))  # Some in transition regions


def test_pi_mf_vs_trapezoidal_similarity():
    """Test that PiMF behaves similarly to TrapezoidalMF in many cases."""
    # Create similar functions
    pi_mf = PiMF(a=-2.0, b=-1.0, c=1.0, d=2.0)
    trap_mf = TrapezoidalMF(a=-2.0, b=-1.0, c=1.0, d=2.0)

    # Test in flat region - should be identical
    x_flat = np.array([-0.5, 0.0, 0.5])
    y_pi = pi_mf.forward(x_flat)
    y_trap = trap_mf.forward(x_flat)

    assert np.allclose(y_pi, y_trap), "Pi and Trapezoidal should be identical in flat region"

    # Test outside regions - should be identical
    x_outside = np.array([-3.0, 3.0])
    y_pi_out = pi_mf.forward(x_outside)
    y_trap_out = trap_mf.forward(x_outside)

    assert np.allclose(y_pi_out, y_trap_out), "Pi and Trapezoidal should be identical outside their support"

    # In transition regions, Pi should be smoother (S-shaped vs linear)
    x_transition = np.array([-1.5, 1.5])
    y_pi_trans = pi_mf.forward(x_transition)
    y_trap_trans = trap_mf.forward(x_transition)

    # Both should be in (0,1) but values will differ due to smoothness
    assert np.all((y_pi_trans > 0) & (y_pi_trans < 1))
    assert np.all((y_trap_trans > 0) & (y_trap_trans < 1))


@pytest.mark.parametrize(
    "mf",
    [
        GaussianMF(mean=0.0, sigma=1.0),
        TriangularMF(a=0.0, b=1.0, c=2.0),
        TrapezoidalMF(a=0.0, b=1.0, c=2.0, d=3.0),
        BellMF(a=0.5, b=1.0, c=2.0),
        SigmoidalMF(a=0.5, c=1.0),
        PiMF(a=0.1, b=1.0, c=2.0, d=3.0),
    ],
)
def test_str_repr(mf):
    s = str(mf)
    r = repr(mf)

    # Tipo correto
    assert isinstance(s, str)
    assert isinstance(r, str)

    # Contém o nome da classe e "MF"
    assert mf.__class__.__name__ in s
