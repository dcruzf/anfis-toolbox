import numpy as np
import pytest

from anfis_toolbox.membership import TriangularMF

# -----------------------------
# Construction / validation
# -----------------------------


def test_init_valid_params():
    """TriangularMF initializes with valid parameters without error."""
    mf = TriangularMF(a=0.0, b=1.0, c=2.0)
    assert mf.parameters == {"a": 0.0, "b": 1.0, "c": 2.0}


def test_init_invalid_ordering_raises():
    # a > b
    with pytest.raises(ValueError):
        TriangularMF(2.0, 1.0, 3.0)
    # b > c
    with pytest.raises(ValueError):
        TriangularMF(0.0, 3.0, 2.0)


def test_init_zero_width_raises():
    # a == c is not allowed
    with pytest.raises(ValueError):
        TriangularMF(1.0, 1.0, 1.0)


# -----------------------------
# Forward piecewise behavior
# -----------------------------


def test_forward_shape_and_values():
    """Forward returns correct μ(x) values for different regions."""
    mf = TriangularMF(a=0.0, b=1.0, c=2.0)
    x = np.array([-1.0, 0.5, 1.0, 1.5, 3.0])
    y = mf.forward(x)
    expected = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    np.testing.assert_allclose(y, expected, rtol=1e-6)


def test_forward_basic_segments():
    mf = TriangularMF(0.0, 1.0, 2.0)
    x = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    y = mf.forward(x)
    expected = np.array([0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
    assert np.allclose(y, expected)


def test_forward_degenerate_left_slope_b_equals_a():
    # When b == a, left slope is disabled; only peak at x==b and right slope exist
    mf = TriangularMF(1.0, 1.0, 3.0)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = mf.forward(x)
    # Right slope values: (c - x)/(c - b) for b < x < c, with c-b = 2
    expected = np.array([0.0, 1.0, 0.5, 0.0, 0.0])
    assert np.allclose(y, expected)


def test_forward_degenerate_right_slope_c_equals_b():
    # When c == b, right slope is disabled; only left slope and peak exist
    mf = TriangularMF(0.0, 1.0, 1.0)
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    y = mf.forward(x)
    expected = np.array([0.0, 0.0, 0.5, 1.0, 0.0])
    assert np.allclose(y, expected)


# -----------------------------
# Backward gradients
# -----------------------------


def test_backward_gradients_sign_and_shape():
    """Backward computes nonzero gradients with expected signs."""
    mf = TriangularMF(a=0.0, b=1.0, c=2.0)
    x = np.array([0.5, 1.5])  # one on left slope, one on right slope
    mf.forward(x)
    dL_dy = np.array([1.0, 1.0])
    mf.backward(dL_dy)

    grads = mf.gradients
    assert all(k in grads for k in ["a", "b", "c"])
    assert grads["a"] < 0  # shifting a left increases μ on left slope
    assert grads["c"] > 0  # shifting c right increases μ on right slope


def test_backward_gradients_simple_triangle():
    # a=0, b=1, c=2 gives clean unit denominators
    mf = TriangularMF(0.0, 1.0, 2.0)
    x = np.array([-1.0, 0.5, 1.0, 1.5, 3.0])
    mf.forward(x)

    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)

    # Expected gradients from one left-slope point (0.5) and one right-slope point (1.5):
    # Left (x=0.5): dmu/da = (0.5-1.0) = -0.5; dmu/db = -(0.5-0.0) = -0.5
    # Right (x=1.5): dmu/db = (1.5-2.0) = -0.5; dmu/dc = (1.5-1.0) = 0.5
    assert mf.gradients["a"] == pytest.approx(-0.5)
    assert mf.gradients["b"] == pytest.approx(-1.0)
    assert mf.gradients["c"] == pytest.approx(0.5)


def test_backward_gradients_degenerate_left_slope():
    # b == a disables left slope; only right slope contributes
    mf = TriangularMF(1.0, 1.0, 3.0)
    x = np.array([1.0, 2.0])  # peak and right-slope
    mf.forward(x)

    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)

    # For x=2.0, (c-b) = 2 -> squared = 4
    # dmu/db = (x-c)/(c-b)^2 = (2-3)/4 = -0.25
    # dmu/dc = (x-b)/(c-b)^2 = (2-1)/4 = 0.25
    assert mf.gradients["a"] == pytest.approx(0.0)
    assert mf.gradients["b"] == pytest.approx(-0.25)
    assert mf.gradients["c"] == pytest.approx(0.25)


def test_backward_gradients_degenerate_right_slope():
    # c == b disables right slope; only left slope contributes
    mf = TriangularMF(0.0, 1.0, 1.0)
    x = np.array([0.5, 1.0])  # left-slope and peak
    mf.forward(x)

    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)

    # For x=0.5 with (b-a)=1:
    # dmu/da = (0.5-1.0) = -0.5, dmu/db = -(0.5-0.0) = -0.5
    assert mf.gradients["a"] == pytest.approx(-0.5)
    assert mf.gradients["b"] == pytest.approx(-0.5)
    assert mf.gradients["c"] == pytest.approx(0.0)


def test_backward_left_slope_only_points_and_boundaries():
    # Triangle with (b-a) = 1 for simple math
    mf = TriangularMF(0.0, 1.0, 2.0)
    # Include boundaries a and b which should not contribute, and two interior left-slope points
    x = np.array([0.0, 0.3, 0.7, 1.0])
    mf.forward(x)

    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)

    # For left slope: dmu/da = (x - b), dmu/db = -(x - a); here (b-a)=1
    # x=0.3 -> da=-0.7, db=-0.3; x=0.7 -> da=-0.3, db=-0.7; boundaries add 0
    assert mf.gradients["a"] == pytest.approx(-1.0)
    assert mf.gradients["b"] == pytest.approx(-1.0)
    assert mf.gradients["c"] == pytest.approx(0.0)


def test_backward_right_slope_only_points_and_boundaries():
    # Triangle with (c-b) = 1 for simple math
    mf = TriangularMF(0.0, 1.0, 2.0)
    # Include boundaries b and c which should not contribute, and two interior right-slope points
    x = np.array([1.0, 1.3, 1.7, 2.0])
    mf.forward(x)

    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)

    # For right slope: dmu/db = (x - c), dmu/dc = (x - b); here (c-b)=1
    # x=1.3 -> db=-0.7, dc=0.3; x=1.7 -> db=-0.3, dc=0.7; boundaries add 0
    assert mf.gradients["a"] == pytest.approx(0.0)
    assert mf.gradients["b"] == pytest.approx(-1.0)
    assert mf.gradients["c"] == pytest.approx(1.0)


# -----------------------------
# String representations
# -----------------------------


def test_str_and_repr_have_class_name():
    mf = TriangularMF(0.0, 1.0, 2.0)
    s = str(mf)
    r = repr(mf)
    assert "TriangularMF(" in s
    assert "TriangularMF(" in r
