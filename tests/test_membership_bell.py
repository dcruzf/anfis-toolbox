import numpy as np
import pytest

from anfis_toolbox.membership import BellMF


def test_bell_forward_shape_symmetry_and_range():
    """Forward: output shape, symmetry around c, value at center, and [0,1] range."""
    a, b, c = 2.0, 3.0, 1.0
    mf = BellMF(a=a, b=b, c=c)
    x = np.array([c - 4, c - 2, c - 1, c, c + 1, c + 2, c + 4], dtype=float)
    y = mf.forward(x)

    # Shape
    assert y.shape == x.shape

    # At center c, μ(c) = 1
    assert np.allclose(y[x == c], 1.0)

    # Symmetry: μ(c - d) == μ(c + d)
    for d in [1.0, 2.0, 4.0]:
        left = mf.forward(np.array([c - d]))[0]
        right = mf.forward(np.array([c + d]))[0]
        assert np.allclose(left, right)

    # Range [0, 1]
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_bell_parameter_validation():
    """Invalid parameters: a and b must be positive; c is unrestricted."""
    with pytest.raises(ValueError):
        BellMF(a=0.0, b=2.0, c=0.0)
    with pytest.raises(ValueError):
        BellMF(a=-1.0, b=2.0, c=0.0)
    with pytest.raises(ValueError):
        BellMF(a=1.0, b=0.0, c=0.0)
    with pytest.raises(ValueError):
        BellMF(a=1.0, b=-1.0, c=0.0)


def test_bell_b_steepness_effect():
    """Higher b yields a steeper curve (lower μ at same distance from center)."""
    a, c = 1.0, 0.0
    mf_gentle = BellMF(a=a, b=1.0, c=c)
    mf_steep = BellMF(a=a, b=5.0, c=c)
    x = np.array([c + 1.5])  # some distance from center
    y_gentle = mf_gentle.forward(x)[0]
    y_steep = mf_steep.forward(x)[0]
    assert y_steep < y_gentle


def test_bell_a_width_effect():
    """Larger a widens the curve (higher μ at same distance from center)."""
    b, c = 2.0, 0.0
    mf_narrow = BellMF(a=1.0, b=b, c=c)
    mf_wide = BellMF(a=3.0, b=b, c=c)
    x = np.array([c + 1.0])
    y_narrow = mf_narrow.forward(x)[0]
    y_wide = mf_wide.forward(x)[0]
    assert y_wide > y_narrow


def test_bell_backward_zero_at_center_and_nonzero_away():
    """Backward: gradients are zero at center-only input and non-zero for off-center points."""
    # Center-only input -> no valid_mask; gradients remain at 0.0
    mf_center = BellMF(a=2.0, b=3.0, c=1.0)
    x_center = np.array([1.0])
    mf_center.forward(x_center)
    mf_center.backward(np.ones_like(x_center))
    assert mf_center.gradients["a"] == 0.0
    assert mf_center.gradients["b"] == 0.0
    assert mf_center.gradients["c"] == 0.0

    # Off-center inputs -> gradients should be finite and not all zero
    mf = BellMF(a=2.0, b=3.0, c=0.0)
    x = np.array([-1.0, 1.0, 2.5])
    y = mf.forward(x)
    mf.backward(np.ones_like(y))
    grads = mf.gradients
    # Finite and not NaN
    for k in ["a", "b", "c"]:
        assert np.isfinite(grads[k])
    # At least one gradient should be non-zero
    assert any(abs(grads[k]) > 0 for k in grads)
