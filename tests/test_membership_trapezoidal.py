import numpy as np
import pytest

from anfis_toolbox.membership import TrapezoidalMF


def test_trapezoidal_forward_shape_and_values():
    """Test the shape and values of the trapezoidal membership function."""
    mf = TrapezoidalMF(a=-1.0, b=0.0, c=1.0, d=2.0)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = mf.forward(x)

    # Check the shape of the output
    assert y.shape == x.shape, "Output shape should match input shape"

    # Check specific values
    assert np.allclose(y[0], 0.0), "Left of the trapezoid should be 0"
    assert np.allclose(y[1], 0.0), "At a boundary exactly is 0 per implementation (strict > a)"
    assert np.allclose(y[2], 1.0), "Within plateau should be 1.0"
    assert np.allclose(y[3], 1.0), "Within plateau should be 1.0"
    assert np.allclose(y[4], 0.0), "At d boundary is 0"


def test_trapezoidal_forward_slopes_and_plateau():
    """Verify forward outputs across left slope, plateau, right slope, and boundaries."""
    mf = TrapezoidalMF(a=0.0, b=1.0, c=3.0, d=4.0)
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0])
    y = mf.forward(x)
    expected = np.array(
        [
            0.0,  # left of a
            0.0,  # at a (strict > a handled)
            0.5,  # rising slope (x-a)/(b-a) = 0.5
            1.0,  # at b -> plateau
            1.0,  # inside plateau
            1.0,  # at c -> plateau
            0.5,  # falling slope (d-x)/(d-c) = (4-3.5)/1 = 0.5
            0.0,  # at d -> 0
            0.0,  # right of d
        ]
    )
    assert np.allclose(y, expected)


def test_trapezoidal_init_validation():
    """Validate parameter ordering and zero-width checks for TrapezoidalMF."""
    # invalid ordering
    with pytest.raises(ValueError):
        TrapezoidalMF(2.0, 1.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        TrapezoidalMF(0.0, 1.0, 4.0, 3.0)
    # zero width a==d
    with pytest.raises(ValueError):
        TrapezoidalMF(1.0, 1.0, 1.0, 1.0)


def test_trapezoidal_backward_left_slope_gradients():
    """Check accumulated gradients on the left slope region (a < x < b)."""
    # Choose (b-a)=1 for easy math
    mf = TrapezoidalMF(a=0.0, b=1.0, c=2.0, d=3.0)
    # Include boundaries a,b (no gradient) + two interior left-slope points
    x = np.array([0.0, 0.25, 0.75, 1.0])
    mf.forward(x)
    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)

    # On left slope with (b-a)=1:
    # Implementation uses ∂μ/∂a = -1/(b-a) constant per interior point -> two points => -2.0 total
    # and ∂μ/∂b = -(x - a) / (b-a)^2 = -x ; points: 0.25->-0.25, 0.75->-0.75 => sum -1.0
    assert mf.gradients["a"] == pytest.approx(-2.0)
    assert mf.gradients["b"] == pytest.approx(-1.0)
    assert mf.gradients["c"] == pytest.approx(0.0)
    assert mf.gradients["d"] == pytest.approx(0.0)


def test_trapezoidal_backward_right_slope_gradients():
    """Check accumulated gradients on the right slope region (c < x < d)."""
    # Choose (d-c)=2 for visible fractions
    mf = TrapezoidalMF(a=0.0, b=1.0, c=2.0, d=4.0)
    # Include boundaries c,d (no gradient) + two interior right-slope points
    x = np.array([2.0, 3.0, 3.5, 4.0])
    mf.forward(x)
    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)

    # On right slope with (d-c)=2, squared=4:
    # Implementation uses ∂μ/∂c = (x - d)/(d-c)^2 -> (3.0-4)/4 = -0.25, (3.5-4)/4 = -0.125 => sum -0.375
    # and ∂μ/∂d = (x - c)/(d-c)^2 -> (3.0-2.0)/4 = 0.25, (3.5-2.0)/4 = 0.375 => sum 0.625
    assert mf.gradients["a"] == pytest.approx(0.0)
    assert mf.gradients["b"] == pytest.approx(0.0)
    assert mf.gradients["c"] == pytest.approx(-0.375)
    assert mf.gradients["d"] == pytest.approx(0.625)


def test_trapezoidal_forward_left_slope_block():
    """Covers the left-slope branch: a < x < b -> (x-a)/(b-a)."""
    mf = TrapezoidalMF(a=0.0, b=2.0, c=3.0, d=4.0)
    x = np.array([1.0])  # strictly inside (a, b)
    y = mf.forward(x)
    assert y.shape == x.shape
    assert y[0] == pytest.approx((1.0 - 0.0) / (2.0 - 0.0))


def test_trapezoidal_forward_no_left_slope_when_b_eq_a():
    """Covers the false branch of `if b > a` by setting b == a."""
    mf = TrapezoidalMF(a=1.0, b=1.0, c=3.0, d=4.0)
    # Values across regions: left of a, at plateau start, inside plateau
    x = np.array([0.5, 1.0, 2.0])
    y = mf.forward(x)
    # Left of a stays 0; b==a so no left slope is computed; plateau from b..c is 1
    assert np.allclose(y, [0.0, 1.0, 1.0])


def test_trapezoidal_forward_plateau_boundaries_and_interior():
    """Covers plateau assignment for b <= x <= c, including boundaries."""
    mf = TrapezoidalMF(a=0.0, b=1.0, c=3.0, d=5.0)
    x = np.array([1.0, 2.0, 3.0])  # b, interior, c
    y = mf.forward(x)
    assert np.allclose(y, [1.0, 1.0, 1.0])


def test_trapezoidal_forward_right_slope_block():
    """Covers the right-slope branch: c < x < d -> (d-x)/(d-c)."""
    mf = TrapezoidalMF(a=0.0, b=1.0, c=3.0, d=5.0)
    x = np.array([4.0])  # strictly inside (c, d)
    y = mf.forward(x)
    assert y.shape == x.shape
    assert y[0] == pytest.approx((5.0 - 4.0) / (5.0 - 3.0))  # 0.5


def test_trapezoidal_forward_no_right_slope_when_d_eq_c():
    """Covers the false branch of `if d > c` by setting d == c."""
    mf = TrapezoidalMF(a=0.0, b=1.0, c=3.0, d=3.0)
    x = np.array([2.5, 3.0, 3.5])  # left of plateau end, at c==d, right of d
    y = mf.forward(x)
    # 2.5 in plateau -> 1.0; 3.0 at c==d still plateau -> 1.0; 3.5 > d -> 0.0
    assert np.allclose(y, [1.0, 1.0, 0.0])


def test_trapezoidal_backward_no_slope_points_boundaries_only():
    """Covers false branches in backward when no points fall in (a,b) or (c,d)."""
    mf = TrapezoidalMF(a=0.0, b=1.0, c=2.0, d=3.0)
    # Only boundaries: a, b, c, d
    x = np.array([0.0, 1.0, 2.0, 3.0])
    mf.forward(x)
    mf.backward(np.ones_like(x))
    # With no interior slope points, all gradients remain zero
    assert mf.gradients["a"] == pytest.approx(0.0)
    assert mf.gradients["b"] == pytest.approx(0.0)
    assert mf.gradients["c"] == pytest.approx(0.0)
    assert mf.gradients["d"] == pytest.approx(0.0)


def test_trapezoidal_backward_degenerate_no_slopes_when_b_eq_a_and_d_eq_c():
    """Covers outer guards in backward when b == a and d == c (no slopes at all)."""
    mf = TrapezoidalMF(a=1.0, b=1.0, c=3.0, d=3.0)
    # Mix of points: left of a, at plateau start, interior plateau, at c==d, right of d
    x = np.array([0.5, 1.0, 2.0, 3.0, 3.5])
    mf.forward(x)
    mf.backward(np.ones_like(x))
    # With both slopes disabled, gradients must remain zero
    assert mf.gradients["a"] == pytest.approx(0.0)
    assert mf.gradients["b"] == pytest.approx(0.0)
    assert mf.gradients["c"] == pytest.approx(0.0)
    assert mf.gradients["d"] == pytest.approx(0.0)
