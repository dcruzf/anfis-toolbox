import numpy as np
import pytest

from anfis_toolbox.membership import SShapedMF


def test_s_forward_regions_and_range():
    mf = SShapedMF(a=0.0, b=1.0)
    x = np.array([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
    y = mf.forward(x)

    # Left of a -> 0, at a -> 0
    assert y[0] == pytest.approx(0.0)
    assert y[1] == pytest.approx(0.0)

    # Transition strictly between a and b is (0,1)
    assert 0.0 < y[2] < 1.0
    assert 0.0 < y[3] < 1.0
    assert 0.0 < y[4] < 1.0

    # At/after b -> 1
    assert y[5] == pytest.approx(1.0)
    assert y[6] == pytest.approx(1.0)

    # Monotonic non-decreasing
    assert np.all(np.diff(y) >= -1e-12)


def test_s_validation_and_reset():
    with pytest.raises(ValueError):
        SShapedMF(a=1.0, b=1.0)
    with pytest.raises(ValueError):
        SShapedMF(a=2.0, b=1.0)

    mf = SShapedMF(a=0.0, b=2.0)
    x = np.linspace(-1, 3, 9)
    _ = mf.forward(x)
    mf.reset()
    assert mf.last_input is None and mf.last_output is None


def test_s_backward_gradients_match_numerical():
    rng = np.random.default_rng(0)
    a, b = -1.0, 2.0
    mf = SShapedMF(a=a, b=b)
    x = np.linspace(-2, 3, 200)
    _y = mf.forward(x)

    # Random upstream gradient
    dL_dy = rng.normal(size=x.shape)
    mf.backward(dL_dy)

    # Numerical gradients wrt a,b via finite differences
    eps = 1e-6

    # a
    mf_a = SShapedMF(a=a + eps, b=b)
    y1 = mf_a.forward(x)
    mf_a = SShapedMF(a=a - eps, b=b)
    y2 = mf_a.forward(x)
    num_da = np.sum(dL_dy * (y1 - y2) / (2 * eps))

    # b
    mf_b = SShapedMF(a=a, b=b + eps)
    y1 = mf_b.forward(x)
    mf_b = SShapedMF(a=a, b=b - eps)
    y2 = mf_b.forward(x)
    num_db = np.sum(dL_dy * (y1 - y2) / (2 * eps))

    assert mf.gradients["a"] == pytest.approx(num_da, rel=1e-4, abs=1e-6)
    assert mf.gradients["b"] == pytest.approx(num_db, rel=1e-4, abs=1e-6)


def test_s_backward_early_return_without_forward():
    mf = SShapedMF(a=0.0, b=1.0)
    # No forward -> gradients remain zero
    mf.backward(np.ones(5))
    assert mf.gradients == {"a": 0.0, "b": 0.0}


def test_s_forward_no_transition_branch():
    """Forward should skip transition block when no points are between a and b."""
    mf = SShapedMF(a=1.0, b=2.0)
    x = np.array([-5.0, -1.0, 0.0, 1.0, 2.0, 3.0])  # points only at boundaries or outside
    y = mf.forward(x)
    # All <= a are 0, >= b are 1; boundaries are 0 at a and 1 at b
    assert y.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_s_backward_early_return_no_transition_points():
    """Backward should early-return if no points fall in [a, b] mask (including boundaries absent)."""
    mf = SShapedMF(a=0.0, b=1.0)
    x = np.array([-2.0, -1.0, 2.0, 3.0])  # no values within [a,b]
    _ = mf.forward(x)
    mf.backward(np.ones_like(x))
    assert mf.gradients == {"a": 0.0, "b": 0.0}
