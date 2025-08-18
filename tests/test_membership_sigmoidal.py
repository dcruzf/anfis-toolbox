import numpy as np
import pytest

from anfis_toolbox.membership import SigmoidalMF


def test_sigmoid_forward_center_limits_and_range():
    """Forward: μ(c)=0.5, limits approach 0/1, and values stay in [0,1]."""
    mf = SigmoidalMF(a=2.0, c=1.0)
    x = np.array([-10.0, 1.0, 10.0])
    y = mf.forward(x)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)
    # At center c: 0.5
    assert np.allclose(y[1], 0.5, atol=1e-12)
    # Far left/right limits
    assert y[0] < 1e-3
    assert y[2] > 1 - 1e-3


def test_sigmoid_monotonicity_positive_a():
    """For a>0, output is monotonically increasing in x."""
    mf = SigmoidalMF(a=1.5, c=0.0)
    x = np.linspace(-5, 5, 21)
    y = mf.forward(x)
    diffs = np.diff(y)
    assert np.all(diffs >= -1e-12)


def test_sigmoid_monotonicity_negative_a():
    """For a<0, output is monotonically decreasing in x."""
    mf = SigmoidalMF(a=-1.5, c=0.0)
    x = np.linspace(-5, 5, 21)
    y = mf.forward(x)
    diffs = np.diff(y)
    assert np.all(diffs <= 1e-12)


def test_sigmoid_parameter_validation():
    """Parameter 'a' cannot be zero."""
    with pytest.raises(ValueError):
        SigmoidalMF(a=0.0, c=0.0)


def test_sigmoid_backward_gradients_center_only():
    """At x=c, dL/da=0 due to (x-c) term; dL/dc is finite and equals -a μ(1-μ)."""
    a, c = 2.0, 1.0
    mf = SigmoidalMF(a=a, c=c)
    x = np.array([c])
    y = mf.forward(x)
    assert np.allclose(y, 0.5)
    mf.backward(np.ones_like(y))
    # dL/da = μ(1-μ)(x-c) = 0
    assert mf.gradients["a"] == pytest.approx(0.0, abs=1e-12)
    # dL/dc = -a μ(1-μ) = -a*0.25 = -0.5
    assert mf.gradients["c"] == pytest.approx(-a * 0.25)


def test_sigmoid_backward_signs_single_point_positive_a():
    """For a>0 and x>c: dL/da>0 and dL/dc<0 (with dL/dy=1)."""
    a, c = 1.2, 0.0
    mf = SigmoidalMF(a=a, c=c)
    x = np.array([c + 1.0])
    y = mf.forward(x)
    mf.backward(np.ones_like(y))
    assert mf.gradients["a"] > 0.0
    assert mf.gradients["c"] < 0.0


def test_sigmoid_backward_signs_single_point_negative_a():
    """For a<0 and x>c: dL/da>0 and dL/dc>0 (with dL/dy=1)."""
    a, c = -1.2, 0.0
    mf = SigmoidalMF(a=a, c=c)
    x = np.array([c + 1.0])
    y = mf.forward(x)
    mf.backward(np.ones_like(y))
    assert mf.gradients["a"] > 0.0
    assert mf.gradients["c"] > 0.0
