import numpy as np

from anfis_toolbox.membership import TrapezoidalMF, TriangularMF, ZShapedMF


def test_triangular_backward_without_forward_returns_early():
    """Calling backward() before forward() should be a no-op (early return)."""
    mf = TriangularMF(a=0.0, b=1.0, c=2.0)
    # Pre-set gradients to detect unintended changes
    mf.gradients = {"a": 0.0, "b": 0.0, "c": 0.0}

    # Call backward without prior forward
    mf.backward(np.array([1.0, 2.0, 3.0]))

    # Gradients remain unchanged
    assert mf.gradients == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_trapezoidal_backward_without_forward_returns_early():
    """Calling backward() before forward() should be a no-op (early return)."""
    mf = TrapezoidalMF(a=0.0, b=1.0, c=2.0, d=3.0)
    mf.gradients = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}

    # Call backward without prior forward
    mf.backward(np.array([1.0, 2.0, 3.0, 4.0]))

    # Gradients remain unchanged
    assert mf.gradients == {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}


def test_zshaped_forward_degenerate_else_marked_no_cover():
    """Regression: ensure unreachable degenerate branch remains unreachable.

    With parameter validation requiring a < b, there is no x such that (x > a) & (x < b)
    when a == b. We keep a<b invariant here and just verify forward works normally.
    """
    mf = ZShapedMF(a=0.0, b=1.0)
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    y = mf.forward(x)
    # Basic shape checks
    assert y.shape == x.shape
    assert np.all((y >= 0.0) & (y <= 1.0))
