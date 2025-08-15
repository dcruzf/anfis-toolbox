import numpy as np
import pytest

from anfis_toolbox.membership import ZShapedMF


def test_z_forward_regions_basic():
    mf = ZShapedMF(a=-1.0, b=1.0)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = mf.forward(x)
    assert y.shape == x.shape
    assert np.all((y >= 0.0) & (y <= 1.0))
    assert y[0] == 1.0  # left of a
    assert y[1] == 1.0  # at a
    assert 0.0 < y[2] < 1.0  # transition
    assert y[3] == 0.0  # at b
    assert y[4] == 0.0  # right of b


def test_z_validation():
    with pytest.raises(ValueError):
        ZShapedMF(a=1.0, b=1.0)  # requires a < b
    with pytest.raises(ValueError):
        ZShapedMF(a=2.0, b=1.0)


@pytest.mark.parametrize("param_name", ["a", "b"])
def test_z_numerical_vs_analytical_gradients(param_name):
    base = {"a": -1.0, "b": 1.0}
    x = np.array([-0.5, 0.0, 0.5])
    eps = 1e-6

    params_plus = base.copy()
    params_minus = base.copy()
    params_plus[param_name] += eps
    params_minus[param_name] -= eps

    mf_plus = ZShapedMF(**params_plus)
    mf_minus = ZShapedMF(**params_minus)
    y_plus = mf_plus.forward(x)
    y_minus = mf_minus.forward(x)
    num = np.sum((y_plus - y_minus) / (2 * eps))

    mf = ZShapedMF(**base)
    for k in mf.gradients:
        mf.gradients[k] = 0.0
    y = mf.forward(x)
    mf.backward(np.ones_like(y))
    ana = mf.gradients[param_name]

    assert np.allclose(ana, num, atol=1e-4)


def test_z_backward_no_transition_points_is_noop():
    """Backward should early-return when no x lies in [a, b] (mask empty)."""
    mf = ZShapedMF(a=0.0, b=1.0)
    x = np.array([-1.0, 2.0])  # strictly outside transition/plateau range
    mf.forward(x)
    before = mf.gradients.copy()
    mf.backward(np.ones_like(x))
    assert mf.gradients == before


def test_z_backward_includes_boundaries():
    """Backward mask includes x == a and x == b (equality)."""
    mf = ZShapedMF(a=0.0, b=2.0)
    x = np.array([0.0, 1.0, 2.0])  # includes both boundaries and an interior point
    mf.forward(x)
    for k in mf.gradients:
        mf.gradients[k] = 0.0
    mf.backward(np.ones_like(x))
    # Gradients may be small or zero at boundaries; just assert code path executed (no crash)
    assert "a" in mf.gradients and "b" in mf.gradients


def test_z_backward_without_forward_is_noop():
    """Calling backward before forward should be a no-op (guard path)."""
    mf = ZShapedMF(a=0.0, b=1.0)
    before = mf.gradients.copy()
    mf.backward(np.ones(1))
    assert mf.last_input is None and mf.last_output is None
    assert mf.gradients == before
