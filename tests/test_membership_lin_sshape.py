import numpy as np
import pytest

from anfis_toolbox.membership import LinSShapedMF


def _fd_grad(mf: LinSShapedMF, x: np.ndarray, param: str, eps: float = 1e-6) -> float:
    mf.forward(x)
    base = np.sum(mf.last_output)
    orig = mf.parameters[param]
    mf.parameters[param] = orig + eps
    y1 = mf.forward(x)
    mf.parameters[param] = orig
    return float((np.sum(y1) - base) / eps)


def test_linsshape_forward_regions():
    mf = LinSShapedMF(a=-1.0, b=1.0)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = mf.forward(x)
    assert np.isclose(y[0], 0.0)
    assert np.isclose(y[1], 0.0)
    assert 0.0 < y[2] < 1.0
    assert np.isclose(y[3], 1.0)
    assert np.isclose(y[4], 1.0)


def test_linsshape_backward_matches_fd():
    rng = np.random.RandomState(0)
    mf = LinSShapedMF(a=-0.5, b=0.8)
    x = rng.uniform(-2, 2, size=257)
    mf.forward(x)
    mf.backward(np.ones_like(x))
    for p in ("a", "b"):
        fd = _fd_grad(mf, x, p)
        an = mf.gradients[p]
        if abs(fd) > 1e-6:
            assert np.isclose(an, fd, rtol=1e-3, atol=1e-5)
        else:
            assert abs(an - fd) < 1e-6


def test_linsshape_validation_and_guards():
    with pytest.raises(ValueError):
        LinSShapedMF(a=1.0, b=0.0)
    # backward before forward is safe
    mf = LinSShapedMF(a=0.0, b=1.0)
    mf.backward(np.ones(3))
    assert all(v == 0.0 for v in mf.gradients.values())


def test_linsshape_accumulation_and_reset():
    mf = LinSShapedMF(a=-1.0, b=1.0)
    x = np.linspace(-2, 2, 33)
    mf.forward(x)
    mf.backward(np.ones_like(x))
    g1 = mf.gradients.copy()
    mf.forward(x)
    mf.backward(np.ones_like(x))
    g2 = mf.gradients.copy()
    for k in g1:
        assert np.isclose(g2[k], 2 * g1[k])
    mf.reset()
    assert all(v == 0.0 for v in mf.gradients.values())
    assert mf.last_input is None and mf.last_output is None


def test_linsshape_backward_no_mid_region_is_noop():
    # All x on left of a
    mf = LinSShapedMF(a=0.0, b=1.0)
    x = np.array([-2.0, -1.0, -0.1])
    mf.forward(x)
    mf.backward(np.ones_like(x))
    assert all(v == 0.0 for v in mf.gradients.values())
    mf.reset()
    # All x on right of b
    x2 = np.array([1.1, 2.0, 3.0])
    mf.forward(x2)
    mf.backward(np.ones_like(x2))
    assert all(v == 0.0 for v in mf.gradients.values())


def test_linsshape_backward_d_equals_zero_guard():
    # Force a==b to hit d==0 guard branch safely
    mf = LinSShapedMF(a=0.0, b=1.0)
    mf.parameters["b"] = mf.parameters["a"]
    x = np.linspace(-1, 1, 5)
    mf.forward(x)
    mf.backward(np.ones_like(x))  # should no-op without errors
    assert all(v == 0.0 for v in mf.gradients.values())
