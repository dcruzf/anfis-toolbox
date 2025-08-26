import numpy as np

from anfis_toolbox.membership import DiffSigmoidalMF


def _fd_grad(mf: DiffSigmoidalMF, x: np.ndarray, param: str, eps: float = 1e-6) -> float:
    # Finite-difference gradient of L = sum(y)
    mf.forward(x)
    y0 = mf.last_output.copy()
    base = float(np.sum(y0))
    orig = mf.parameters[param]
    mf.parameters[param] = orig + eps
    y1 = mf.forward(x)
    mf.parameters[param] = orig
    return float((np.sum(y1) - base) / eps)


def test_diff_sigmoidal_forward_monotone_plateau():
    # For a1>0 and a2>0 with c1 < c2, function ~ plateau around (c1,c2)
    mf = DiffSigmoidalMF(a1=5.0, c1=-1.0, a2=5.0, c2=1.0)
    x = np.linspace(-4, 4, 201)
    y = mf.forward(x)
    # Outside far left near 0 and far right near 0; middle near 1
    assert y.min() >= -1e-6
    assert y.max() <= 1.0 + 1e-6
    assert float(y[0]) < 0.05
    assert float(y[-1]) < 0.05
    mid_idx = np.argmin(np.abs(x - 0.0))
    assert float(y[mid_idx]) > 0.8


def test_diff_sigmoidal_backward_matches_fd():
    rng = np.random.RandomState(0)
    x = rng.uniform(-3, 3, size=257)
    mf = DiffSigmoidalMF(a1=1.7, c1=-0.3, a2=1.2, c2=0.9)
    mf.forward(x)
    dL_dy = np.ones_like(x)
    dx = mf.backward(dL_dy)
    # Parameter grads vs finite diff
    for p in ("a1", "c1", "a2", "c2"):
        fd = _fd_grad(mf, x, p)
        an = mf.gradients[p]
        if abs(fd) > 1e-6:
            assert np.isclose(an, fd, rtol=1e-3, atol=1e-5), f"{p}: {an} vs {fd}"
        else:
            assert abs(an - fd) < 1e-6
    # Return shape matches input
    assert isinstance(dx, np.ndarray)
    assert dx.shape == x.shape


def test_diff_sigmoidal_backward_without_forward_is_safe():
    mf = DiffSigmoidalMF(a1=1.0, c1=0.0, a2=1.0, c2=1.0)
    # No forward called
    out = mf.backward(np.ones(5))
    assert out is None
    assert all(v == 0.0 for v in mf.gradients.values())


def test_diff_sigmoidal_accumulation_and_reset():
    x = np.linspace(-2, 2, 33)
    mf = DiffSigmoidalMF(a1=2.0, c1=-0.5, a2=1.5, c2=0.6)
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
