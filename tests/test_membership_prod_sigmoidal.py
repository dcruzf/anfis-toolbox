import numpy as np

from anfis_toolbox.membership import ProdSigmoidalMF


def _fd_grad(mf: ProdSigmoidalMF, x: np.ndarray, param: str, eps: float = 1e-6) -> float:
    mf.forward(x)
    base = float(np.sum(mf.last_output))
    orig = mf.parameters[param]
    mf.parameters[param] = orig + eps
    y1 = mf.forward(x)
    mf.parameters[param] = orig
    return float((np.sum(y1) - base) / eps)


def test_prod_sigmoidal_forward_shape():
    # Product of two sigmoids yields a bump-like shape when centers differ
    mf = ProdSigmoidalMF(a1=4.0, c1=-1.0, a2=4.0, c2=1.0)
    x = np.linspace(-4, 4, 201)
    y = mf.forward(x)
    assert y.min() >= 0.0 - 1e-6
    assert y.max() <= 1.0 + 1e-6
    # Low on far left and high on far right for increasing sigmoids
    assert float(y[0]) < 0.05
    assert float(y[-1]) > 0.95
    # Monotone rising: middle between ends
    mid_idx = np.argmin(np.abs(x - 0.0))
    assert float(y[0]) < float(y[mid_idx]) < float(y[-1])


def test_prod_sigmoidal_backward_without_forward_is_safe():
    mf = ProdSigmoidalMF(a1=1.0, c1=0.0, a2=1.0, c2=1.0)
    out = mf.backward(np.ones(5))
    assert out is None
    assert all(v == 0.0 for v in mf.gradients.values())


def test_prod_sigmoidal_accumulation_and_reset():
    x = np.linspace(-2, 2, 33)
    mf = ProdSigmoidalMF(a1=2.0, c1=-0.5, a2=1.5, c2=0.6)
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
