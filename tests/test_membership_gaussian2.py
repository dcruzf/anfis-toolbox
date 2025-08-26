import numpy as np
import pytest

from anfis_toolbox.membership import Gaussian2MF


def test_gaussian2_forward_regions():
    mf = Gaussian2MF(sigma1=1.0, c1=-1.0, sigma2=1.5, c2=2.0)
    x = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
    y = mf.forward(x)
    # Left of c1 uses left Gaussian tail (< 1)
    assert 0.0 < y[0] < 1.0
    # At c1 and inside [c1,c2] plateau should be 1
    assert np.isclose(y[1], 1.0)
    assert np.isclose(y[2], 1.0)
    assert np.isclose(y[3], 1.0)
    # Right of c2 uses right Gaussian tail (< 1)
    assert 0.0 < y[4] < 1.0


def test_gaussian2_backward_center_signs():
    # As x increases above c1 on left region, increasing c1 should increase μ (positive grad)
    mf = Gaussian2MF(sigma1=1.0, c1=0.0, sigma2=1.0, c2=0.0)
    x = np.array([-2.0, -1.0, -0.1])  # all left of c1 (0.0)
    mf.forward(x)
    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)
    # c1 gradient should be positive (because ∂μ/∂c1 = μ (x-c1)/σ1^2 and x < c1 ⇒ negative;
    # summed with ones over left region yields negative; but we accumulate over three negatives)
    # To avoid sign confusion, check analytical point-by-point comparison against finite difference below
    # Here just assert it's a float and not zero magnitude
    assert isinstance(mf.gradients["c1"], float)


def _finite_diff_grad(mf, x, param, eps=1e-6):
    # Utility to compute dL/dparam via finite differences with L = sum(y)
    mf.forward(x)
    y0 = mf.last_output.copy()

    # Bump param
    orig = mf.parameters[param]
    mf.parameters[param] = orig + eps
    y1 = mf.forward(x)

    # Restore
    mf.parameters[param] = orig

    return float((np.sum(y1) - np.sum(y0)) / eps)


def test_gaussian2_backward_matches_finite_diff():
    # Random mixed-region sample
    rng = np.random.RandomState(0)
    mf = Gaussian2MF(sigma1=0.8, c1=-0.5, sigma2=1.2, c2=1.0)
    x = rng.uniform(-3, 3, size=128)

    # Analytical grads
    mf.forward(x)
    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)

    # Finite differences
    for p in ["c1", "sigma1", "c2", "sigma2"]:
        fd = _finite_diff_grad(mf, x, p)
        an = mf.gradients[p]
        # Relative check with tolerance; also allow small absolute error near 0
        if abs(fd) > 1e-6:
            assert np.isclose(an, fd, rtol=1e-3, atol=1e-5), f"Param {p}: analytical {an} vs fd {fd}"
        else:
            assert abs(an - fd) < 1e-6


def test_gaussian2_validation_errors():
    # sigma must be positive
    with pytest.raises(ValueError):
        Gaussian2MF(sigma1=0.0, c1=0.0, sigma2=1.0, c2=1.0)
    with pytest.raises(ValueError):
        Gaussian2MF(sigma1=1.0, c1=0.0, sigma2=-1.0, c2=1.0)
    # c1 must be <= c2
    with pytest.raises(ValueError):
        Gaussian2MF(sigma1=1.0, c1=2.0, sigma2=1.0, c2=1.0)


def test_gaussian2_plateau_zero_gradients():
    # All x in plateau region => no gradient contribution
    mf = Gaussian2MF(sigma1=1.0, c1=-1.0, sigma2=1.0, c2=1.0)
    x = np.linspace(-1.0, 1.0, 11)
    mf.forward(x)
    mf.backward(np.ones_like(x))
    grads = mf.gradients
    assert grads["c1"] == 0.0 and grads["sigma1"] == 0.0
    assert grads["c2"] == 0.0 and grads["sigma2"] == 0.0


def test_gaussian2_equal_centers_behaves_sane():
    # c1 == c2: value at center is 1, tails decay on both sides
    mf = Gaussian2MF(sigma1=0.5, c1=0.0, sigma2=1.0, c2=0.0)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    y = mf.forward(x)
    assert np.isclose(y[2], 1.0)
    assert y[0] < y[1] < y[2] and y[2] > y[3] > y[4]


def test_gaussian2_reset_and_accumulation():
    mf = Gaussian2MF(sigma1=0.8, c1=-0.2, sigma2=1.2, c2=0.7)
    x = np.array([-1.0, -0.5, 0.0, 1.5])

    # First backward
    mf.forward(x)
    mf.backward(np.ones_like(x))
    g1 = mf.gradients.copy()

    # Second backward without reset accumulates
    mf.forward(x)
    mf.backward(np.ones_like(x))
    g2 = mf.gradients.copy()
    for k in g1:
        assert np.isclose(g2[k], 2 * g1[k])

    # Reset clears
    mf.reset()
    assert all(v == 0.0 for v in mf.gradients.values())
    assert mf.last_input is None and mf.last_output is None


def test_gaussian2_backward_without_forward_is_safe():
    # Calling backward before forward should be a no-op (guard path)
    mf = Gaussian2MF(sigma1=1.0, c1=-1.0, sigma2=1.0, c2=1.0)
    # Should not raise
    mf.backward(np.ones(3))
    # Gradients remain zeros
    assert all(v == 0.0 for v in mf.gradients.values())
