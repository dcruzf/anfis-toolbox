import numpy as np

from anfis_toolbox.membership import GaussianMF


def test_gaussian_forward_center_peak():
    """Test that the Gaussian MF outputs 1.0 at its mean."""
    mf = GaussianMF(mean=0.0, sigma=1.0)
    x = np.array([0.0])
    y = mf.forward(x)
    assert np.allclose(y, 1.0), "Gaussian MF should output 1.0 at the mean"


def test_gaussian_forward_symmetric_decay():
    """Test that values at equal distance from mean have equal membership."""
    mf = GaussianMF(mean=0.0, sigma=1.0)
    x = np.array([-1.0, 1.0])
    y = mf.forward(x)
    assert np.allclose(y[0], y[1]), "Gaussian should be symmetric around mean"


def test_gaussian_forward_zero_sigma_behavior():
    """Test that sigma=0 produces correct handling (very sharp peak)."""
    mf = GaussianMF(mean=0.0, sigma=1e-9)  # Very small sigma
    x = np.array([-1.0, 0.0, 1.0])
    y = mf.forward(x)
    assert y[1] > y[0] and y[1] > y[2], "Peak should be at the mean"


def test_gaussian_backward_gradients_sign_and_shape():
    """Test that backward produces gradients of correct sign and shape."""
    mf = GaussianMF(mean=0.0, sigma=1.0)
    x = np.array([-1.0, 0.0, 1.0])
    mf.forward(x)
    dL_dy = np.ones_like(x)
    mf.backward(dL_dy)

    grads = mf.gradients
    assert "mean" in grads and "sigma" in grads, "Both gradients must be computed"
    assert np.isclose(grads["mean"], 0.0), "Gradient wrt mean should be zero for symmetric input with uniform dL/dy"
    assert isinstance(grads["sigma"], float), "Sigma gradient should be a float"


def test_gaussian_reset_clears_state():
    """Test that reset clears gradients and cached input/output."""
    mf = GaussianMF(mean=0.0, sigma=1.0)
    mf.forward(np.array([0.0]))
    mf.backward(np.array([1.0]))
    mf.reset()

    assert all(v == 0.0 for v in mf.gradients.values()), "Gradients should be reset to zero"
    assert mf.last_input is None and mf.last_output is None, "Internal state should be cleared"
