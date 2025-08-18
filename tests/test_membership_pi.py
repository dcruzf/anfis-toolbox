import numpy as np
import pytest

from anfis_toolbox.membership import PiMF


def test_pi_forward_regions_basic():
    """PiMF forward: outside=0, left foot=0, plateau=1, right foot=0, transitions in (0,1)."""
    mf = PiMF(a=-2.0, b=-1.0, c=1.0, d=2.0)
    x = np.array([-3.0, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0])
    y = mf.forward(x)

    assert y.shape == x.shape
    assert np.all((y >= 0.0) & (y <= 1.0))
    assert y[0] == 0.0  # outside left
    assert y[1] == 0.0  # at a
    assert y[3] == 1.0  # at b
    assert y[4] == 1.0  # inside plateau
    assert y[5] == 1.0  # at c
    assert y[7] == 0.0  # at d
    assert y[8] == 0.0  # outside right
    assert 0.0 < y[2] < 1.0  # rising edge
    assert 0.0 < y[6] < 1.0  # falling edge


def test_pi_parameter_validation_and_no_flat_region():
    """Validate ordering and behavior when b == c (no flat region allowed)."""
    # Valid ordering
    mf = PiMF(a=0.0, b=1.0, c=2.0, d=3.0)
    assert mf.parameters == {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}

    # No flat region (b == c) is valid
    mf2 = PiMF(a=0.0, b=1.0, c=1.0, d=2.0)
    y = mf2.forward(np.array([0.5, 1.0, 1.5]))
    assert 0.0 < y[0] < 1.0
    assert y[1] == 1.0
    assert 0.0 < y[2] < 1.0

    # Invalid orderings
    with pytest.raises(ValueError):
        PiMF(a=1.0, b=1.0, c=2.0, d=3.0)  # a < b violated
    with pytest.raises(ValueError):
        PiMF(a=0.0, b=2.0, c=1.0, d=3.0)  # b <= c violated
    with pytest.raises(ValueError):
        PiMF(a=0.0, b=1.0, c=3.0, d=3.0)  # c < d violated


@pytest.mark.parametrize("param_name", ["a", "b", "c", "d"])
def test_pi_numerical_gradients(param_name):
    """Numerical vs analytical gradients for PiMF parameters on mixed-region inputs."""
    base = {"a": -1.0, "b": 0.0, "c": 1.0, "d": 2.0}
    x = np.array([-0.5, 0.5, 1.5])  # rising, flat, falling

    eps = 1e-6
    params_plus = base.copy()
    params_minus = base.copy()
    params_plus[param_name] = base[param_name] + eps
    params_minus[param_name] = base[param_name] - eps

    try:
        mf_plus = PiMF(**params_plus)
        mf_minus = PiMF(**params_minus)
    except ValueError:
        eps = 1e-7
        params_plus[param_name] = base[param_name] + eps
        params_minus[param_name] = base[param_name] - eps
        mf_plus = PiMF(**params_plus)
        mf_minus = PiMF(**params_minus)

    y_plus = mf_plus.forward(x)
    y_minus = mf_minus.forward(x)
    num = np.sum((y_plus - y_minus) / (2 * eps))

    mf = PiMF(**base)
    # Reset grads without clearing caches
    for k in mf.gradients:
        mf.gradients[k] = 0.0
    y = mf.forward(x)
    mf.backward(np.ones_like(y))
    ana = mf.gradients[param_name]

    assert np.allclose(ana, num, atol=1e-4)


def test_pi_backward_regions_zero_and_nonzero():
    """Gradients non-zero on S/Z regions and zero on plateau."""
    mf = PiMF(a=-1.0, b=0.0, c=1.0, d=2.0)

    # Rising region -> grads for a,b only
    mf.forward(np.array([-0.5]))
    for k in mf.gradients:
        mf.gradients[k] = 0.0
    mf.backward(np.array([1.0]))
    assert mf.gradients["a"] != 0.0 and mf.gradients["b"] != 0.0
    assert mf.gradients["c"] == 0.0 and mf.gradients["d"] == 0.0

    # Plateau -> all zero
    mf.forward(np.array([0.5]))
    for k in mf.gradients:
        mf.gradients[k] = 0.0
    mf.backward(np.array([1.0]))
    assert all(v == 0.0 for v in mf.gradients.values())

    # Falling region -> grads for c,d only
    mf.forward(np.array([1.5]))
    for k in mf.gradients:
        mf.gradients[k] = 0.0
    mf.backward(np.array([1.0]))
    assert mf.gradients["c"] != 0.0 and mf.gradients["d"] != 0.0
    assert mf.gradients["a"] == 0.0 and mf.gradients["b"] == 0.0


def test_pi_forward_degenerate_branches():
    """Cover forward degenerate branches: b == a yields 1.0 at that point, d == c yields 0.0."""
    mf = PiMF(a=0.0, b=0.5, c=1.0, d=2.0)

    # Force b == a (post-construction to bypass validation)
    mf.parameters["b"] = mf.parameters["a"]
    a = mf.parameters["a"]
    y = mf.forward(np.array([a]))
    assert y[0] == 1.0

    # Force d == c
    mf.parameters["d"] = mf.parameters["c"]
    c = mf.parameters["c"]
    y2 = mf.forward(np.array([c]))
    assert y2[0] == 0.0


def test_pi_symmetry_and_reset():
    """Symmetry around center and reset clears caches and grads."""
    mf = PiMF(a=-2.0, b=-1.0, c=1.0, d=2.0)
    y_left = mf.forward(np.array([-1.5]))
    y_right = mf.forward(np.array([1.5]))
    assert np.allclose(y_left, y_right)

    # Exercise backward then reset
    mf.backward(np.array([1.0]))
    mf.reset()
    assert mf.last_input is None and mf.last_output is None
    assert all(v == 0.0 for v in mf.gradients.values())


def test_pi_backward_without_forward_is_noop():
    """Calling backward before any forward should be a no-op (early return guard)."""
    mf = PiMF(a=0.0, b=0.5, c=1.0, d=2.0)
    before = mf.gradients.copy()
    mf.backward(np.array([1.0]))
    assert mf.last_input is None and mf.last_output is None
    assert mf.gradients == before
