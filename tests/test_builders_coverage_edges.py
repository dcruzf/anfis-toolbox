import numpy as np
import pytest

from anfis_toolbox.builders import ANFISBuilder
from anfis_toolbox.membership import PiMF, SShapedMF, ZShapedMF


def _get_mfs(model, name="x"):
    return model.input_mfs[name]


def test_fcm_unknown_type_raises():
    data = np.linspace(-1, 1, 50)
    b = ANFISBuilder()
    with pytest.raises(ValueError):
        b.add_input_from_data("x", data, n_mfs=3, mf_type="unknown_kind", init="fcm").build()


def test_fcm_too_few_samples_raises():
    data = np.array([0.0, 1.0])  # 2 samples
    b = ANFISBuilder()
    with pytest.raises(ValueError):
        b.add_input_from_data("x", data, n_mfs=3, mf_type="gaussian", init="fcm").build()


def test_grid_sshape_zero_range_fallback():
    # zero-range triggers fallback tiny-span path inside _create_sshape_mfs
    model = ANFISBuilder().add_input("x", 2.0, 2.0, n_mfs=2, mf_type="sshape").build()
    mfs = _get_mfs(model)
    assert all(isinstance(m, SShapedMF) for m in mfs)
    for m in mfs:
        a, b = m.parameters["a"], m.parameters["b"]
        assert a < b and (b - a) < 1e-3


def test_grid_zshape_zero_range_fallback():
    model = ANFISBuilder().add_input("x", -3.0, -3.0, n_mfs=1, mf_type="zshape").build()
    m = _get_mfs(model)[0]
    assert isinstance(m, ZShapedMF)
    a, b = m.parameters["a"], m.parameters["b"]
    assert a < b and (b - a) < 1e-3


def test_grid_pi_zero_range_fallback():
    model = ANFISBuilder().add_input("x", 0.0, 0.0, n_mfs=1, mf_type="pi").build()
    m = _get_mfs(model)[0]
    assert isinstance(m, PiMF)
    a, b, c, d = m.parameters["a"], m.parameters["b"], m.parameters["c"], m.parameters["d"]
    assert a < b <= c < d
    # Tiny span created by fallback around the center
    assert (d - a) < 1e-2


def test_grid_clamp_edges_for_linear_shapes():
    # values near edges to trigger clamp logic branches (a,b clamped to range)
    model = ANFISBuilder().add_input("x", 0.0, 1.0, n_mfs=3, mf_type="linsshape", overlap=1.0).build()
    for m in model.input_mfs["x"]:
        a, b = m.parameters["a"], m.parameters["b"]
        assert 0.0 <= a < b <= 1.0

    model2 = ANFISBuilder().add_input("x", 0.0, 1.0, n_mfs=3, mf_type="linzshape", overlap=1.0).build()
    for m in model2.input_mfs["x"]:
        a, b = m.parameters["a"], m.parameters["b"]
        assert 0.0 <= a < b <= 1.0
