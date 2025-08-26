import numpy as np

from anfis_toolbox.builders import ANFISBuilder
from anfis_toolbox.membership import (
    DiffSigmoidalMF,
    LinSShapedMF,
    LinZShapedMF,
    ProdSigmoidalMF,
)


def _get_single_input_mfs(model, name="x"):
    assert name in model.input_mfs
    return model.input_mfs[name]


def test_builder_grid_linear_shapes():
    b = ANFISBuilder()
    b.add_input("x", 0.0, 10.0, n_mfs=3, mf_type="linsshape", overlap=0.5)
    model = b.build()
    mfs = _get_single_input_mfs(model, "x")
    assert len(mfs) == 3
    assert all(isinstance(m, LinSShapedMF) for m in mfs)
    for m in mfs:
        a, b_ = m.parameters["a"], m.parameters["b"]
        assert a < b_
        # in-range clamp
        assert 0.0 <= a < b_ <= 10.0

    b2 = ANFISBuilder().add_input("x", -1.0, 1.0, n_mfs=4, mf_type="linzshape", overlap=0.3)
    model2 = b2.build()
    mfs2 = _get_single_input_mfs(model2, "x")
    assert len(mfs2) == 4
    assert all(isinstance(m, LinZShapedMF) for m in mfs2)
    for m in mfs2:
        a, b_ = m.parameters["a"], m.parameters["b"]
        assert a < b_
        assert -1.0 <= a < b_ <= 1.0


def test_builder_grid_sigmoidal_combos():
    # Difference of sigmoids should form a band-like shape (plateau-ish)
    b = ANFISBuilder().add_input("x", 0.0, 5.0, n_mfs=3, mf_type="diffsigmoidal", overlap=0.4)
    model = b.build()
    mfs = _get_single_input_mfs(model, "x")
    assert len(mfs) == 3
    assert all(isinstance(m, DiffSigmoidalMF) for m in mfs)
    for m in mfs:
        p = m.parameters
        assert p["a1"] > 0 and p["a2"] > 0
        assert p["c1"] < p["c2"]

    # Product of sigmoidals: one increasing and one decreasing
    b2 = ANFISBuilder().add_input("x", -2.0, 2.0, n_mfs=2, mf_type="prodsigmoidal", overlap=0.6)
    model2 = b2.build()
    mfs2 = _get_single_input_mfs(model2, "x")
    assert len(mfs2) == 2
    assert all(isinstance(m, ProdSigmoidalMF) for m in mfs2)
    for m in mfs2:
        p = m.parameters
        assert p["a1"] > 0 and p["a2"] < 0  # increasing then decreasing
        assert p["c1"] < p["c2"]
        # basic bump sanity: mid greater than ends
        x = np.linspace(-2, 2, 101)
        y = m.forward(x)
        assert y[50] >= y[0] and y[50] >= y[-1]


def test_builder_grid_aliases_and_zero_range_fallback():
    # Aliases: ls and lz; zero-range forces tiny-span fallback path
    b = ANFISBuilder().add_input("x", 1.0, 1.0, n_mfs=2, mf_type="ls", overlap=0.9)
    model = b.build()
    mfs = _get_single_input_mfs(model, "x")
    assert all(isinstance(m, LinSShapedMF) for m in mfs)
    for m in mfs:
        a, b_ = m.parameters["a"], m.parameters["b"]
        assert a < b_
        # centered near 1.0 with tiny span from fallback
        assert abs((a + b_) / 2.0 - 1.0) < 1e-3
        assert (b_ - a) > 0 and (b_ - a) < 1e-3

    b2 = ANFISBuilder().add_input("x", 0.0, 0.0, n_mfs=1, mf_type="lz", overlap=0.5)
    model2 = b2.build()
    mfs2 = _get_single_input_mfs(model2, "x")
    assert len(mfs2) == 1 and isinstance(mfs2[0], LinZShapedMF)
    a, b_ = mfs2[0].parameters["a"], mfs2[0].parameters["b"]
    assert a < b_
    assert abs((a + b_) / 2.0 - 0.0) < 1e-3
    assert (b_ - a) > 0 and (b_ - a) < 1e-3


def test_single_mf_branches_for_new_grid_creators():
    b = ANFISBuilder()

    # linsshape single
    mfs_ls = b._create_linsshape_mfs(0.0, 10.0, 1, overlap=0.5)
    assert len(mfs_ls) == 1 and isinstance(mfs_ls[0], LinSShapedMF)
    a, bb = mfs_ls[0].parameters["a"], mfs_ls[0].parameters["b"]
    assert a < bb

    # linzshape single
    mfs_lz = b._create_linzshape_mfs(-5.0, 5.0, 1, overlap=0.3)
    assert len(mfs_lz) == 1 and isinstance(mfs_lz[0], LinZShapedMF)
    a2, b2 = mfs_lz[0].parameters["a"], mfs_lz[0].parameters["b"]
    assert a2 < b2

    # diffsigmoidal single
    mfs_diff = b._create_diff_sigmoidal_mfs(0.0, 4.0, 1, overlap=0.4)
    assert len(mfs_diff) == 1 and isinstance(mfs_diff[0], DiffSigmoidalMF)
    p = mfs_diff[0].parameters
    assert p["a1"] > 0 and p["a2"] > 0 and p["c1"] < p["c2"]

    # prodsigmoidal single
    mfs_prod = b._create_prod_sigmoidal_mfs(-3.0, 3.0, 1, overlap=0.6)
    assert len(mfs_prod) == 1 and isinstance(mfs_prod[0], ProdSigmoidalMF)
    pp = mfs_prod[0].parameters
    assert pp["a1"] > 0 and pp["a2"] < 0 and pp["c1"] < pp["c2"]


def test_grid_zero_range_fallback_sigmoidal_combos():
    b = ANFISBuilder()
    # diff sigmoidal zero-range -> c1<c2 fallback path
    mfs_diff = b._create_diff_sigmoidal_mfs(1.0, 1.0, 1, overlap=0.5)
    p = mfs_diff[0].parameters
    assert p["c1"] < p["c2"]
    assert (p["c2"] - p["c1"]) < 1e-3

    # prod sigmoidal zero-range -> c1<c2 fallback path
    mfs_prod = b._create_prod_sigmoidal_mfs(-2.0, -2.0, 1, overlap=0.5)
    pp = mfs_prod[0].parameters
    assert pp["c1"] < pp["c2"]
    assert (pp["c2"] - pp["c1"]) < 1e-3


def test_fcm_constant_data_fallback_for_new_types():
    # Constant data causes rmin==rmax so clamps create a>=b or c1>=c2; test fallbacks
    data = np.full(30, 3.14)

    # linsshape
    b1 = ANFISBuilder()
    b1.add_input_from_data("x", data, n_mfs=2, mf_type="linsshape", init="fcm", random_state=0)
    for m in b1.build().input_mfs["x"]:
        a, bb = m.parameters["a"], m.parameters["b"]
        assert a < bb and (bb - a) < 1e-3

    # linzshape
    b2 = ANFISBuilder()
    b2.add_input_from_data("x", data, n_mfs=2, mf_type="linzshape", init="fcm", random_state=0)
    for m in b2.build().input_mfs["x"]:
        a, bb = m.parameters["a"], m.parameters["b"]
        assert a < bb and (bb - a) < 1e-3

    # diffsigmoidal
    b3 = ANFISBuilder()
    b3.add_input_from_data("x", data, n_mfs=2, mf_type="diffsigmoidal", init="fcm", random_state=0)
    for m in b3.build().input_mfs["x"]:
        p = m.parameters
        assert p["c1"] < p["c2"] and (p["c2"] - p["c1"]) < 1e-3

    # prodsigmoidal
    b4 = ANFISBuilder()
    b4.add_input_from_data("x", data, n_mfs=2, mf_type="prodsigmoidal", init="fcm", random_state=0)
    for m in b4.build().input_mfs["x"]:
        p = m.parameters
        assert p["c1"] < p["c2"] and (p["c2"] - p["c1"]) < 1e-3


def test_builder_fcm_linear_shapes():
    rng = np.random.default_rng(0)
    data = np.concatenate(
        [
            rng.normal(-1.0, 0.1, size=100),
            rng.normal(0.0, 0.1, size=100),
            rng.normal(1.0, 0.1, size=100),
        ]
    )
    b = ANFISBuilder()
    b.add_input_from_data("x", data, n_mfs=3, mf_type="linsshape", init="fcm", random_state=0)
    model = b.build()
    mfs = _get_single_input_mfs(model, "x")
    assert len(mfs) == 3
    assert all(isinstance(m, LinSShapedMF) for m in mfs)

    b2 = ANFISBuilder()
    b2.add_input_from_data("x", data, n_mfs=3, mf_type="linzshape", init="fcm", random_state=0)
    model2 = b2.build()
    mfs2 = _get_single_input_mfs(model2, "x")
    assert len(mfs2) == 3
    assert all(isinstance(m, LinZShapedMF) for m in mfs2)


def test_builder_fcm_sigmoidal_combos():
    # Use smooth unimodal data; FCM still yields centers and widths
    x = np.linspace(-3, 3, 600)
    data = np.tanh(x) + 0.05 * np.sin(5 * x)

    b = ANFISBuilder()
    b.add_input_from_data("x", data, n_mfs=3, mf_type="diffsigmoidal", init="fcm", random_state=1)
    model = b.build()
    mfs = _get_single_input_mfs(model, "x")
    assert len(mfs) == 3
    assert all(isinstance(m, DiffSigmoidalMF) for m in mfs)
    for m in mfs:
        p = m.parameters
        assert p["a1"] > 0 and p["a2"] > 0 and p["c1"] < p["c2"]

    b2 = ANFISBuilder()
    b2.add_input_from_data("x", data, n_mfs=2, mf_type="prodsigmoidal", init="fcm", random_state=2)
    model2 = b2.build()
    mfs2 = _get_single_input_mfs(model2, "x")
    assert len(mfs2) == 2
    assert all(isinstance(m, ProdSigmoidalMF) for m in mfs2)
    for m in mfs2:
        p = m.parameters
        assert p["a1"] > 0 and p["a2"] < 0 and p["c1"] < p["c2"]

    # Aliases for FCM path
    b3 = ANFISBuilder()
    b3.add_input_from_data("x", data, n_mfs=3, mf_type="diffsigmoid", init="fcm", random_state=4)
    model3 = b3.build()
    assert all(isinstance(m, DiffSigmoidalMF) for m in _get_single_input_mfs(model3, "x"))

    b4 = ANFISBuilder()
    b4.add_input_from_data("x", data, n_mfs=2, mf_type="prodsigmoid", init="fcm", random_state=5)
    model4 = b4.build()
    assert all(isinstance(m, ProdSigmoidalMF) for m in _get_single_input_mfs(model4, "x"))
