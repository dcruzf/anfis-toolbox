import numpy as np

from anfis_toolbox import QuickANFIS
from anfis_toolbox.builders import ANFISBuilder


def test_builder_gaussian2_grid():
    b = ANFISBuilder()
    b.add_input("x1", -2.0, 2.0, n_mfs=3, mf_type="gaussian2", overlap=0.6)
    model = b.build()
    # Expect 3 MFs for one input
    assert len(model.membership_functions["x1"]) == 3


def test_quick_gaussian2_grid():
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    model = QuickANFIS.for_regression(X, n_mfs=4, mf_type="gaussian2", init="grid")
    # Should forward without error
    y = model.predict(X)
    assert y.shape == (50, 1)


def test_quick_gaussian2_fcm():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(80, 1))
    model = QuickANFIS.for_regression(X, n_mfs=3, mf_type="gaussian2", init="fcm", random_state=42)
    y = model.predict(X)
    assert y.shape == (80, 1)


def test_gaussian2_grid_single_and_zero_range_fallback():
    # Cover n_mfs==1 branch and zero-range fallback in _create_gaussian2_mfs
    b = ANFISBuilder()
    # Single MF with non-zero range
    mfs = b._create_gaussian2_mfs(-1.0, 1.0, 1, overlap=0.6)
    assert len(mfs) == 1
    mf = mfs[0]
    assert mf.__class__.__name__ == "Gaussian2MF"
    # Should produce a small plateau strictly inside the range
    c1, c2 = mf.parameters["c1"], mf.parameters["c2"]
    assert c1 < c2
    # Single MF with zero range triggers fallback for c1/c2 but overall raises
    # because sigma becomes zero and Gaussian2MF validates sigma>0.
    import pytest

    with pytest.raises(ValueError):
        _ = b._create_gaussian2_mfs(0.0, 0.0, 1, overlap=0.5)


def test_add_input_from_data_fcm_gaussian2_constant_data_fallback():
    # Constant data with FCM should trigger c1<c2 fallback inside gaussian2 branch
    x = np.full(20, 1.234)
    b = ANFISBuilder()
    b.add_input_from_data("x", x, n_mfs=2, mf_type="gaussian2", init="fcm", random_state=0)
    mfs = b.input_mfs["x"]
    assert len(mfs) == 2
    for mf in mfs:
        assert mf.__class__.__name__ == "Gaussian2MF"
        c1, c2 = mf.parameters["c1"], mf.parameters["c2"]
        assert c1 < c2
        assert (c2 - c1) < 1e-3
