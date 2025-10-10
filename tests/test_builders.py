"""
Tests for ANFIS builders module.

This module tests the ANFISBuilder class,
focusing on proper model construction, parameter validation,
and integration with membership functions.
"""

import numpy as np
import pytest

from anfis_toolbox.builders import ANFISBuilder
from anfis_toolbox.membership import (
    BellMF,
    DiffSigmoidalMF,
    GaussianMF,
    LinSShapedMF,
    LinZShapedMF,
    PiMF,
    ProdSigmoidalMF,
    SigmoidalMF,
    SShapedMF,
    TrapezoidalMF,
    TriangularMF,
    ZShapedMF,
)
from anfis_toolbox.model import ANFIS


class TestANFISBuilder:
    """Test cases for ANFISBuilder class."""

    def test_init(self):
        """Test ANFISBuilder initialization."""
        builder = ANFISBuilder()
        assert isinstance(builder, ANFISBuilder)
        assert builder.input_mfs == {}

    def test_add_input_basic(self):
        """Test adding a basic input with default parameters."""
        builder = ANFISBuilder()

        result = builder.add_input("x1", -1.0, 1.0)

        assert result is builder  # Should return self for chaining
        assert "x1" in builder.input_mfs
        assert len(builder.input_mfs["x1"]) == 3  # Default 3 MFs
        assert all(isinstance(mf, GaussianMF) for mf in builder.input_mfs["x1"])

    def test_add_input_custom_params(self):
        """Test adding input with custom parameters."""
        builder = ANFISBuilder()

        builder.add_input("temperature", 0.0, 100.0, n_mfs=5, mf_type="triangular")

        assert "temperature" in builder.input_mfs
        assert len(builder.input_mfs["temperature"]) == 5
        assert all(isinstance(mf, TriangularMF) for mf in builder.input_mfs["temperature"])

    def test_add_input_trapezoidal(self):
        """Test adding input with trapezoidal membership functions."""
        builder = ANFISBuilder()

        builder.add_input("speed", 0.0, 120.0, n_mfs=4, mf_type="trapezoidal")

        assert "speed" in builder.input_mfs
        assert len(builder.input_mfs["speed"]) == 4
        assert all(isinstance(mf, TrapezoidalMF) for mf in builder.input_mfs["speed"])

    def test_add_input_invalid_mf_type(self):
        """Test adding input with invalid MF type raises error."""
        builder = ANFISBuilder()

        with pytest.raises(ValueError, match="Unknown membership function type"):
            builder.add_input("x1", 0.0, 1.0, mf_type="invalid")

    def test_add_multiple_inputs(self):
        """Test adding multiple inputs."""
        builder = ANFISBuilder()

        builder.add_input("x1", -1.0, 1.0, n_mfs=2)
        builder.add_input("x2", 0.0, 10.0, n_mfs=3)

        assert len(builder.input_mfs) == 2
        assert "x1" in builder.input_mfs
        assert "x2" in builder.input_mfs
        assert len(builder.input_mfs["x1"]) == 2
        assert len(builder.input_mfs["x2"]) == 3

    def test_build_model_no_inputs(self):
        """Test building model without inputs raises error."""
        builder = ANFISBuilder()

        with pytest.raises(ValueError, match="No input variables defined"):
            builder.build()

    def test_build_model_single_input(self):
        """Test building model with single input."""
        builder = ANFISBuilder()
        builder.add_input("x1", -2.0, 2.0, n_mfs=3)

        model = builder.build()

        assert isinstance(model, ANFIS)
        assert model.n_inputs == 1
        assert model.n_rules == 3

    def test_build_model_multiple_inputs(self):
        """Test building model with multiple inputs."""
        builder = ANFISBuilder()
        builder.add_input("x1", -1.0, 1.0, n_mfs=2)
        builder.add_input("x2", 0.0, 5.0, n_mfs=3)

        model = builder.build()

        assert isinstance(model, ANFIS)
        assert model.n_inputs == 2
        assert model.n_rules == 2 * 3  # Product of MFs per input

    def test_build_model_with_explicit_rules(self):
        builder = ANFISBuilder()
        builder.add_input("x1", -1.0, 1.0, n_mfs=2)
        builder.add_input("x2", -2.0, 2.0, n_mfs=2)

        explicit_rules = [(0, 0), (0, 1), (1, 1)]
        builder.set_rules(explicit_rules)

        model = builder.build()

        assert model.n_rules == len(explicit_rules)
        assert model.rules == explicit_rules

    def test_method_chaining(self):
        """Test that methods can be chained."""
        builder = ANFISBuilder()

        model = builder.add_input("x1", -1, 1, n_mfs=2).add_input("x2", 0, 10, n_mfs=2).build()

        assert isinstance(model, ANFIS)
        assert model.n_inputs == 2

    def test_create_gaussian_mfs(self):
        """Test creation of Gaussian membership functions."""
        builder = ANFISBuilder()

        mfs = builder._create_gaussian_mfs(-2.0, 2.0, 3, overlap=0.5)

        assert len(mfs) == 3
        assert all(isinstance(mf, GaussianMF) for mf in mfs)

        # Test that MFs are distributed across the range
        means = [mf.parameters["mean"] for mf in mfs]
        assert means == [-2.0, 0.0, 2.0]  # Centers should be evenly spaced

        # Test sigma calculation
        expected_sigma = (2.0 - (-2.0)) / (3 - 1) * 0.5  # 2.0
        for mf in mfs:
            assert abs(mf.parameters["sigma"] - expected_sigma) < 1e-10

    def test_create_triangular_mfs(self):
        """Test creation of triangular membership functions."""
        builder = ANFISBuilder()

        mfs = builder._create_triangular_mfs(0.0, 10.0, 4, overlap=0.3)

        assert len(mfs) == 4
        assert all(isinstance(mf, TriangularMF) for mf in mfs)

        # Test that MFs span the range appropriately
        lefts = [mf.parameters["a"] for mf in mfs]
        rights = [mf.parameters["c"] for mf in mfs]
        assert min(lefts) <= 0.0
        assert max(rights) >= 10.0

    def test_create_trapezoidal_mfs(self):
        """Test creation of trapezoidal membership functions."""
        builder = ANFISBuilder()

        mfs = builder._create_trapezoidal_mfs(-5.0, 5.0, 2, overlap=0.4)

        assert len(mfs) == 2
        assert all(isinstance(mf, TrapezoidalMF) for mf in mfs)

    def test_overlap_parameter(self):
        """Test that overlap parameter affects MF spacing."""
        builder = ANFISBuilder()

        # Test with low overlap
        mfs_low = builder._create_gaussian_mfs(-1.0, 1.0, 3, overlap=0.1)
        widths_low = [mf.parameters["sigma"] for mf in mfs_low]

        # Test with high overlap
        mfs_high = builder._create_gaussian_mfs(-1.0, 1.0, 3, overlap=0.9)
        widths_high = [mf.parameters["sigma"] for mf in mfs_high]

        # Higher overlap should result in wider functions
        assert all(w_h > w_l for w_h, w_l in zip(widths_high, widths_low, strict=False))

    def test_different_n_mfs(self):
        """Test creating different numbers of membership functions."""
        builder = ANFISBuilder()

        for n_mfs in [2, 3, 5, 7]:  # Skip n_mfs=1 which causes division by zero
            mfs = builder._create_gaussian_mfs(-1.0, 1.0, n_mfs, overlap=0.5)
            assert len(mfs) == n_mfs
            assert all(isinstance(mf, GaussianMF) for mf in mfs)

    def test_edge_case_single_mf(self):
        """Test edge case with single membership function."""
        builder = ANFISBuilder()

        # For single MF, we need to handle division by zero
        mfs = builder._create_gaussian_mfs(-1.0, 1.0, 1, overlap=0.5)
        assert len(mfs) == 1
        assert isinstance(mfs[0], GaussianMF)

        # Single MF should be at the center of linspace range
        assert mfs[0].parameters["mean"] == -1.0  # linspace with n=1 returns start value

    def test_create_bell_single_mf(self):
        """Bell MF: single MF branch sets half-width from range and default slope."""
        builder = ANFISBuilder()
        mfs = builder._create_bell_mfs(0.0, 10.0, 1, overlap=0.7)
        assert len(mfs) == 1
        mf = mfs[0]
        assert isinstance(mf, BellMF)
        # a = 0.25 * (range_max - range_min)
        assert np.isclose(mf.parameters["a"], 2.5)
        assert np.isclose(mf.parameters["b"], 2.0)
        # c equals start of linspace when n=1
        assert np.isclose(mf.parameters["c"], 0.0)

    def test_create_sigmoidal_single_mf(self):
        """Sigmoidal MF: single MF branch sets width from range and computes slope."""
        builder = ANFISBuilder()
        mfs = builder._create_sigmoidal_mfs(0.0, 10.0, 1, overlap=0.5)
        assert len(mfs) == 1
        mf = mfs[0]
        assert isinstance(mf, SigmoidalMF)
        # width = 0.5 * (range_max - range_min) => 5, a = 4.4 / width ≈ 0.88
        assert np.isclose(mf.parameters["a"], 0.88)
        assert np.isclose(mf.parameters["c"], 0.0)

    def test_create_sshape_zero_range_fallback(self):
        """S-shaped MF: zero range triggers tiny-span fallback (a < b)."""
        builder = ANFISBuilder()
        mfs = builder._create_sshape_mfs(1.23, 1.23, 1, overlap=0.5)
        assert len(mfs) == 1
        a, b = mfs[0].parameters["a"], mfs[0].parameters["b"]
        assert a < b
        # very small span around center
        assert (b - a) < 1e-5
        assert abs((a + b) / 2.0 - 1.23) < 1e-3

    def test_create_zshape_zero_range_fallback(self):
        """Z-shaped MF: zero range triggers tiny-span fallback (a < b)."""
        builder = ANFISBuilder()
        mfs = builder._create_zshape_mfs(-2.0, -2.0, 1, overlap=0.5)
        assert len(mfs) == 1
        a, b = mfs[0].parameters["a"], mfs[0].parameters["b"]
        assert a < b
        assert (b - a) < 1e-5
        assert abs((a + b) / 2.0 - (-2.0)) < 1e-3

    def test_create_pi_single_and_zero_range(self):
        """Pi MF: cover single-MF width branch and zero-range fallback branch."""
        builder = ANFISBuilder()
        # Single MF normal range
        mfs = builder._create_pi_mfs(0.0, 10.0, 1, overlap=0.4)
        assert len(mfs) == 1
        a, b, c, d = (mfs[0].parameters[k] for k in ("a", "b", "c", "d"))
        assert a < b <= c < d
        # Zero range triggers clamp and fallback
        mfs_zero = builder._create_pi_mfs(0.0, 0.0, 1, overlap=0.4)
        a2, b2, c2, d2 = (mfs_zero[0].parameters[k] for k in ("a", "b", "c", "d"))
        assert a2 < b2 <= c2 < d2
        assert (d2 - a2) < 1e-4

    def test_add_input_bell_and_alias(self):
        """Bell MF creation with canonical and alias names."""
        builder = ANFISBuilder()
        for mf_name in ("bell", "gbell"):
            builder.add_input("x", -1.0, 1.0, n_mfs=3, mf_type=mf_name)
            assert all(isinstance(mf, BellMF) for mf in builder.input_mfs["x"])

    def test_add_input_sigmoidal_and_alias(self):
        """Sigmoidal MF creation with canonical and alias names."""
        builder = ANFISBuilder()
        for mf_name in ("sigmoidal", "sigmoid"):
            builder.add_input("x", -2.0, 2.0, n_mfs=4, mf_type=mf_name)
            assert all(isinstance(mf, SigmoidalMF) for mf in builder.input_mfs["x"])

    def test_add_input_sshape_and_alias(self):
        """S-shaped MF creation with canonical and alias names and parameter ordering."""
        builder = ANFISBuilder()
        for mf_name in ("sshape", "s"):
            builder.add_input("x", 0.0, 10.0, n_mfs=3, mf_type=mf_name)
            mfs = builder.input_mfs["x"]
            assert all(isinstance(mf, SShapedMF) for mf in mfs)
            # Ensure a < b for each S-shaped MF
            for mf in mfs:
                a, b = mf.parameters["a"], mf.parameters["b"]
                assert a < b

    def test_add_input_zshape_and_alias(self):
        """Z-shaped MF creation with canonical and alias names and parameter ordering."""
        builder = ANFISBuilder()
        for mf_name in ("zshape", "z"):
            builder.add_input("x", -5.0, 5.0, n_mfs=3, mf_type=mf_name)
            mfs = builder.input_mfs["x"]
            assert all(isinstance(mf, ZShapedMF) for mf in mfs)
            for mf in mfs:
                a, b = mf.parameters["a"], mf.parameters["b"]
                assert a < b

    def test_add_input_pi_and_alias(self):
        """Pi MF creation with canonical and alias names and parameter ordering."""
        builder = ANFISBuilder()
        for mf_name in ("pi", "pimf"):
            builder.add_input("x", -3.0, 3.0, n_mfs=3, mf_type=mf_name)
            mfs = builder.input_mfs["x"]
            assert all(isinstance(mf, PiMF) for mf in mfs)
            for mf in mfs:
                a = mf.parameters["a"]
                b = mf.parameters["b"]
                c = mf.parameters["c"]
                d = mf.parameters["d"]
                assert a < b <= c < d

    def test_add_input_from_data(self):
        """add_input_from_data infers ranges and creates requested MF types."""
        data = np.array([1.0, 1.5, 2.0, 2.5])
        builder = ANFISBuilder()
        builder.add_input_from_data("x", data, n_mfs=2, mf_type="sigmoidal", overlap=0.6, margin=0.2)
        assert "x" in builder.input_mfs
        mfs = builder.input_mfs["x"]
        assert len(mfs) == 2
        assert all(isinstance(mf, SigmoidalMF) for mf in mfs)

    def test_add_input_from_data_fcm_gaussian(self):
        """FCM init maps clusters to Gaussian MFs with deterministic centers."""
        rng = np.random.RandomState(0)
        x = np.concatenate([rng.normal(-2.0, 0.2, 40), rng.normal(2.0, 0.2, 40)])
        builder = ANFISBuilder()
        builder.add_input_from_data("x", x, n_mfs=2, mf_type="gaussian", init="fcm", random_state=7)
        mfs = builder.input_mfs["x"]
        assert len(mfs) == 2
        assert all(isinstance(mf, GaussianMF) for mf in mfs)
        centers = np.array([mf.parameters["mean"] for mf in mfs])
        assert np.all(np.diff(centers) > 0)  # sorted order

    def test_add_input_from_data_fcm_bell(self):
        """FCM init also supports Bell MFs via alias 'gbell'."""
        rng = np.random.RandomState(1)
        x = np.concatenate([rng.normal(-1.0, 0.1, 30), rng.normal(1.0, 0.1, 30)])
        builder = ANFISBuilder()
        builder.add_input_from_data("x", x, n_mfs=2, mf_type="gbell", init="fcm", random_state=3)
        mfs = builder.input_mfs["x"]
        assert len(mfs) == 2
        assert all(isinstance(mf, BellMF) for mf in mfs)

    def test_add_input_from_data_fcm_triangular(self):
        rng = np.random.RandomState(4)
        x = np.concatenate([rng.normal(-2.0, 0.2, 50), rng.normal(2.0, 0.2, 50)])
        builder = ANFISBuilder()
        builder.add_input_from_data("x", x, n_mfs=2, mf_type="triangular", init="fcm", random_state=5)
        mfs = builder.input_mfs["x"]
        assert len(mfs) == 2
        assert all(isinstance(mf, TriangularMF) for mf in mfs)
        for mf in mfs:
            a, b, c = mf.parameters["a"], mf.parameters["b"], mf.parameters["c"]
            assert a < b < c

    def test_add_input_from_data_fcm_trapezoidal(self):
        rng = np.random.RandomState(6)
        x = np.concatenate([rng.normal(-1.5, 0.3, 60), rng.normal(1.5, 0.3, 60)])
        builder = ANFISBuilder()
        builder.add_input_from_data("x", x, n_mfs=2, mf_type="trapezoidal", init="fcm", random_state=7)
        mfs = builder.input_mfs["x"]
        assert len(mfs) == 2
        assert all(isinstance(mf, TrapezoidalMF) for mf in mfs)
        for mf in mfs:
            a, b, c, d = (mf.parameters[k] for k in ("a", "b", "c", "d"))
            assert a < b <= c < d

    def test_add_input_from_data_fcm_sigmoidal(self):
        rng = np.random.RandomState(8)
        x = np.concatenate([rng.normal(-0.5, 0.1, 40), rng.normal(0.5, 0.1, 40)])
        builder = ANFISBuilder()
        builder.add_input_from_data("x", x, n_mfs=2, mf_type="sigmoidal", init="fcm", random_state=9)
        mfs = builder.input_mfs["x"]
        assert len(mfs) == 2
        assert all(isinstance(mf, SigmoidalMF) for mf in mfs)
        # slopes should be positive
        for mf in mfs:
            assert mf.parameters["a"] > 0

    def test_add_input_from_data_fcm_sshape_zshape(self):
        rng = np.random.RandomState(10)
        x = np.concatenate([rng.normal(-3.0, 0.4, 80), rng.normal(3.0, 0.4, 80)])
        builder = ANFISBuilder()
        builder.add_input_from_data("xs", x, n_mfs=2, mf_type="sshape", init="fcm", random_state=11)
        builder.add_input_from_data("xz", x, n_mfs=2, mf_type="zshape", init="fcm", random_state=11)
        mfs_s = builder.input_mfs["xs"]
        mfs_z = builder.input_mfs["xz"]
        assert all(isinstance(mf, SShapedMF) for mf in mfs_s)
        assert all(isinstance(mf, ZShapedMF) for mf in mfs_z)
        for mf in mfs_s + mfs_z:
            a, b = mf.parameters["a"], mf.parameters["b"]
            assert a < b

    def test_add_input_from_data_fcm_pi(self):
        rng = np.random.RandomState(12)
        x = np.concatenate([rng.normal(-2.5, 0.2, 50), rng.normal(2.5, 0.2, 50)])
        builder = ANFISBuilder()
        builder.add_input_from_data("x", x, n_mfs=2, mf_type="pi", init="fcm", random_state=13)
        mfs = builder.input_mfs["x"]
        assert len(mfs) == 2
        assert all(isinstance(mf, PiMF) for mf in mfs)
        for mf in mfs:
            a, b, c, d = (mf.parameters[k] for k in ("a", "b", "c", "d"))
            assert a < b <= c < d

    def test_add_input_from_data_random_reproducible(self):
        data = np.linspace(-1.0, 1.0, 25)
        builder_a = ANFISBuilder()
        builder_a.add_input_from_data("x", data, n_mfs=3, mf_type="gaussian", init="random", random_state=42)
        builder_b = ANFISBuilder()
        builder_b.add_input_from_data("x", data, n_mfs=3, mf_type="gaussian", init="random", random_state=42)

        mfs_a = builder_a.input_mfs["x"]
        mfs_b = builder_b.input_mfs["x"]
        assert len(mfs_a) == len(mfs_b) == 3
        assert all(isinstance(mf, GaussianMF) for mf in mfs_a)

        centers_a = np.array([mf.parameters["mean"] for mf in mfs_a])
        centers_b = np.array([mf.parameters["mean"] for mf in mfs_b])
        sigmas_a = np.array([mf.parameters["sigma"] for mf in mfs_a])
        sigmas_b = np.array([mf.parameters["sigma"] for mf in mfs_b])

        assert np.allclose(centers_a, centers_b)
        assert np.allclose(sigmas_a, sigmas_b)

        builder_c = ANFISBuilder()
        builder_c.add_input_from_data("x", data, n_mfs=3, mf_type="gaussian", init="random", random_state=7)
        centers_c = np.array([mf.parameters["mean"] for mf in builder_c.input_mfs["x"]])
        # It's highly unlikely the layouts are identical with a different seed
        assert not np.allclose(centers_a, centers_c)

        low, high = builder_a.input_ranges["x"]
        rmin, rmax = float(np.min(data)), float(np.max(data))
        span = rmax - rmin
        expected_low = pytest.approx(rmin - span * 0.10)
        expected_high = pytest.approx(rmax + span * 0.10)
        assert low == expected_low
        assert high == expected_high

    def test_add_input_from_data_random_empty_raises(self):
        builder = ANFISBuilder()
        with pytest.raises(ValueError, match="Cannot initialize membership functions from empty data array"):
            builder.add_input_from_data("x", np.array([]), init="random")

    def test_add_input_from_data_random_with_generator(self):
        data = np.linspace(-3.0, 3.0, 40)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        builder_a = ANFISBuilder()
        builder_a.add_input_from_data("x", data, n_mfs=4, mf_type="gaussian", init="random", random_state=rng1)

        builder_b = ANFISBuilder()
        builder_b.add_input_from_data("x", data, n_mfs=4, mf_type="gaussian", init="random", random_state=rng2)

        mfs_a = builder_a.input_mfs["x"]
        mfs_b = builder_b.input_mfs["x"]
        centers_a = np.array([mf.parameters["mean"] for mf in mfs_a])
        centers_b = np.array([mf.parameters["mean"] for mf in mfs_b])
        sigmas_a = np.array([mf.parameters["sigma"] for mf in mfs_a])
        sigmas_b = np.array([mf.parameters["sigma"] for mf in mfs_b])

        np.testing.assert_allclose(centers_a, centers_b)
        np.testing.assert_allclose(sigmas_a, sigmas_b)

        low, high = builder_a.input_ranges["x"]
        assert low < high

    def test_add_input_from_data_random_clamps_min_width(self):
        """Random init clamps widths to the computed floor (line 255)."""
        data = np.linspace(0.0, 0.01, 10)
        builder = ANFISBuilder()
        builder.add_input_from_data(
            "x",
            data,
            n_mfs=4,
            mf_type="gaussian",
            init="random",
            random_state=0,
            overlap=1.0,
            margin=0.0,
        )

        mfs = builder.input_mfs["x"]
        centers = np.array([mf.parameters["mean"] for mf in mfs])
        sigmas = np.array([mf.parameters["sigma"] for mf in mfs])
        low, high = builder.input_ranges["x"]

        base_span = (high - low) / len(mfs)
        floor = max(base_span * max(1.0, 0.1), 1e-3)
        widths = sigmas * 2.0

        assert widths.min() >= floor
        # At least one width hits the clamp exactly, proving np.maximum ran.
        assert np.isclose(widths, floor).any()
        # Original centers remain sorted and inside [low, high].
        assert np.all(np.diff(centers) >= 0)
        assert low <= centers[0] <= centers[-1] <= high

    def test_add_input_from_data_random_single_mf_widths(self):
        """Random init with one MF hits single-width branch (line 225)."""
        data = np.array([-0.5, 0.5])
        builder = ANFISBuilder()
        builder.add_input_from_data(
            "x",
            data,
            n_mfs=1,
            mf_type="gaussian",
            init="random",
            margin=0.2,
            overlap=0.3,
            random_state=0,
        )

        mfs = builder.input_mfs["x"]
        assert len(mfs) == 1
        low, high = builder.input_ranges["x"]
        span = high - low
        sigma = mfs[0].parameters["sigma"]
        # When n_mfs==1 widths start as high-low before floor clamp
        assert np.isclose(span, 1.4)
        assert np.isclose(sigma * 2.0, span)

    def test_add_input_from_data_fcm_fallbacks_constant_data(self):
        """Constant data triggers fallback branches ensuring valid parameter ordering."""
        x = np.full(20, 1.234)
        # Triangular fallback
        b1 = ANFISBuilder()
        b1.add_input_from_data("x", x, n_mfs=2, mf_type="triangular", init="fcm", random_state=0)
        for mf in b1.input_mfs["x"]:
            a, b, c = mf.parameters["a"], mf.parameters["b"], mf.parameters["c"]
            assert a < b < c
        # Trapezoidal fallback
        b2 = ANFISBuilder()
        b2.add_input_from_data("x", x, n_mfs=2, mf_type="trapezoidal", init="fcm", random_state=0)
        for mf in b2.input_mfs["x"]:
            a, b, c, d = (mf.parameters[k] for k in ("a", "b", "c", "d"))
            assert a < b <= c < d
        # S/Z-shape fallbacks
        b3 = ANFISBuilder()
        b3.add_input_from_data("xs", x, n_mfs=2, mf_type="sshape", init="fcm", random_state=0)
        b3.add_input_from_data("xz", x, n_mfs=2, mf_type="zshape", init="fcm", random_state=0)
        for mf in b3.input_mfs["xs"] + b3.input_mfs["xz"]:
            a, b = mf.parameters["a"], mf.parameters["b"]
            assert a < b
        # Pi fallback
        b4 = ANFISBuilder()
        b4.add_input_from_data("x", x, n_mfs=2, mf_type="pi", init="fcm", random_state=0)
        for mf in b4.input_mfs["x"]:
            a, b, c, d = (mf.parameters[k] for k in ("a", "b", "c", "d"))
            assert a < b <= c < d

    def test_add_input_from_data_fcm_unsupported_type_raises(self):
        x = np.linspace(0, 1, 10)
        builder = ANFISBuilder()
        with pytest.raises(ValueError):
            builder.add_input_from_data("x", x, n_mfs=2, mf_type="unknown", init="fcm", random_state=0)

    def test_add_input_from_data_fcm_insufficient_samples_raises(self):
        builder = ANFISBuilder()
        with pytest.raises(ValueError):
            builder.add_input_from_data("x", np.array([0.0]), n_mfs=2, mf_type="gaussian", init="fcm")

    def test_builder_for_regression_with_fcm(self):
        """Builder supports inferring inputs via FCM across multiple features."""
        rng = np.random.RandomState(2)
        X = np.column_stack(
            [
                np.concatenate([rng.normal(-1.0, 0.1, 40), rng.normal(1.0, 0.1, 40)]),
                np.concatenate([rng.normal(0.0, 0.5, 40), rng.normal(3.0, 0.5, 40)]),
            ]
        )
        builder = ANFISBuilder()
        for i in range(X.shape[1]):
            builder.add_input_from_data(
                f"x{i + 1}",
                X[:, i],
                n_mfs=2,
                mf_type="gaussian",
                init="fcm",
                random_state=11,
            )
        model = builder.build()
        # 2 inputs, 2 MFs each -> 4 rules
        assert model.n_inputs == 2
        assert model.n_rules == 4


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


def test_builder_gaussian2_grid():
    b = ANFISBuilder()
    b.add_input("x1", -2.0, 2.0, n_mfs=3, mf_type="gaussian2", overlap=0.6)
    model = b.build()
    # Expect 3 MFs for one input
    assert len(model.membership_functions["x1"]) == 3


def test_builder_gaussian2_grid_prediction():
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    builder = ANFISBuilder()
    builder.add_input_from_data("x1", X[:, 0], n_mfs=4, mf_type="gaussian2", init="grid")
    model = builder.build()
    y = model.predict(X)
    assert y.shape == (50, 1)


def test_builder_gaussian2_fcm_prediction():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, size=(80, 1))
    builder = ANFISBuilder()
    builder.add_input_from_data("x1", X[:, 0], n_mfs=3, mf_type="gaussian2", init="fcm", random_state=42)
    model = builder.build()
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
