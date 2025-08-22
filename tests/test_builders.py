"""
Tests for ANFIS builders module.

This module tests the ANFISBuilder and QuickANFIS classes,
focusing on proper model construction, parameter validation,
and integration with membership functions.
"""

import numpy as np
import pytest

from anfis_toolbox.builders import ANFISBuilder, QuickANFIS
from anfis_toolbox.membership import (
    BellMF,
    GaussianMF,
    PiMF,
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


class TestQuickANFIS:
    """Test cases for QuickANFIS class."""

    def test_for_regression_basic(self):
        """Test basic regression model creation."""
        X = np.random.uniform(-1, 1, (10, 2))

        model = QuickANFIS.for_regression(X)

        assert isinstance(model, ANFIS)
        assert model.n_inputs == 2
        assert model.n_rules == 3 * 3  # Default 3 MFs per input

    def test_for_regression_custom_mfs(self):
        """Test regression with custom number of MFs."""
        X = np.random.uniform(-2, 2, (15, 3))

        model = QuickANFIS.for_regression(X, n_mfs=2)

        assert isinstance(model, ANFIS)
        assert model.n_inputs == 3
        assert model.n_rules == 2 * 2 * 2  # 2 MFs per input

    def test_for_regression_invalid_input(self):
        """Test regression with invalid input raises error."""
        X = np.random.uniform(-1, 1, (10,))  # 1D instead of 2D

        with pytest.raises(ValueError, match="Input data must be 2D"):
            QuickANFIS.for_regression(X)

    def test_for_function_approximation_1d(self):
        """Test function approximation for 1D input."""
        input_ranges = [(-2.0, 2.0)]

        model = QuickANFIS.for_function_approximation(input_ranges)

        assert isinstance(model, ANFIS)
        assert model.n_inputs == 1
        assert model.n_rules == 5  # Default 5 MFs

    def test_for_function_approximation_multi_input(self):
        """Test function approximation for multiple inputs."""
        input_ranges = [(-1.0, 1.0), (0.0, 10.0), (-5.0, 5.0)]

        model = QuickANFIS.for_function_approximation(input_ranges, n_mfs=2)

        assert isinstance(model, ANFIS)
        assert model.n_inputs == 3
        assert model.n_rules == 2 * 2 * 2  # 2 MFs per input

    def test_automatic_range_detection(self):
        """Test automatic range detection with margin."""
        X = np.array([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0]])

        # Inspect the created model to verify range detection
        builder = ANFISBuilder()

        # Simulate the range detection logic
        for i in range(X.shape[1]):
            col_data = X[:, i]
            range_min = float(np.min(col_data))
            range_max = float(np.max(col_data))

            # Add some margin (10%)
            margin = (range_max - range_min) * 0.1
            range_min -= margin
            range_max += margin

            builder.add_input(f"x{i + 1}", range_min, range_max, 3, "gaussian")

        model = builder.build()

        assert isinstance(model, ANFIS)
        assert model.n_inputs == 2

    def test_input_range_expansion(self):
        """Test that input ranges are properly expanded with margins."""
        # Data with tight range
        X = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])

        model = QuickANFIS.for_regression(X, n_mfs=2)

        assert isinstance(model, ANFIS)
        assert model.n_inputs == 2

    def test_model_prediction(self):
        """Test that created models can make predictions."""
        X = np.random.uniform(-1, 1, (20, 2))
        model = QuickANFIS.for_regression(X, n_mfs=2)

        # Test prediction on single sample
        pred_single = model.predict(np.array([[0.0, 0.0]]))
        assert isinstance(pred_single, np.ndarray)
        assert pred_single.shape == (1, 1)

        # Test prediction on multiple samples
        pred_multi = model.predict(X[:5])
        assert isinstance(pred_multi, np.ndarray)
        assert pred_multi.shape == (5, 1)

    def test_different_mf_types(self):
        """Test creating models with different MF types."""
        X = np.random.uniform(-1, 1, (10, 2))

        # Test with different MF types
        for mf_type in [
            "gaussian",
            "triangular",
            "trapezoidal",
            "bell",
            "sigmoidal",
            "sshape",
            "zshape",
            "pi",
        ]:
            model = QuickANFIS.for_regression(X, n_mfs=2, mf_type=mf_type)
            assert isinstance(model, ANFIS)
            assert model.n_inputs == 2
