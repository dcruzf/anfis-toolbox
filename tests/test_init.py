"""Tests for package initialization and imports."""

import pytest

import anfis_toolbox


class TestPackageInit:
    """Test package initialization and version."""

    def test_version(self):
        """Test that package has version."""
        assert hasattr(anfis_toolbox, "__version__")
        assert isinstance(anfis_toolbox.__version__, str)
        assert len(anfis_toolbox.__version__) > 0

    def test_core_imports(self):
        """Test that core components are importable."""
        # Core ANFIS components
        assert hasattr(anfis_toolbox, "ANFIS")
        assert hasattr(anfis_toolbox, "ANFISBuilder")
        assert hasattr(anfis_toolbox, "QuickANFIS")

        # Membership functions
        assert hasattr(anfis_toolbox, "GaussianMF")
        assert hasattr(anfis_toolbox, "TriangularMF")
        assert hasattr(anfis_toolbox, "TrapezoidalMF")
        assert hasattr(anfis_toolbox, "BellMF")
        assert hasattr(anfis_toolbox, "SigmoidalMF")
        assert hasattr(anfis_toolbox, "PiMF")

        # Layers
        assert hasattr(anfis_toolbox, "MembershipLayer")
        assert hasattr(anfis_toolbox, "RuleLayer")
        assert hasattr(anfis_toolbox, "NormalizationLayer")
        assert hasattr(anfis_toolbox, "ConsequentLayer")

        # Logging
        assert hasattr(anfis_toolbox, "setup_logging")
        assert hasattr(anfis_toolbox, "enable_training_logs")
        assert hasattr(anfis_toolbox, "disable_training_logs")

    def test_all_exports(self):
        """Test that all exported items are available."""
        for item in anfis_toolbox.__all__:
            assert hasattr(anfis_toolbox, item), f"Missing export: {item}"

    def test_core_functionality_without_optional_deps(self):
        """Test that core functionality works without optional dependencies."""
        # These should work regardless of optional dependencies
        # Test creating a simple model (should not require sklearn or matplotlib)
        import numpy as np

        from anfis_toolbox import ANFIS, QuickANFIS

        X = np.random.uniform(-1, 1, (10, 2))
        model = QuickANFIS.for_regression(X, n_mfs=2)

        assert isinstance(model, ANFIS)


class TestOptionalImports:
    """Test optional import behavior."""

    def test_validation_imports_with_sklearn(self):
        """Test validation imports when sklearn is available."""
        try:
            import sklearn  # noqa: F401

            # If sklearn is available, these should be the real classes
            from anfis_toolbox import ANFISValidator

            # Check that they're not the dummy classes
            assert ANFISValidator.__doc__ != "Dummy class for validation when scikit-learn is not available."

        except ImportError:
            # If sklearn not available, skip this test
            pytest.skip("scikit-learn not available")

    def test_visualization_imports_with_matplotlib(self):
        """Test visualization imports when matplotlib is available."""
        try:
            import matplotlib  # noqa: F401

            # If matplotlib is available, these should be the real classes
            from anfis_toolbox import ANFISVisualizer

            # Check that they're not the dummy classes
            assert ANFISVisualizer.__doc__ != "Dummy class for visualization when matplotlib is not available."

        except ImportError:
            # If matplotlib not available, skip this test
            pytest.skip("matplotlib not available")

    def test_basic_imports_without_optional_deps(self):
        """Test that basic functionality works without optional dependencies."""
        # Test that we can import core modules
        import anfis_toolbox

        assert hasattr(anfis_toolbox, "ANFIS")
        assert hasattr(anfis_toolbox, "GaussianMF")
        assert hasattr(anfis_toolbox, "ANFISBuilder")
        assert hasattr(anfis_toolbox, "QuickANFIS")

    def test_optional_import_flags(self):
        """Test that optional import flags are properly set."""
        import anfis_toolbox

        # These should be boolean flags
        assert isinstance(anfis_toolbox._HAS_VALIDATION, bool)
        assert isinstance(anfis_toolbox._HAS_VISUALIZATION, bool)

    def test_has_visualization_flag(self):
        """Test _HAS_VISUALIZATION flag is set correctly."""
        try:
            import matplotlib  # noqa: F401

            assert anfis_toolbox._HAS_VISUALIZATION is True
        except ImportError:
            assert anfis_toolbox._HAS_VISUALIZATION is False


class TestModuleStructure:
    """Test module structure and organization."""

    def test_submodules_importable(self):
        """Test that submodules can be imported."""
        # These should always be importable
        from anfis_toolbox import builders, layers, logging_config, membership, model

        assert builders is not None
        assert layers is not None
        assert logging_config is not None
        assert membership is not None
        assert model is not None

    def test_membership_functions_inheritance(self):
        """Test that all MF classes inherit from base class."""
        from anfis_toolbox.membership import MembershipFunction

        mf_classes = [
            anfis_toolbox.GaussianMF,
            anfis_toolbox.TriangularMF,
            anfis_toolbox.TrapezoidalMF,
            anfis_toolbox.BellMF,
            anfis_toolbox.SigmoidalMF,
            anfis_toolbox.PiMF,
        ]

        for mf_class in mf_classes:
            assert issubclass(mf_class, MembershipFunction)

    def test_layer_classes_exist(self):
        """Test that all layer classes are properly defined."""
        layer_classes = [
            anfis_toolbox.MembershipLayer,
            anfis_toolbox.RuleLayer,
            anfis_toolbox.NormalizationLayer,
            anfis_toolbox.ConsequentLayer,
        ]

        for layer_class in layer_classes:
            assert layer_class is not None
            assert hasattr(layer_class, "__init__")

    def test_builder_classes_exist(self):
        """Test that builder classes are properly defined."""
        assert anfis_toolbox.ANFISBuilder is not None
        assert anfis_toolbox.QuickANFIS is not None

        # Test that they have expected methods
        assert hasattr(anfis_toolbox.ANFISBuilder, "add_input")
        assert hasattr(anfis_toolbox.ANFISBuilder, "build")
        assert hasattr(anfis_toolbox.QuickANFIS, "for_regression")
        assert hasattr(anfis_toolbox.QuickANFIS, "for_function_approximation")

    def test_logging_functions_exist(self):
        """Test that logging functions are available."""
        logging_funcs = [
            anfis_toolbox.setup_logging,
            anfis_toolbox.enable_training_logs,
            anfis_toolbox.disable_training_logs,
        ]

        for func in logging_funcs:
            assert callable(func)


class TestIntegration:
    """Test integration between components."""

    def test_model_creation_integration(self):
        """Test that model creation works end-to-end."""
        import numpy as np

        # Create data
        X = np.random.uniform(-1, 1, (20, 2))

        # Test QuickANFIS creation
        model = anfis_toolbox.QuickANFIS.for_regression(X, n_mfs=2)
        assert isinstance(model, anfis_toolbox.ANFIS)

        # Test ANFISBuilder creation
        builder = anfis_toolbox.ANFISBuilder()
        builder.add_input("x1", -1, 1, n_mfs=2, mf_type="gaussian")
        builder.add_input("x2", -1, 1, n_mfs=2, mf_type="gaussian")
        model2 = builder.build()
        assert isinstance(model2, anfis_toolbox.ANFIS)

        # Test basic predictions (without training)
        pred = model.predict(X[:3])
        assert isinstance(pred, np.ndarray)
        assert pred.shape[0] == 3

    def test_membership_function_integration(self):
        """Test that membership functions work with models."""
        import numpy as np

        # Create MFs manually
        mf1 = anfis_toolbox.GaussianMF(mean=0.0, sigma=1.0)
        mf2 = anfis_toolbox.TriangularMF(a=-1.0, b=0.0, c=1.0)

        assert isinstance(mf1, anfis_toolbox.GaussianMF)
        assert isinstance(mf2, anfis_toolbox.TriangularMF)

        # Test MF evaluation
        x = np.array([0.0, 0.5, 1.0])
        result = mf1.forward(x)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_logging_integration(self):
        """Test that logging setup works."""
        # Test that logging functions can be called without error
        try:
            anfis_toolbox.setup_logging()
            anfis_toolbox.enable_training_logs()
            anfis_toolbox.disable_training_logs()
        except Exception as e:
            pytest.fail(f"Logging integration failed: {e}")

    def test_error_handling(self):
        """Test error handling in package initialization."""
        # Test that package handles missing optional dependencies gracefully
        assert hasattr(anfis_toolbox, "ANFISValidator")
        assert hasattr(anfis_toolbox, "ANFISVisualizer")

        # These may be dummy classes or real classes depending on dependencies
        # But they should always exist in the namespace
