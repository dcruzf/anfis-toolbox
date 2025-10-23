"""Tests for configuration management utilities."""

import tempfile
from pathlib import Path

import pytest

from anfis_toolbox.config import (
    PREDEFINED_CONFIGS,
    ANFISConfig,
    ANFISModelManager,
    create_config_from_preset,
    list_presets,
)
from anfis_toolbox.model import TSKANFIS


class TestANFISConfig:
    """Test cases for ANFISConfig class."""

    def test_init(self):
        """Test config initialization."""
        config = ANFISConfig()

        assert config.config["inputs"] == {}
        assert config.config["training"]["method"] == "hybrid"
        assert config.config["training"]["epochs"] == 50
        assert config.config["training"]["learning_rate"] == 0.01
        assert config.config["training"]["verbose"] is False

    def test_add_input_config(self):
        """Test adding input configuration."""
        config = ANFISConfig()

        result = config.add_input_config("x1", -1.0, 1.0, n_mfs=3, mf_type="gaussian", overlap=0.5)

        # Test method chaining
        assert result is config

        # Test configuration was added
        assert "x1" in config.config["inputs"]
        input_config = config.config["inputs"]["x1"]
        assert input_config["range_min"] == -1.0
        assert input_config["range_max"] == 1.0
        assert input_config["n_mfs"] == 3
        assert input_config["mf_type"] == "gaussian"
        assert input_config["overlap"] == 0.5

    def test_add_multiple_inputs(self):
        """Test adding multiple input configurations."""
        config = ANFISConfig()

        config.add_input_config("x1", -2.0, 2.0, n_mfs=3, mf_type="gaussian")
        config.add_input_config("x2", 0.0, 10.0, n_mfs=4, mf_type="triangular")

        assert len(config.config["inputs"]) == 2
        assert "x1" in config.config["inputs"]
        assert "x2" in config.config["inputs"]

        assert config.config["inputs"]["x1"]["n_mfs"] == 3
        assert config.config["inputs"]["x2"]["n_mfs"] == 4

    def test_set_training_config(self):
        """Test setting training configuration."""
        config = ANFISConfig()

        result = config.set_training_config(method="backprop", epochs=100, learning_rate=0.02, verbose=False)

        # Test method chaining
        assert result is config

        # Test configuration was updated
        training_config = config.config["training"]
        assert training_config["method"] == "backprop"
        assert training_config["epochs"] == 100
        assert training_config["learning_rate"] == 0.02
        assert training_config["verbose"] is False

    def test_set_training_config_partial(self):
        """Test partial training configuration update."""
        config = ANFISConfig()

        config.set_training_config(epochs=200)

        training_config = config.config["training"]
        assert training_config["method"] == "hybrid"  # Default unchanged
        assert training_config["epochs"] == 200  # Updated
        assert training_config["learning_rate"] == 0.01  # Default unchanged

    def test_build_model_no_inputs(self):
        """Test building model without inputs raises error."""
        config = ANFISConfig()

        with pytest.raises(ValueError, match="No inputs configured"):
            config.build_model()

    def test_build_model_with_inputs(self):
        """Test building model with configured inputs."""
        config = ANFISConfig()
        config.add_input_config("x1", -1.0, 1.0, n_mfs=2, mf_type="gaussian")
        config.add_input_config("x2", 0.0, 2.0, n_mfs=2, mf_type="gaussian")

        model = config.build_model()

        assert isinstance(model, TSKANFIS)
        assert model.n_inputs == 2
        assert model.n_rules == 4  # 2 * 2 = 4 rules

    def test_save_and_load(self):
        """Test saving and loading configuration."""
        config = ANFISConfig()
        config.add_input_config("x1", -1.0, 1.0, n_mfs=3, mf_type="gaussian")
        config.set_training_config(epochs=100, learning_rate=0.02)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "config.json"

            # Save configuration
            config.save(filepath)

            # Verify file was created
            assert filepath.exists()

            # Load configuration
            loaded_config = ANFISConfig.load(filepath)

            # Verify loaded config matches original
            assert loaded_config.config == config.config
            assert len(loaded_config.config["inputs"]) == 1
            assert loaded_config.config["training"]["epochs"] == 100

    def test_save_creates_directory(self):
        """Test that save creates directory if it doesn't exist."""
        config = ANFISConfig()
        config.add_input_config("x1", -1.0, 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "config.json"

            # Directory doesn't exist initially
            assert not filepath.parent.exists()

            # Save should create directory
            config.save(filepath)

            # Verify directory and file were created
            assert filepath.parent.exists()
            assert filepath.exists()

    def test_save_with_tmp_path(self, tmp_path):
        """Test saving config file using tmp_path fixture."""
        config = ANFISConfig()
        config.add_input_config("x1", -2.0, 2.0, n_mfs=4, mf_type="triangular")
        config.set_training_config(method="backprop", epochs=75)

        config_file = tmp_path / "test_config.json"

        # Save config
        config.save(config_file)

        # Test file creation
        assert config_file.exists()
        assert config_file.is_file()
        assert config_file.stat().st_size > 0  # File has content

        # Test file content
        import json

        with open(config_file) as f:
            saved_data = json.load(f)

        assert "inputs" in saved_data
        assert "x1" in saved_data["inputs"]
        assert saved_data["training"]["method"] == "backprop"
        assert saved_data["training"]["epochs"] == 75

    def test_save_creates_nested_directories(self, tmp_path):
        """Test that save creates deeply nested parent directories."""
        config = ANFISConfig()
        config.add_input_config("test", 0, 1, 2)

        # Create deeply nested path
        nested_file = tmp_path / "deep" / "nested" / "path" / "structure" / "config.json"

        # Verify parent directories don't exist
        assert not nested_file.parent.exists()

        # Save should create all parent directories
        config.save(nested_file)

        # Verify all directories were created
        assert nested_file.parent.exists()
        assert nested_file.exists()

        # Verify all intermediate directories exist
        assert (tmp_path / "deep").exists()
        assert (tmp_path / "deep" / "nested").exists()
        assert (tmp_path / "deep" / "nested" / "path").exists()
        assert (tmp_path / "deep" / "nested" / "path" / "structure").exists()

    def test_save_overwrites_existing_file(self, tmp_path):
        """Test that save overwrites existing file."""
        config1 = ANFISConfig()
        config1.add_input_config("x1", -1, 1, 2)

        config2 = ANFISConfig()
        config2.add_input_config("x2", -2, 2, 3)
        config2.set_training_config(epochs=999)  # Different config to ensure different content

        config_file = tmp_path / "overwrite_test.json"

        # Save first config
        config1.save(config_file)

        # Save second config (overwrite)
        config2.save(config_file)

        # File should still exist but content changed
        assert config_file.exists()

        # Verify content is from second config
        import json

        with open(config_file) as f:
            data = json.load(f)
        assert "x2" in data["inputs"]
        assert "x1" not in data["inputs"]
        assert data["training"]["epochs"] == 999

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nonexistent.json"

            with pytest.raises(FileNotFoundError):
                ANFISConfig.load(filepath)

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ANFISConfig()
        config.add_input_config("x1", -1.0, 1.0, n_mfs=3)
        config.set_training_config(epochs=100)

        config_dict = config.to_dict()

        # Should be a copy, not reference
        assert config_dict is not config.config

        # Check content is the same initially
        assert config_dict["training"]["epochs"] == 100

        # Modifying copy shouldn't affect original
        config_dict["training"]["epochs"] = 999
        assert config.config["training"]["epochs"] == 100  # Original unchanged    def test_repr(self):
        """Test string representation."""
        config = ANFISConfig()
        config.add_input_config("x1", -1.0, 1.0, n_mfs=3)
        config.add_input_config("x2", 0.0, 2.0, n_mfs=2)
        config.set_training_config(method="backprop")

        repr_str = repr(config)

        assert "ANFISConfig" in repr_str
        assert "inputs=2" in repr_str
        assert "total_mfs=5" in repr_str  # 3 + 2 = 5
        assert "method=backprop" in repr_str

    def test_repr_empty_config(self):
        """Test string representation with empty config."""
        config = ANFISConfig()

        repr_str = repr(config)

        assert "ANFISConfig" in repr_str
        assert "inputs=0" in repr_str
        assert "total_mfs=0" in repr_str
        assert "method=hybrid" in repr_str


class TestANFISModelManager:
    """Test cases for ANFISModelManager class."""

    def create_simple_model(self):
        """Create a simple ANFIS model for testing."""
        import numpy as np

        from anfis_toolbox.builders import ANFISBuilder

        X = np.random.uniform(-1, 1, (10, 2))
        builder = ANFISBuilder()
        for i in range(X.shape[1]):
            col = X[:, i]
            rmin = float(np.min(col))
            rmax = float(np.max(col))
            margin = (rmax - rmin) * 0.1
            builder.add_input(f"x{i + 1}", rmin - margin, rmax + margin, n_mfs=2, mf_type="gaussian")
        return builder.build()

    def test_save_model_basic(self):
        """Test basic model saving functionality."""
        model = self.create_simple_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"

            ANFISModelManager.save_model(model, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_save_model_with_config(self):
        """Test saving model with configuration (now working with standardized interface)."""
        model = self.create_simple_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"

            # This should save both the model and config successfully
            ANFISModelManager.save_model(model, filepath, include_config=True)

            # Model file should always be created
            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Config file should NOW be created because we standardized the interface
            config_file = filepath.with_suffix(".config.json")
            assert config_file.exists()
            assert config_file.stat().st_size > 0

            # Verify the config file contains valid JSON
            import json

            with open(config_file) as f:
                config_data = json.load(f)
            assert "model_info" in config_data
            assert "membership_functions" in config_data

    def test_save_model_config_file_creation(self, tmp_path):
        """Test that model config file is created with standardized interface."""
        model = self.create_simple_model()

        model_file = tmp_path / "test_model.pkl"

        # Save model with config - this should work now with standardized interface
        ANFISModelManager.save_model(model, model_file, include_config=True)

        # Model file should exist
        assert model_file.exists()

        # Config file should NOW exist because we standardized the interface
        config_file = model_file.with_suffix(".config.json")
        assert config_file.exists()

    def test_save_model_config_success_path(self, tmp_path):
        """Test successful config file creation by mocking _extract_config."""
        model = self.create_simple_model()
        model_file = tmp_path / "test_model.pkl"

        # Mock _extract_config to return valid config
        import unittest.mock

        mock_config = {
            "model_info": {"n_inputs": 2, "n_rules": 4, "input_names": ["x1", "x2"]},
            "membership_functions": {"x1": [], "x2": []},
        }

        with unittest.mock.patch.object(ANFISModelManager, "_extract_config", return_value=mock_config):
            # This should successfully create both files
            ANFISModelManager.save_model(model, model_file, include_config=True)

            # Both files should exist
            assert model_file.exists()

            config_file = model_file.with_suffix(".config.json")
            assert config_file.exists()

            # Test the specific lines 167-168: JSON dump with indent
            import json

            with open(config_file) as f:
                content = f.read()
                # Verify it's properly indented JSON (from lines 167-168)
                assert content.count("\n") > 3  # Multi-line due to indent=2

                # Parse it back to verify it's valid JSON
                f.seek(0)
                config_data = json.load(f)

            # Verify expected structure
            assert config_data == mock_config

    def test_save_model_without_config(self):
        """Test saving model without configuration."""
        model = self.create_simple_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"

            ANFISModelManager.save_model(model, filepath, include_config=False)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_load_model(self):
        """Test loading saved model."""
        model = self.create_simple_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"

            # Save model
            ANFISModelManager.save_model(model, filepath)

            # Load model
            loaded_model = ANFISModelManager.load_model(filepath)

            # Verify model was loaded correctly
            assert isinstance(loaded_model, TSKANFIS)
            assert loaded_model.n_inputs == model.n_inputs
            assert loaded_model.n_rules == model.n_rules

    def test_load_nonexistent_model(self):
        """Test loading non-existent model raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nonexistent.pkl"

            with pytest.raises(FileNotFoundError):
                ANFISModelManager.load_model(filepath)

    def test_save_and_load_model(self):
        """Test saving and loading model with manager."""
        model = self.create_simple_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            ANFISModelManager.save_model(model, filepath)

            # Test file was created
            assert filepath.exists()

            # Test loading model
            loaded_model = ANFISModelManager.load_model(filepath)
            assert isinstance(loaded_model, TSKANFIS)
            assert loaded_model.n_inputs == model.n_inputs
            assert loaded_model.n_rules == model.n_rules


class TestExtractConfig:
    """Tests for ANFISModelManager._extract_config method."""

    def create_mock_membership_function(self, mf_type="GaussianMF"):
        """Create a mock membership function for testing."""

        class MockMF:
            def __init__(self, mf_type, parameters):
                self.__class__.__name__ = mf_type
                self.parameters = parameters

        if mf_type == "GaussianMF":
            return MockMF("GaussianMF", {"mean": 0.0, "std": 1.0})
        elif mf_type == "TriangularMF":
            return MockMF("TriangularMF", {"a": -1.0, "b": 0.0, "c": 1.0})
        else:
            return MockMF(mf_type, {"param1": 1.0, "param2": 2.0})

    def create_mock_model(self, n_inputs=2, n_rules=4, input_names=None, mf_types=None):
        """Create a mock ANFIS model for testing _extract_config."""
        if input_names is None:
            input_names = [f"x{i + 1}" for i in range(n_inputs)]

        if mf_types is None:
            mf_types = ["GaussianMF"] * n_inputs

        class MockMembershipLayer:
            def __init__(self, input_names, mf_types):
                self.membership_functions = {}
                for i, input_name in enumerate(input_names):
                    # Create 2 MFs per input for testing
                    mf_type = mf_types[i] if i < len(mf_types) else "GaussianMF"
                    self.membership_functions[input_name] = [
                        self.create_mock_mf(mf_type, f"{input_name}_mf1"),
                        self.create_mock_mf(mf_type, f"{input_name}_mf2"),
                    ]

            def create_mock_mf(self, mf_type, name):
                class MockMF:
                    def __init__(self, mf_type, name):
                        self.__class__.__name__ = mf_type
                        if mf_type == "GaussianMF":
                            self.parameters = {"mean": float(hash(name) % 10), "std": 1.5}
                        elif mf_type == "TriangularMF":
                            base = float(hash(name) % 5)
                            self.parameters = {"a": base - 1, "b": base, "c": base + 1}
                        else:
                            self.parameters = {"param": float(hash(name) % 10)}

                return MockMF(mf_type, name)

        class MockModel:
            def __init__(self, n_inputs, n_rules, input_names, mf_types):
                self.n_inputs = n_inputs
                self.n_rules = n_rules
                self.input_names = input_names
                self.membership_layer = MockMembershipLayer(input_names, mf_types)

            @property
            def membership_functions(self):
                """Provide standardized interface to membership functions."""
                return self.membership_layer.membership_functions

        return MockModel(n_inputs, n_rules, input_names, mf_types)

    def test_extract_config_basic_structure(self):
        """Test that _extract_config returns correct basic structure."""
        model = self.create_mock_model(n_inputs=2, n_rules=4, input_names=["x1", "x2"])

        config = ANFISModelManager._extract_config(model)

        # Check top-level structure
        assert isinstance(config, dict)
        assert "model_info" in config
        assert "membership_functions" in config

        # Check model_info structure
        model_info = config["model_info"]
        assert model_info["n_inputs"] == 2
        assert model_info["n_rules"] == 4
        assert model_info["input_names"] == ["x1", "x2"]

        # Check membership_functions structure
        mfs = config["membership_functions"]
        assert isinstance(mfs, dict)
        assert "x1" in mfs
        assert "x2" in mfs

    def test_extract_config_membership_functions(self):
        """Test that membership functions are extracted correctly."""
        model = self.create_mock_model(
            n_inputs=2, n_rules=4, input_names=["input1", "input2"], mf_types=["GaussianMF", "TriangularMF"]
        )

        config = ANFISModelManager._extract_config(model)
        mfs = config["membership_functions"]

        # Check input1 (GaussianMF)
        input1_mfs = mfs["input1"]
        assert len(input1_mfs) == 2  # We create 2 MFs per input

        for mf_info in input1_mfs:
            assert "type" in mf_info
            assert "parameters" in mf_info
            assert mf_info["type"] == "GaussianMF"
            assert "mean" in mf_info["parameters"]
            assert "std" in mf_info["parameters"]

        # Check input2 (TriangularMF)
        input2_mfs = mfs["input2"]
        assert len(input2_mfs) == 2

        for mf_info in input2_mfs:
            assert mf_info["type"] == "TriangularMF"
            assert "a" in mf_info["parameters"]
            assert "b" in mf_info["parameters"]
            assert "c" in mf_info["parameters"]

    def test_extract_config_single_input(self):
        """Test _extract_config with single input model."""
        model = self.create_mock_model(n_inputs=1, n_rules=3, input_names=["temperature"], mf_types=["GaussianMF"])

        config = ANFISModelManager._extract_config(model)

        assert config["model_info"]["n_inputs"] == 1
        assert config["model_info"]["n_rules"] == 3
        assert config["model_info"]["input_names"] == ["temperature"]

        assert len(config["membership_functions"]) == 1
        assert "temperature" in config["membership_functions"]

    def test_extract_config_many_inputs(self):
        """Test _extract_config with multiple inputs."""
        input_names = ["x1", "x2", "x3", "x4", "x5"]
        model = self.create_mock_model(
            n_inputs=5,
            n_rules=32,
            input_names=input_names,
            mf_types=["GaussianMF", "TriangularMF", "GaussianMF", "TriangularMF", "GaussianMF"],
        )

        config = ANFISModelManager._extract_config(model)

        assert config["model_info"]["n_inputs"] == 5
        assert config["model_info"]["n_rules"] == 32
        assert config["model_info"]["input_names"] == input_names

        assert len(config["membership_functions"]) == 5
        for input_name in input_names:
            assert input_name in config["membership_functions"]
            assert len(config["membership_functions"][input_name]) == 2  # 2 MFs per input

    def test_extract_config_parameters_are_copied(self):
        """Test that MF parameters are properly copied (not referenced)."""
        model = self.create_mock_model(n_inputs=1, input_names=["x"])

        config = ANFISModelManager._extract_config(model)

        # Get the original MF from the model
        original_mf = model.membership_layer.membership_functions["x"][0]

        # Get the extracted parameters
        extracted_params = config["membership_functions"]["x"][0]["parameters"]

        # Modify the extracted parameters
        if "mean" in extracted_params:
            extracted_params["mean"] = 999.0
        else:
            extracted_params["param"] = 999.0

        # Original parameters should remain unchanged (they were copied)
        if hasattr(original_mf, "parameters"):
            assert 999.0 not in original_mf.parameters.values()

    def test_extract_config_with_missing_attributes(self):
        """Test _extract_config behavior with missing model attributes."""

        # Create a model with missing membership_layer
        class BrokenModel:
            def __init__(self):
                self.n_inputs = 2
                self.n_rules = 4
                # Missing membership_layer

        broken_model = BrokenModel()

        with pytest.raises(AttributeError):
            ANFISModelManager._extract_config(broken_model)

    def test_extract_config_with_empty_membership_functions(self):
        """Test _extract_config with model that has no membership functions."""

        class EmptyMembershipLayer:
            def __init__(self):
                self.membership_functions = {}

        class EmptyModel:
            def __init__(self):
                self.n_inputs = 0
                self.n_rules = 0
                self.input_names = []
                self.membership_layer = EmptyMembershipLayer()

            @property
            def membership_functions(self):
                """Provide standardized interface to membership functions."""
                return {}

        empty_model = EmptyModel()

        config = ANFISModelManager._extract_config(empty_model)

        assert config["model_info"]["n_inputs"] == 0
        assert config["model_info"]["n_rules"] == 0
        assert config["model_info"]["input_names"] == []
        assert config["membership_functions"] == {}

    def test_extract_config_different_mf_types(self):
        """Test _extract_config with different membership function types."""
        # Test with custom MF type
        model = self.create_mock_model(
            n_inputs=3, input_names=["a", "b", "c"], mf_types=["CustomMF1", "CustomMF2", "CustomMF3"]
        )

        config = ANFISModelManager._extract_config(model)

        # Check that different MF types are preserved
        mfs = config["membership_functions"]

        # Each input should have MFs of the specified type
        for input_name in ["a", "b", "c"]:
            mf_list = mfs[input_name]
            assert len(mf_list) == 2  # 2 MFs per input

            for mf_info in mf_list:
                assert mf_info["type"] in ["CustomMF1", "CustomMF2", "CustomMF3"]
                assert isinstance(mf_info["parameters"], dict)


def test_create_config_from_preset_success_and_list_presets():
    # Ensure list_presets exposes all descriptions
    presets = list_presets()
    assert isinstance(presets, dict)
    for name, info in PREDEFINED_CONFIGS.items():
        assert name in presets
        assert presets[name] == info["description"]

    # Build config from a known preset and verify content
    cfg = create_config_from_preset("1d_function")
    assert isinstance(cfg, ANFISConfig)
    d = cfg.to_dict()
    assert d["inputs"].keys() == {"x"}
    assert d["training"]["method"] == PREDEFINED_CONFIGS["1d_function"]["training"]["method"]


def test_create_config_from_preset_invalid_name():
    import pytest

    with pytest.raises(ValueError) as exc:
        create_config_from_preset("does_not_exist")
    # Error message lists available presets
    for name in PREDEFINED_CONFIGS.keys():
        assert name in str(exc.value)


def test_save_model_config_extraction_failure_logs_warning(tmp_path, caplog):
    # Hit the warning path when _extract_config raises (lines 169-170)
    from unittest.mock import patch

    model = TestANFISModelManager().create_simple_model()
    model_file = tmp_path / "model.pkl"
    cfg_file = model_file.with_suffix(".config.json")

    with patch.object(ANFISModelManager, "_extract_config", side_effect=RuntimeError("boom")):
        caplog.set_level("WARNING")
        ANFISModelManager.save_model(model, model_file, include_config=True)
        # Model is still saved
        assert model_file.exists()
        # Config should not be created and a warning should be logged
        assert not cfg_file.exists()
        assert any("Could not save model configuration" in rec.message for rec in caplog.records)
