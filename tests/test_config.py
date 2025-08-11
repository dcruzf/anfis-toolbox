"""Tests for configuration management utilities."""

import tempfile
from pathlib import Path

import pytest

from anfis_toolbox.config import ANFISConfig, ANFISModelManager
from anfis_toolbox.model import ANFIS


class TestANFISConfig:
    """Test cases for ANFISConfig class."""

    def test_init(self):
        """Test config initialization."""
        config = ANFISConfig()

        assert config.config["inputs"] == {}
        assert config.config["training"]["method"] == "hybrid"
        assert config.config["training"]["epochs"] == 50
        assert config.config["training"]["learning_rate"] == 0.01
        assert config.config["training"]["verbose"] is True

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

        assert isinstance(model, ANFIS)
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

        from anfis_toolbox.builders import QuickANFIS

        X = np.random.uniform(-1, 1, (10, 2))
        return QuickANFIS.for_regression(X, n_mfs=2)

    def test_save_model_basic(self):
        """Test basic model saving functionality."""
        model = self.create_simple_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"

            ANFISModelManager.save_model(model, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_save_model_with_config(self):
        """Test saving model with configuration."""
        model = self.create_simple_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"

            ANFISModelManager.save_model(model, filepath, include_config=True)

            assert filepath.exists()
            # With config should create additional files or larger file
            assert filepath.stat().st_size > 0

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
            assert isinstance(loaded_model, ANFIS)
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
            assert isinstance(loaded_model, ANFIS)
            assert loaded_model.n_inputs == model.n_inputs
            assert loaded_model.n_rules == model.n_rules
