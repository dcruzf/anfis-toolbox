"""Tests for ANFIS model."""

import logging

import numpy as np
import pytest

from anfis_toolbox import ANFIS
from anfis_toolbox.membership import GaussianMF

# Disable logging during tests to keep output clean
logging.getLogger("anfis_toolbox").setLevel(logging.CRITICAL)


@pytest.fixture
def sample_anfis():
    """Create a sample ANFIS model for testing."""
    input_mfs = {
        "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
        "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    }
    return ANFIS(input_mfs)


def test_anfis_initialization(sample_anfis):
    """Test ANFIS model initialization."""
    assert sample_anfis.n_inputs == 2
    assert sample_anfis.n_rules == 4  # 2 * 2 = 4 rules
    assert sample_anfis.input_names == ["x1", "x2"]

    # Check that all layers are initialized
    assert sample_anfis.membership_layer is not None
    assert sample_anfis.rule_layer is not None
    assert sample_anfis.normalization_layer is not None
    assert sample_anfis.consequent_layer is not None


def test_anfis_forward_pass(sample_anfis):
    """Test ANFIS forward pass."""
    # Create sample input
    x = np.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]])  # (3, 2)

    # Forward pass
    output = sample_anfis.forward(x)

    # Check output shape
    assert output.shape == (3, 1)

    # Output should be finite
    assert np.all(np.isfinite(output))


def test_anfis_predict(sample_anfis):
    """Test ANFIS predict method."""
    x = np.array([[0.0, 0.0], [1.0, -1.0]])  # (2, 2)

    # Predict should give same result as forward
    output1 = sample_anfis.forward(x)
    output2 = sample_anfis.predict(x)

    np.testing.assert_array_equal(output1, output2)


def test_anfis_backward_pass(sample_anfis):
    """Test ANFIS backward pass."""
    x = np.array([[0.0, 0.0]])  # (1, 2)

    # Forward pass
    output = sample_anfis.forward(x)

    # Create dummy loss gradient
    dL_dy = np.ones_like(output)

    # Backward pass should not raise an error
    sample_anfis.backward(dL_dy)

    # Check that gradients were computed
    gradients = sample_anfis.get_gradients()
    assert "membership" in gradients
    assert "consequent" in gradients

    # Membership function gradients should exist
    for name in ["x1", "x2"]:
        assert name in gradients["membership"]
        assert len(gradients["membership"][name]) == 2  # 2 MFs per input

        for mf_grads in gradients["membership"][name]:
            assert "mean" in mf_grads
            assert "sigma" in mf_grads


def test_anfis_reset_gradients(sample_anfis):
    """Test ANFIS gradient reset functionality."""
    x = np.array([[0.0, 0.0]])

    # Forward and backward pass to create gradients
    output = sample_anfis.forward(x)
    dL_dy = np.ones_like(output)
    sample_anfis.backward(dL_dy)

    # Check that gradients exist
    gradients_before = sample_anfis.get_gradients()
    assert np.any(gradients_before["consequent"] != 0)

    # Reset gradients
    sample_anfis.reset_gradients()

    # Check that gradients are zero
    gradients_after = sample_anfis.get_gradients()
    np.testing.assert_array_equal(gradients_after["consequent"], 0)

    # Check membership function gradients are reset
    for name in ["x1", "x2"]:
        for mf_grads in gradients_after["membership"][name]:
            assert mf_grads["mean"] == 0.0
            assert mf_grads["sigma"] == 0.0


def test_anfis_parameter_management(sample_anfis):
    """Test parameter get/set functionality."""
    # Get initial parameters
    params_initial = sample_anfis.get_parameters()

    # Modify parameters
    params_modified = params_initial.copy()
    params_modified["consequent"] = np.ones_like(params_modified["consequent"])

    # Modify membership parameters
    for name in ["x1", "x2"]:
        for i in range(len(params_modified["membership"][name])):
            params_modified["membership"][name][i]["mean"] = 5.0
            params_modified["membership"][name][i]["sigma"] = 2.0

    # Set modified parameters
    sample_anfis.set_parameters(params_modified)

    # Verify parameters were set
    params_current = sample_anfis.get_parameters()
    np.testing.assert_array_equal(params_current["consequent"], np.ones_like(params_initial["consequent"]))

    # Check membership parameters
    for name in ["x1", "x2"]:
        for i in range(len(params_current["membership"][name])):
            assert params_current["membership"][name][i]["mean"] == 5.0
            assert params_current["membership"][name][i]["sigma"] == 2.0


def test_anfis_train_step(sample_anfis):
    """Test single training step."""
    # Create simple training data
    x = np.array([[0.0, 0.0], [1.0, 1.0]])  # (2, 2)
    y = np.array([[1.0], [2.0]])  # (2, 1)

    # Get initial parameters
    params_initial = sample_anfis.get_parameters()

    # Perform training step
    loss = sample_anfis.train_step(x, y, learning_rate=0.1)

    # Check that loss is a finite number
    assert np.isfinite(loss)
    assert loss >= 0  # MSE is non-negative

    # Check that parameters changed
    params_after = sample_anfis.get_parameters()

    # At least some parameters should have changed
    param_changed = False
    if not np.allclose(params_initial["consequent"], params_after["consequent"]):
        param_changed = True

    for name in ["x1", "x2"]:
        for i in range(len(params_initial["membership"][name])):
            if (
                params_initial["membership"][name][i]["mean"] != params_after["membership"][name][i]["mean"]
                or params_initial["membership"][name][i]["sigma"] != params_after["membership"][name][i]["sigma"]
            ):
                param_changed = True
                break

    assert param_changed, "No parameters were updated during training step"


def test_anfis_fit(sample_anfis):
    """Test ANFIS training over multiple epochs."""
    # Create simple training data (linear function)
    np.random.seed(42)
    x = np.random.randn(20, 2)
    y = np.sum(x, axis=1, keepdims=True) + 0.1 * np.random.randn(20, 1)  # y = x1 + x2 + noise

    # Train the model
    losses = sample_anfis.fit(x, y, epochs=10, learning_rate=0.01, verbose=False)

    # Check that we got the right number of loss values
    assert len(losses) == 10

    # Check that all losses are finite and non-negative
    assert all(np.isfinite(loss) and loss >= 0 for loss in losses)

    # Losses should generally decrease (may not be monotonic due to noise)
    # We'll just check that the final loss is reasonable
    assert losses[-1] < 100.0  # Should be much lower for this simple problem


def test_anfis_nonlinear_function():
    """Test ANFIS on a nonlinear function approximation task."""
    # Create ANFIS with more membership functions for better approximation
    input_mfs = {
        "x": [GaussianMF(mean=-2.0, sigma=1.0), GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=2.0, sigma=1.0)]
    }
    model = ANFIS(input_mfs)

    # Create nonlinear function: y = x^2
    x = np.linspace(-3, 3, 30).reshape(-1, 1)
    y = x**2

    # Train the model
    _losses = model.fit(x, y, epochs=50, learning_rate=0.1, verbose=False)

    # Test prediction accuracy
    x_test = np.array([[-1.5], [0.0], [1.5]])
    y_pred = model.predict(x_test)
    y_true = x_test**2

    # Check that predictions are reasonable (not exact due to limited training)
    mse = np.mean((y_pred - y_true) ** 2)
    assert mse < 1.0  # Should be able to approximate x^2 reasonably well


def test_anfis_string_representations(sample_anfis):
    """Test string representations of ANFIS model."""
    str_repr = str(sample_anfis)
    repr_repr = repr(sample_anfis)

    assert "ANFIS Model" in str_repr
    assert "2" in str_repr  # number of inputs
    assert "4" in str_repr  # number of rules

    assert "ANFIS" in repr_repr
    assert "n_inputs=2" in repr_repr
    assert "n_rules=4" in repr_repr


def test_anfis_edge_cases():
    """Test ANFIS with edge cases."""
    # Single input, single MF per input
    input_mfs = {"x": [GaussianMF(mean=0.0, sigma=1.0)]}
    model = ANFIS(input_mfs)

    assert model.n_inputs == 1
    assert model.n_rules == 1

    # Test forward pass
    x = np.array([[0.0], [1.0], [-1.0]])
    output = model.forward(x)
    assert output.shape == (3, 1)

    # Single training step should work
    y = np.array([[1.0], [2.0], [0.0]])
    loss = model.train_step(x, y)
    assert np.isfinite(loss)


def test_anfis_hybrid_algorithm():
    """Test ANFIS hybrid learning algorithm (original Jang 1993)."""
    # Create simple ANFIS model
    input_mfs = {
        "x1": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
        "x2": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)],
    }
    model = ANFIS(input_mfs)

    # Create simple training data
    np.random.seed(42)
    x = np.random.randn(20, 2)
    y = np.sum(x, axis=1, keepdims=True) + 0.1 * np.random.randn(20, 1)

    # Test hybrid training step
    loss = model.hybrid_train_step(x, y, learning_rate=0.1)
    assert np.isfinite(loss)
    assert loss >= 0

    # Test hybrid training over multiple epochs
    losses = model.fit_hybrid(x, y, epochs=10, learning_rate=0.1, verbose=False)

    assert len(losses) == 10
    assert all(np.isfinite(loss) and loss >= 0 for loss in losses)

    # Should show some improvement
    assert losses[-1] <= losses[0] + 1e-6  # Allow for slight numerical variations


def test_anfis_hybrid_vs_backprop_comparison():
    """Test that both hybrid and backprop algorithms work on same data."""
    input_mfs = {"x": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]}

    # Create identical models
    model_hybrid = ANFIS(input_mfs)
    model_backprop = ANFIS({"x": [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]})

    # Simple quadratic function
    x = np.array([[-2], [-1], [0], [1], [2]], dtype=float)
    y = x**2

    # Train both models
    losses_hybrid = model_hybrid.fit_hybrid(x, y, epochs=20, learning_rate=0.1, verbose=False)
    losses_backprop = model_backprop.fit(x, y, epochs=20, learning_rate=0.1, verbose=False)

    # Both should converge
    assert losses_hybrid[-1] < losses_hybrid[0]
    assert losses_backprop[-1] < losses_backprop[0]

    # Both should make reasonable predictions
    x_test = np.array([[0.5], [1.5]])
    y_pred_hybrid = model_hybrid.predict(x_test)
    y_pred_backprop = model_backprop.predict(x_test)

    assert y_pred_hybrid.shape == (2, 1)
    assert y_pred_backprop.shape == (2, 1)
    assert np.all(np.isfinite(y_pred_hybrid))
    assert np.all(np.isfinite(y_pred_backprop))
    """Test ANFIS logging configuration."""
    from anfis_toolbox import disable_training_logs, enable_training_logs, setup_logging

    # Test enabling training logs
    enable_training_logs()
    logger = logging.getLogger("anfis_toolbox")
    assert logger.level == logging.INFO

    # Test disabling training logs
    disable_training_logs()
    assert logger.level == logging.WARNING

    # Test custom setup
    setup_logging(level="DEBUG")
    assert logger.level == logging.DEBUG

    # Reset to critical level for other tests
    logger.setLevel(logging.CRITICAL)
