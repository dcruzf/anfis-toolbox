"""Tests for ANFIS model."""

import logging

import numpy as np
import pytest

from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.model import ANFIS
from anfis_toolbox.optim import HybridTrainer, SGDTrainer

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


def test_backprop_updates_params_via_trainer(sample_anfis):
    """Ensure backprop updates occur via SGDTrainer."""
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[1.0], [2.0]])
    params_initial = sample_anfis.get_parameters()
    trainer = SGDTrainer(learning_rate=0.1, epochs=1, verbose=False)
    history = trainer.fit(sample_anfis, x, y)
    train_losses = history["train"]
    assert len(train_losses) == 1 and np.isfinite(train_losses[0]) and train_losses[0] >= 0
    params_after = sample_anfis.get_parameters()
    param_changed = not np.allclose(params_initial["consequent"], params_after["consequent"])
    if not param_changed:
        for name in ["x1", "x2"]:
            for i in range(len(params_initial["membership"][name])):
                if (
                    params_initial["membership"][name][i]["mean"] != params_after["membership"][name][i]["mean"]
                    or params_initial["membership"][name][i]["sigma"] != params_after["membership"][name][i]["sigma"]
                ):
                    param_changed = True
                    break
    assert param_changed


def test_anfis_fit(sample_anfis):
    """Test ANFIS training over multiple epochs."""
    # Create simple training data (linear function)
    np.random.seed(42)
    x = np.random.randn(20, 2)
    y = np.sum(x, axis=1, keepdims=True) + 0.1 * np.random.randn(20, 1)  # y = x1 + x2 + noise

    # Train the model
    history = sample_anfis.fit(x, y, epochs=10, learning_rate=0.01, verbose=False)
    losses = history["train"]

    # Check that we got the right number of loss values
    assert len(losses) == 10

    # Check that all losses are finite and non-negative
    assert all(np.isfinite(loss) and loss >= 0 for loss in losses)

    # Losses should generally decrease (may not be monotonic due to noise)
    # We'll just check that the final loss is reasonable
    assert losses[-1] < 100.0  # Should be much lower for this simple problem


def test_trainer_api_sgd_and_hybrid_and_fit_compat():
    # Simple data
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] - 0.5 * X[:, 1]).reshape(-1, 1)

    model1 = _make_simple_model()
    # Use trainer argument on fit (sklearn-style entry)
    sgd = SGDTrainer(learning_rate=0.05, epochs=5, batch_size=None, shuffle=False, verbose=False)
    history1 = model1.fit(X, y, trainer=sgd)
    assert len(history1["train"]) == 5

    model2 = _make_simple_model()
    # Explicit HybridTrainer
    hyb = HybridTrainer(learning_rate=0.1, epochs=5, verbose=False)
    history2 = hyb.fit(model2, X, y)
    assert len(history2["train"]) == 5

    # Backward-compatible path still works (no trainer param)
    model3 = _make_simple_model()
    history3 = model3.fit(X, y, epochs=5, learning_rate=0.05, verbose=False)
    assert len(history3["train"]) == 5


def test_hybrid_trainer_accepts_1d_y_and_reshapes():
    # y provided as 1D; HybridTrainer should reshape internally to (n, 1)
    rng = np.random.default_rng(123)
    X = rng.normal(size=(20, 2))
    y = X[:, 0] + X[:, 1]  # 1D target

    model = _make_simple_model()
    trainer = HybridTrainer(learning_rate=0.05, epochs=3, verbose=False)
    history = trainer.fit(model, X, y)
    assert len(history["train"]) == 3


def test_sgd_trainer_minibatch_no_shuffle_and_shuffle_and_1d_y():
    rng = np.random.default_rng(321)
    X = rng.normal(size=(25, 2))
    y = 2 * X[:, 0] - 0.5 * X[:, 1]  # 1D target
    # No shuffle
    model_ns = _make_simple_model()
    sgd_ns = SGDTrainer(learning_rate=0.05, epochs=4, batch_size=5, shuffle=False, verbose=False)
    history_ns = sgd_ns.fit(model_ns, X, y)
    assert len(history_ns["train"]) == 4
    # With shuffle
    model_s = _make_simple_model()
    sgd_s = SGDTrainer(learning_rate=0.05, epochs=4, batch_size=6, shuffle=True, verbose=False)
    history_s = sgd_s.fit(model_s, X, y)
    assert len(history_s["train"]) == 4


def test_tskanfis_fit_forwards_validation_kwargs():
    model = _make_simple_model()
    X = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=float)
    y = (X[:, 0] + 0.25 * X[:, 1]).reshape(-1, 1)
    X_val = np.array([[0.5, 0.5]], dtype=float)
    y_val = (X_val[:, 0] + 0.25 * X_val[:, 1]).reshape(-1, 1)

    captured: dict[str, object] = {}

    class RecordingTrainer:
        def fit(self, model, X_fit, y_fit, **kwargs):
            captured["model"] = model
            captured["X"] = X_fit
            captured["y"] = y_fit
            captured["kwargs"] = kwargs
            return {"train": [0.0], "val": [0.0]}

    trainer = RecordingTrainer()
    history = model.fit(
        X,
        y,
        trainer=trainer,
        validation_data=(X_val, y_val),
        validation_frequency=3,
    )

    assert history == {"train": [0.0], "val": [0.0]}
    assert captured["model"] is model
    np.testing.assert_array_equal(captured["X"], X)
    np.testing.assert_array_equal(captured["y"], y)
    forwarded = captured["kwargs"]
    assert forwarded["validation_frequency"] == 3
    val_X, val_y = forwarded["validation_data"]
    np.testing.assert_array_equal(val_X, X_val)
    np.testing.assert_array_equal(val_y, y_val)


def test_tskanfis_fit_raises_for_non_dict_history():
    model = _make_simple_model()
    X = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=float)
    y = (X[:, 0] + 0.5 * X[:, 1]).reshape(-1, 1)

    class BadTrainer:
        def fit(self, model, X_fit, y_fit, **kwargs):
            return [0.0]

    with pytest.raises(TypeError, match="Trainer.fit must return a TrainingHistory dictionary"):
        model.fit(X, y, trainer=BadTrainer())


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
    _history = model.fit(x, y, epochs=100, learning_rate=0.1, verbose=False)

    # Test prediction accuracy
    x_test = np.array([[-1.5], [0.0], [1.5]])
    y_pred = model.predict(x_test)
    y_true = x_test**2

    # Check that predictions are reasonable (not exact due to limited training)
    mse = np.mean((y_pred - y_true) ** 2)
    assert mse < 2.0  # Should be able to approximate x^2 reasonably well


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
    # Use SGDTrainer single-epoch backprop to simulate one step
    from anfis_toolbox.optim import SGDTrainer

    history = SGDTrainer(learning_rate=0.1, epochs=1, verbose=False).fit(model, x, y)
    loss = history["train"][0]
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

    # Ensure a single-epoch hybrid fit works (via default HybridTrainer)
    history_once = model.fit(x, y, epochs=1, learning_rate=0.1, verbose=False)
    losses_once = history_once["train"]
    assert len(losses_once) == 1 and np.isfinite(losses_once[0]) and losses_once[0] >= 0

    # Test hybrid training over multiple epochs via default fit (HybridTrainer)
    history = model.fit(x, y, epochs=10, learning_rate=0.1, verbose=False)
    losses = history["train"]

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
    history_hybrid = model_hybrid.fit(x, y, epochs=20, learning_rate=0.1, verbose=False)
    history_backprop = model_backprop.fit(x, y, epochs=20, learning_rate=0.1, verbose=False)
    losses_hybrid = history_hybrid["train"]
    losses_backprop = history_backprop["train"]

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


def test_logging_configuration():
    """Test ANFIS logging configuration."""
    from anfis_toolbox.logging_config import disable_training_logs, enable_training_logs, setup_logging

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


def _make_simple_model(n_inputs: int = 2, n_mfs: int = 2) -> ANFIS:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)][:n_mfs]
    return ANFIS(input_mfs)


def test_predict_invalid_shapes():
    model = _make_simple_model(n_inputs=2, n_mfs=2)
    # 1D wrong feature count
    with pytest.raises(ValueError, match="Expected 2 features"):
        model.predict(np.array([1.0, 2.0, 3.0]))
    # 2D wrong feature count
    with pytest.raises(ValueError, match="Expected input with 2 features"):
        model.predict(np.zeros((4, 3)))
    # Higher dimensional input
    with pytest.raises(ValueError, match="Expected input with shape"):
        model.predict(np.zeros((2, 2, 2)))


def test_predict_1d_valid_length_triggers_reshape():
    # Ensure that a 1D array with correct feature count is reshaped and processed
    model = _make_simple_model(n_inputs=2, n_mfs=2)
    x1d = np.array([0.5, -1.2])  # length matches n_inputs
    y1 = model.predict(x1d)
    y2 = model.predict(x1d.reshape(1, -1))
    # Outputs should be identical and shape should be (1, 1)
    np.testing.assert_array_equal(y1, y2)
    assert y1.shape == (1, 1)


def test_str_and_repr_cover():
    model = _make_simple_model()
    s = str(model)
    r = repr(model)
    assert "ANFIS Model" in s
    assert "ANFIS(" in r


def test_apply_membership_gradients_private_helper():
    # Cover ANFIS._apply_membership_gradients branch
    model = _make_simple_model()
    X = np.array([[0.0, 0.0], [1.0, -1.0]])
    # Create some gradients via a minibackward step
    _ = model.forward(X)
    model.backward(np.ones((X.shape[0], 1)))
    params_before = model.get_parameters()
    model._apply_membership_gradients(learning_rate=0.01)
    params_after = model.get_parameters()
    # Membership parameters should have changed for at least one MF
    changed = False
    for name in params_before["membership"]:
        for i, mf_before in enumerate(params_before["membership"][name]):
            mf_after = params_after["membership"][name][i]
            if not (
                np.isclose(mf_before["mean"], mf_after["mean"]) and np.isclose(mf_before["sigma"], mf_after["sigma"])
            ):
                changed = True
                break
        if changed:
            break
    assert changed


def test_fit_logging_branch_and_hybrid_logging(monkeypatch, caplog):
    # Small dataset
    X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [1.5, -0.5]])
    y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
    model = _make_simple_model()

    # Trigger logging with epochs=1 (always logs at least once)
    caplog.set_level("INFO")
    history = model.fit(X, y, epochs=1, learning_rate=0.01, verbose=True)
    losses = history["train"]
    assert len(losses) == 1

    # For hybrid, force the pseudo-inverse fallback to exercise the except path
    def raise_lin_alg_error(*args, **kwargs):  # pragma: no cover - covered by branch
        raise np.linalg.LinAlgError

    monkeypatch.setattr(np.linalg, "solve", raise_lin_alg_error)
    hyb_history = model.fit(X, y, epochs=1, learning_rate=0.01, verbose=True)
    hyb_losses = hyb_history["train"]
    assert len(hyb_losses) == 1


def test_tskanfis_fit_requires_trainer_protocol():
    model = _make_simple_model()
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = (X[:, 0] + X[:, 1]).reshape(-1, 1)

    with pytest.raises(TypeError, match="trainer must implement fit"):
        model.fit(X, y, trainer=object())


def test_membership_functions_property_and_parameter_set_get():
    model = _make_simple_model()
    # Property alias
    assert model.membership_functions is model.input_mfs

    params = model.get_parameters()
    # Modify one consequent parameter and one MF parameter
    _original_consequent = params["consequent"].copy()
    # ensure consequent has correct shape; if uninitialized, run a forward/backward to fill (not needed)
    # tweak membership mean of first MF of x1
    params["membership"]["x1"][0]["mean"] += 0.123
    model.set_parameters(params)
    new_params = model.get_parameters()
    assert np.isclose(new_params["membership"]["x1"][0]["mean"], params["membership"]["x1"][0]["mean"])


def test_set_parameters_without_consequent_updates_only_membership():
    model = _make_simple_model()
    params_before = model.get_parameters()
    # Build a parameters dict without 'consequent' to exercise the 199->203 branch
    params = {"membership": params_before["membership"]}
    # Tweak a membership param
    params["membership"]["x1"][0]["mean"] += 0.5
    model.set_parameters(params)
    params_after = model.get_parameters()
    # Consequent stays the same
    np.testing.assert_array_equal(params_after["consequent"], params_before["consequent"])
    # Membership changed for x1 first MF
    assert params_after["membership"]["x1"][0]["mean"] == params["membership"]["x1"][0]["mean"]


def test_set_parameters_without_membership_updates_only_consequent():
    model = _make_simple_model()
    params_before = model.get_parameters()
    # Build a parameters dict without 'membership' to exercise the 203->exit branch
    new_consequent = np.ones_like(params_before["consequent"]) * 3.14
    params = {"consequent": new_consequent}
    model.set_parameters(params)
    params_after = model.get_parameters()
    # Consequent updated
    np.testing.assert_array_equal(params_after["consequent"], new_consequent)
    # Membership remains unchanged
    assert params_after["membership"] == params_before["membership"]


def test_set_parameters_membership_missing_name_skips_safely():
    model = _make_simple_model()
    params_before = model.get_parameters()
    # Provide membership params only for x1; x2 entry omitted to trigger line 208 'continue'
    partial_membership = {"x1": params_before["membership"]["x1"]}
    # Modify x1
    partial_membership["x1"][0]["mean"] += 0.25
    model.set_parameters({"membership": partial_membership})
    params_after = model.get_parameters()
    # x1 updated
    assert params_after["membership"]["x1"][0]["mean"] == partial_membership["x1"][0]["mean"]
    # x2 unchanged
    assert params_after["membership"]["x2"] == params_before["membership"]["x2"]


def test_hybrid_lsm_fallback_runs_with_pseudoinverse(monkeypatch):
    """Force LinAlgError in LSM to exercise pseudo-inverse fallback and log warning."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(10, 2))
    y = (X[:, 0] + 0.3 * X[:, 1]).reshape(-1, 1)

    model = _make_simple_model()

    # Force np.linalg.solve to raise LinAlgError
    def _raise(*args, **kwargs):  # pragma: no cover - executed by this test
        raise np.linalg.LinAlgError

    monkeypatch.setattr(np.linalg, "solve", _raise)

    # One epoch fit triggers the hybrid fallback path inside HybridTrainer
    history = model.fit(X, y, epochs=1, learning_rate=0.01, verbose=False)
    losses = history["train"]
    # Should complete without error and return a finite loss
    assert len(losses) == 1 and np.isfinite(losses[0])
