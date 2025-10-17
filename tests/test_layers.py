import numpy as np
import pytest

from anfis_toolbox.layers import ConsequentLayer, MembershipLayer, NormalizationLayer, RuleLayer
from anfis_toolbox.membership import GaussianMF


@pytest.fixture
def sample_input_mfs():
    """Create sample membership functions for testing."""
    mf_x1 = [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    mf_x2 = [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return {"x1": mf_x1, "x2": mf_x2}


@pytest.fixture
def simple_membership_layer(sample_input_mfs):
    """Create a simple membership layer for testing."""
    return MembershipLayer(sample_input_mfs)


@pytest.fixture
def simple_rule_layer():
    """Create a simple rule layer for testing."""
    return RuleLayer(input_names=["x1", "x2"], mf_per_input=[2, 2])


def test_membership_layer_property(simple_membership_layer):
    """Test MembershipLayer properties."""
    assert simple_membership_layer.membership_functions


def test_membership_layer_forward(simple_membership_layer):
    """Test MembershipLayer forward pass."""
    x = np.array([[0.0, 0.0], [1.0, 1.0]])  # shape (2, 2)

    membership_outputs = simple_membership_layer.forward(x)

    # Check output structure
    assert isinstance(membership_outputs, dict)
    assert set(membership_outputs.keys()) == {"x1", "x2"}

    # Check shapes
    assert membership_outputs["x1"].shape == (2, 2)  # (batch_size, n_mfs)
    assert membership_outputs["x2"].shape == (2, 2)

    # Check that membership values are in [0, 1]
    for name in ["x1", "x2"]:
        assert np.all(membership_outputs[name] >= 0)
        assert np.all(membership_outputs[name] <= 1)


def test_membership_layer_backward(simple_membership_layer, sample_input_mfs):
    """Test MembershipLayer backward pass."""
    x = np.array([[0.0, 0.0]])
    _ = simple_membership_layer.forward(x)

    # Create dummy gradients
    gradients = {"x1": np.ones((1, 2)), "x2": np.ones((1, 2))}

    simple_membership_layer.backward(gradients)

    # Verify that gradients were propagated to membership functions
    for name in ["x1", "x2"]:
        for mf in sample_input_mfs[name]:
            assert "mean" in mf.gradients
            assert "sigma" in mf.gradients
            assert np.isscalar(mf.gradients["mean"])
            assert np.isscalar(mf.gradients["sigma"])


def test_membership_layer_reset(simple_membership_layer, sample_input_mfs):
    """Test MembershipLayer reset functionality."""
    # First, create some gradients
    x = np.array([[0.0, 0.0]])
    simple_membership_layer.forward(x)
    gradients = {"x1": np.ones((1, 2)), "x2": np.ones((1, 2))}
    simple_membership_layer.backward(gradients)

    # Reset
    simple_membership_layer.reset()

    # Verify that all membership functions were reset
    for name in ["x1", "x2"]:
        for mf in sample_input_mfs[name]:
            assert mf.gradients["mean"] == 0.0
            assert mf.gradients["sigma"] == 0.0

    # Verify that cache was cleared
    assert simple_membership_layer.last == {}


def test_rule_layer_forward(simple_rule_layer):
    """Test RuleLayer forward pass with new interface."""
    # Create sample membership outputs
    membership_outputs = {
        "x1": np.array([[0.8, 0.2]]),  # (1, 2) - high activation for first MF
        "x2": np.array([[0.6, 0.4]]),  # (1, 2) - moderate activation for first MF
    }

    rule_strengths = simple_rule_layer.forward(membership_outputs)

    # Should have 4 rules (2x2 combinations)
    assert rule_strengths.shape == (1, 4)

    # Manual verification of rule combinations
    expected_strengths = [
        0.8 * 0.6,  # Rule (0, 0): x1_mf0 * x2_mf0
        0.8 * 0.4,  # Rule (0, 1): x1_mf0 * x2_mf1
        0.2 * 0.6,  # Rule (1, 0): x1_mf1 * x2_mf0
        0.2 * 0.4,  # Rule (1, 1): x1_mf1 * x2_mf1
    ]

    np.testing.assert_allclose(rule_strengths[0], expected_strengths, rtol=1e-5)


def test_rule_layer_with_custom_rules_subset():
    """Ensure RuleLayer restricts to explicitly provided rules."""
    layer = RuleLayer(input_names=["x1", "x2"], mf_per_input=[2, 2], rules=[(0, 1), (1, 0)])

    membership_outputs = {
        "x1": np.array([[0.8, 0.2]]),
        "x2": np.array([[0.6, 0.4]]),
    }

    strengths = layer.forward(membership_outputs)

    assert layer.n_rules == 2
    assert layer.rules == [(0, 1), (1, 0)]
    np.testing.assert_allclose(strengths, np.array([[0.8 * 0.4, 0.2 * 0.6]]), rtol=1e-6)


def test_rule_layer_rejects_rule_length_mismatch():
    with pytest.raises(ValueError, match="Each rule must specify exactly one membership index per input"):
        RuleLayer(input_names=["x1", "x2"], mf_per_input=[2, 2], rules=[(0, 1, 2)])


def test_rule_layer_rejects_index_out_of_range():
    with pytest.raises(ValueError, match="Rule membership index out of range"):
        RuleLayer(input_names=["x1", "x2"], mf_per_input=[2, 2], rules=[(0, 2)])


def test_rule_layer_requires_at_least_one_rule():
    with pytest.raises(ValueError, match="At least one rule must be provided"):
        RuleLayer(input_names=["x1", "x2"], mf_per_input=[2, 2], rules=[])


def test_rule_layer_backward(simple_rule_layer):
    """Test RuleLayer backward pass with new interface."""
    # Create sample membership outputs
    membership_outputs = {"x1": np.array([[0.8, 0.2]]), "x2": np.array([[0.6, 0.4]])}

    rule_strengths = simple_rule_layer.forward(membership_outputs)
    dL_dw = np.ones_like(rule_strengths)  # dL/dw = 1 for all rules

    gradients = simple_rule_layer.backward(dL_dw)

    # Check output structure
    assert isinstance(gradients, dict)
    assert set(gradients.keys()) == {"x1", "x2"}

    # Check shapes
    assert gradients["x1"].shape == (1, 2)  # (batch_size, n_mfs)
    assert gradients["x2"].shape == (1, 2)

    # Verify gradients are reasonable (non-zero for this case)
    assert np.all(gradients["x1"] > 0)
    assert np.all(gradients["x2"] > 0)


def test_integrated_membership_and_rule_layers(sample_input_mfs):
    """Test integration between MembershipLayer and RuleLayer."""
    membership_layer = MembershipLayer(sample_input_mfs)
    rule_layer = RuleLayer(input_names=["x1", "x2"], mf_per_input=[2, 2])

    x = np.array([[0.0, 0.0], [1.0, 1.0]])  # shape (2, 2)

    # Forward pass through both layers
    membership_outputs = membership_layer.forward(x)
    rule_strengths = rule_layer.forward(membership_outputs)

    # Backward pass through both layers
    dL_dw = np.ones_like(rule_strengths)
    gradients = rule_layer.backward(dL_dw)
    membership_layer.backward(gradients)

    # Verify final gradients reached membership functions
    for name in ["x1", "x2"]:
        for mf in sample_input_mfs[name]:
            assert "mean" in mf.gradients
            assert "sigma" in mf.gradients


def test_normalization_forward():
    """Test NormalizationLayer forward pass."""
    layer = NormalizationLayer()
    w = np.array([[1.0, 2.0, 3.0]])
    norm = layer.forward(w)

    expected = w / np.sum(w, axis=1, keepdims=True)
    np.testing.assert_allclose(norm, expected, rtol=1e-6)


def test_normalization_backward():
    """Test NormalizationLayer backward pass with numerical gradient verification."""
    layer = NormalizationLayer()
    w = np.array([[1.0, 2.0, 3.0]])
    layer.forward(w)

    dL_dnorm = np.array([[1.0, 0.0, 0.0]])  # only first output has gradient
    dL_dw = layer.backward(dL_dnorm)

    # Numerical gradient verification using finite differences
    epsilon = 1e-5
    numerical = np.zeros_like(w)

    for i in range(w.shape[1]):
        w_pos = w.copy()
        w_neg = w.copy()
        w_pos[0, i] += epsilon
        w_neg[0, i] -= epsilon

        out_pos = w_pos / np.sum(w_pos, axis=1, keepdims=True)
        out_neg = w_neg / np.sum(w_neg, axis=1, keepdims=True)

        loss_pos = out_pos[0, 0]  # since dL/dnorm = [1, 0, 0]
        loss_neg = out_neg[0, 0]

        numerical[0, i] = (loss_pos - loss_neg) / (2 * epsilon)

    np.testing.assert_allclose(dL_dw, numerical, rtol=1e-4, atol=1e-6)


def test_consequent_forward_shape():
    """Test ConsequentLayer forward pass output shape."""
    layer = ConsequentLayer(n_rules=3, n_inputs=2)
    x = np.array([[1.0, 2.0]])
    norm_w = np.array([[0.2, 0.3, 0.5]])

    y_hat = layer.forward(x, norm_w)

    assert y_hat.shape == (1, 1)


def test_consequent_backward_gradients():
    """Test ConsequentLayer backward pass with numerical gradient verification."""
    np.random.seed(0)
    n_rules = 3
    n_inputs = 2
    batch_size = 1

    layer = ConsequentLayer(n_rules=n_rules, n_inputs=n_inputs)
    x = np.random.randn(batch_size, n_inputs)
    norm_w = np.random.rand(batch_size, n_rules)
    norm_w /= norm_w.sum(axis=1, keepdims=True)  # ensure normalization

    y_hat = layer.forward(x, norm_w)
    dL_dy = np.ones_like(y_hat)  # gradient of loss w.r.t. output

    dL_dnorm_w, dL_dx = layer.backward(dL_dy)

    # Numerical gradient verification for parameters
    epsilon = 1e-5
    numerical_grad = np.zeros_like(layer.parameters)

    for i in range(n_rules):
        for j in range(n_inputs + 1):
            original = layer.parameters[i, j]

            layer.parameters[i, j] = original + epsilon
            y_pos = layer.forward(x, norm_w)

            layer.parameters[i, j] = original - epsilon
            y_neg = layer.forward(x, norm_w)

            numerical_grad[i, j] = (y_pos - y_neg).squeeze() / (2 * epsilon)
            layer.parameters[i, j] = original  # restore

    np.testing.assert_allclose(layer.gradients, numerical_grad, rtol=1e-4, atol=1e-6)


def test_full_anfis_pipeline(sample_input_mfs):
    """Test full ANFIS pipeline with all layers working together."""
    # Initialize all layers
    membership_layer = MembershipLayer(sample_input_mfs)
    rule_layer = RuleLayer(input_names=["x1", "x2"], mf_per_input=[2, 2])
    normalization_layer = NormalizationLayer()
    consequent_layer = ConsequentLayer(n_rules=4, n_inputs=2)

    # Sample input
    x = np.array([[0.5, -0.5]])  # shape (1, 2)

    # Forward pass through all layers
    membership_outputs = membership_layer.forward(x)  # Layer 1: Fuzzification
    rule_strengths = rule_layer.forward(membership_outputs)  # Layer 2: Rule strength
    norm_weights = normalization_layer.forward(rule_strengths)  # Layer 3: Normalization
    final_output = consequent_layer.forward(x, norm_weights)  # Layer 4: Output

    # Verify shapes at each stage
    assert len(membership_outputs) == 2  # x1 and x2
    assert membership_outputs["x1"].shape == (1, 2)
    assert membership_outputs["x2"].shape == (1, 2)
    assert rule_strengths.shape == (1, 4)  # 2x2 = 4 rules
    assert norm_weights.shape == (1, 4)
    assert final_output.shape == (1, 1)

    # Verify normalization constraint
    np.testing.assert_allclose(norm_weights.sum(axis=1), 1.0, rtol=1e-6)

    # Backward pass through all layers (reverse order)
    dL_dy = np.ones_like(final_output)
    dL_dnorm_w, dL_dx = consequent_layer.backward(dL_dy)  # Layer 4
    dL_dw = normalization_layer.backward(dL_dnorm_w)  # Layer 3
    gradients = rule_layer.backward(dL_dw)  # Layer 2
    membership_layer.backward(gradients)  # Layer 1

    # Verify that gradients reached the membership functions
    for name in ["x1", "x2"]:
        for mf in sample_input_mfs[name]:
            assert "mean" in mf.gradients
            assert "sigma" in mf.gradients
            # Gradients should be non-zero (accumulated from forward pass)
            assert mf.gradients["mean"] != 0.0 or mf.gradients["sigma"] != 0.0
