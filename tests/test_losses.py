import numpy as np
import pytest

from anfis_toolbox.losses import (
    CrossEntropyLoss,
    LossFunction,
    resolve_loss,
)


def test_lossfunction_prepare_targets_casts_to_float():
    base = LossFunction()
    result = base.prepare_targets([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))


def test_cross_entropy_prepare_targets_infers_class_count_without_model():
    ce = CrossEntropyLoss()
    labels = np.array([0, 2])
    encoded = ce.prepare_targets(labels)
    expected = np.eye(3)[[0, 2]]
    np.testing.assert_array_equal(encoded, expected)


def test_cross_entropy_prepare_targets_invalid_dim_raises():
    ce = CrossEntropyLoss()
    with pytest.raises(ValueError, match="1D labels or 2D one-hot"):
        ce.prepare_targets(np.zeros((2, 2, 1)))


def test_cross_entropy_prepare_targets_enforces_model_class_count():
    class Dummy:
        n_classes = 4

    ce = CrossEntropyLoss()
    with pytest.raises(ValueError, match="4 columns"):
        ce.prepare_targets(np.zeros((3, 3)), model=Dummy())


def test_cross_entropy_prepare_targets_accepts_one_hot_without_model():
    ce = CrossEntropyLoss()
    y_oh = np.eye(2)
    result = ce.prepare_targets(y_oh)
    np.testing.assert_array_equal(result, y_oh.astype(float))


def test_resolve_loss_rejects_invalid_type():
    with pytest.raises(TypeError, match="must be None, str, or a LossFunction"):
        resolve_loss(3.14)


def test_cross_entropy_loss_empty_batch_returns_zero():
    ce = CrossEntropyLoss()
    y_true = np.array([], dtype=int)
    y_pred = np.zeros((0, 3))
    assert ce.loss(y_true, y_pred) == 0.0


def test_cross_entropy_loss_integer_labels_computes_expected_value():
    ce = CrossEntropyLoss()
    y_true = np.array([0, 1])
    logits = np.zeros((2, 2))
    expected = float(np.log(2.0))  # -log(0.5) averaged over two samples
    assert np.isclose(ce.loss(y_true, logits), expected)


def test_cross_entropy_loss_label_length_mismatch_raises():
    ce = CrossEntropyLoss()
    y_true = np.array([0])
    logits = np.zeros((2, 2))
    with pytest.raises(ValueError, match="match logits batch size"):
        ce.loss(y_true, logits)


def test_cross_entropy_loss_one_hot_shape_mismatch_raises():
    ce = CrossEntropyLoss()
    y_true = np.zeros((2, 2))
    logits = np.zeros((2, 3))
    with pytest.raises(ValueError, match="shape must match logits"):
        ce.loss(y_true, logits)


def test_cross_entropy_gradient_builds_one_hot_from_labels():
    ce = CrossEntropyLoss()
    y_true = np.array([1, 0])
    logits = np.array([[0.0, 0.0], [2.0, -2.0]])
    # Manual softmax
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    expected = (probs - np.array([[0.0, 1.0], [1.0, 0.0]])) / 2.0
    np.testing.assert_allclose(ce.gradient(y_true, logits), expected)


def test_cross_entropy_gradient_one_hot_shape_mismatch_raises():
    ce = CrossEntropyLoss()
    y_true = np.zeros((2, 2))
    logits = np.zeros((2, 3))
    with pytest.raises(ValueError, match="one-hot must have same shape"):
        ce.gradient(y_true, logits)


def test_resolve_loss_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown loss 'bogus'"):
        resolve_loss("bogus")
