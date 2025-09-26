import numpy as np
import pytest

from anfis_toolbox.losses import (
    CrossEntropyLoss,
    LossFunction,
    MSELoss,
    cross_entropy_grad,
    cross_entropy_loss,
    mse_grad,
    mse_loss,
    resolve_loss,
)
from anfis_toolbox.metrics import cross_entropy as ce_metric
from anfis_toolbox.metrics import softmax


def test_mse_loss_matches_metric_and_grad_formula():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(10, 3))
    y_pred = rng.normal(size=(10, 3))
    # Value matches mean over squared diffs
    expected = float(np.mean((y_true - y_pred) ** 2))
    assert np.isclose(mse_loss(y_true, y_pred), expected)
    # Gradient matches 2*(pred-true)/n
    grad = mse_grad(y_true, y_pred)
    assert grad.shape == y_pred.shape
    assert np.allclose(grad, 2.0 * (y_pred - y_true) / y_true.shape[0])


def test_cross_entropy_loss_and_grad_with_int_labels():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    y_int = np.array([0, 1, 1])
    # Loss matches metrics.cross_entropy
    val = cross_entropy_loss(y_int, logits)
    assert np.isclose(val, ce_metric(y_int, logits))
    # Grad shape and basic sanity: sum over classes of grad equals 0 per sample
    grad = cross_entropy_grad(y_int, logits)
    assert grad.shape == logits.shape
    assert np.allclose(np.sum(grad, axis=1), 0.0)


def test_cross_entropy_loss_and_grad_with_one_hot():
    logits = np.array([[0.5, -0.5, 0.0], [0.0, 0.0, 0.0]])
    y_oh = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Loss equals metrics; grad equals softmax - y over n
    val = cross_entropy_loss(y_oh, logits)
    assert np.isclose(val, ce_metric(y_oh, logits))
    probs = softmax(logits, axis=1)
    grad = cross_entropy_grad(y_oh, logits)
    assert np.allclose(grad, (probs - y_oh) / logits.shape[0])


def test_loss_classes_prepare_targets_and_resolve():
    mse = MSELoss()
    y = np.array([1.0, 2.0])
    prepared = mse.prepare_targets(y)
    assert prepared.shape == (2, 1)
    preds = np.array([[1.5], [2.5]])
    assert np.isclose(mse.loss(prepared, preds), mse_loss(prepared, preds))
    assert np.allclose(mse.gradient(prepared, preds), mse_grad(prepared, preds))

    class Dummy:
        n_classes = 3

    ce = CrossEntropyLoss()
    y_int = np.array([0, 2, 1])
    prepared_int = ce.prepare_targets(y_int, model=Dummy())
    assert prepared_int.shape == (3, 3)
    y_oh = np.eye(3)
    prepared_oh = ce.prepare_targets(y_oh, model=Dummy())
    np.testing.assert_array_equal(prepared_oh, y_oh)
    with pytest.raises(ValueError, match="columns"):
        ce.prepare_targets(np.zeros((3, 2)), model=Dummy())

    assert isinstance(resolve_loss("mse"), MSELoss)
    resolved_ce = resolve_loss("cross_entropy")
    assert isinstance(resolved_ce, CrossEntropyLoss)
    existing = CrossEntropyLoss()
    assert resolve_loss(existing) is existing
    with pytest.raises(ValueError, match="Unknown loss"):
        resolve_loss("unknown")


def test_lossfunction_prepare_targets_casts_to_float():
    base = LossFunction()
    result = base.prepare_targets([1, 2, 3])
    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))


def test_cross_entropy_grad_raises_on_shape_mismatch():
    logits = np.zeros((2, 3))
    bad_one_hot = np.zeros((2, 2))
    with pytest.raises(ValueError, match="same shape"):
        cross_entropy_grad(bad_one_hot, logits)


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
