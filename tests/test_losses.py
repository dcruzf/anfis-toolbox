import numpy as np

from anfis_toolbox.losses import (
    CrossEntropyLoss,
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
    try:
        ce.prepare_targets(np.zeros((3, 2)), model=Dummy())
    except ValueError:
        pass
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError for mismatched one-hot columns")

    assert isinstance(resolve_loss("mse"), MSELoss)
    resolved_ce = resolve_loss("cross_entropy")
    assert isinstance(resolved_ce, CrossEntropyLoss)
    existing = CrossEntropyLoss()
    assert resolve_loss(existing) is existing
    try:
        resolve_loss("unknown")
    except ValueError:
        pass
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError for unknown loss name")
