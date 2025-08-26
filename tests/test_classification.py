import numpy as np
import pytest

from anfis_toolbox import ANFISClassifier, QuickANFIS, accuracy
from anfis_toolbox.membership import GaussianMF


def make_simple_input_mfs(n_features=1, n_mfs=2):
    input_mfs = {}
    for i in range(n_features):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)][:n_mfs]
    return input_mfs


def test_classifier_forward_and_predict_shapes():
    mfs = make_simple_input_mfs(n_features=2, n_mfs=2)
    clf = ANFISClassifier(mfs, n_classes=3)
    X = np.array([[0.0, 0.5], [1.0, -1.0]])
    logits = clf.forward(X)
    assert logits.shape == (2, 3)
    proba = clf.predict_proba(X)
    assert proba.shape == (2, 3)
    preds = clf.predict(X)
    assert preds.shape == (2,)


def test_classifier_predict_proba_accepts_1d_input_and_repr_and_property():
    mfs = make_simple_input_mfs(n_features=1, n_mfs=2)
    clf = ANFISClassifier(mfs, n_classes=2)
    x1 = np.array([0.0])
    proba = clf.predict_proba(x1)
    assert proba.shape == (1, 2)
    r = repr(clf)
    assert "ANFISClassifier(" in r and "n_classes=2" in r
    # property
    mf_prop = clf.membership_functions
    assert isinstance(mf_prop, dict) and "x1" in mf_prop


def test_classifier_fit_binary_toy():
    # 1D, two clusters -> linearly separable labels
    rng = np.random.default_rng(0)
    X_left = rng.normal(loc=-1.0, scale=0.2, size=(30, 1))
    X_right = rng.normal(loc=1.0, scale=0.2, size=(30, 1))
    X = np.vstack([X_left, X_right])
    y = np.array([0] * len(X_left) + [1] * len(X_right))

    model = QuickANFIS.for_classification(X, n_classes=2, n_mfs=2, mf_type="gaussian", init="fcm", random_state=0)
    losses = model.fit(X, y, epochs=10, learning_rate=0.1, verbose=False)
    assert len(losses) == 10
    proba = model.predict_proba(X)
    acc = accuracy(y, proba)
    assert acc >= 0.8


def test_classifier_fit_with_one_hot_and_invalid_shapes():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    y_int = (X[:, 0] > 0).astype(int)
    y_oh = np.zeros((X.shape[0], 2), dtype=float)
    y_oh[np.arange(X.shape[0]), y_int] = 1.0
    model = QuickANFIS.for_classification(X, n_classes=2, n_mfs=2)
    losses = model.fit(X, y_oh, epochs=2, learning_rate=0.05, verbose=False)
    assert len(losses) == 2
    # invalid one-hot columns
    with pytest.raises(ValueError):
        model.fit(X, np.zeros((X.shape[0], 3)))


def test_classifier_input_validation():
    mfs = make_simple_input_mfs(n_features=2, n_mfs=2)
    clf = ANFISClassifier(mfs, n_classes=2)
    with pytest.raises(ValueError):
        clf.predict_proba(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        clf.predict_proba(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        clf.predict_proba(np.zeros((2, 2, 2)))
    # invalid n_classes
    with pytest.raises(ValueError):
        ANFISClassifier(make_simple_input_mfs(), n_classes=1)


def test_quickanfis_for_classification_fcm_and_invalid_ndim():
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.3, size=(20, 1)),
            rng.normal(loc=1.0, scale=0.3, size=(20, 1)),
        ]
    )
    model = QuickANFIS.for_classification(X, n_classes=2, n_mfs=2, mf_type="gaussian", init="fcm", random_state=0)
    assert isinstance(model, ANFISClassifier)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    # invalid ndim
    with pytest.raises(ValueError):
        QuickANFIS.for_classification(X.reshape(-1), n_classes=2)


def test_classifier_parameters_and_gradients_management():
    rng = np.random.default_rng(7)
    X = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(10, 1)),
            rng.normal(loc=1.0, scale=0.2, size=(10, 1)),
        ]
    )
    y = np.array([0] * 10 + [1] * 10)
    model = QuickANFIS.for_classification(X, n_classes=2, n_mfs=2)
    # One epoch to populate gradients
    losses = model.fit(X, y, epochs=1, learning_rate=0.01, verbose=False)
    assert len(losses) == 1
    grads = model.get_gradients()
    assert "consequent" in grads and np.any(grads["consequent"] != 0)
    # Params roundtrip
    params = model.get_parameters()
    model.set_parameters(params)
    # Reset gradients clears accumulators
    model.reset_gradients()
    grads2 = model.get_gradients()
    np.testing.assert_array_equal(grads2["consequent"], 0)
