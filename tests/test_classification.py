import numpy as np
import pytest

from anfis_toolbox import ANFISClassifier, QuickANFIS, TSKANFISClassifier, accuracy
from anfis_toolbox.estimator_utils import NotFittedError
from anfis_toolbox.losses import LossFunction
from anfis_toolbox.losses import resolve_loss as _resolve_loss
from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.optim import SGDTrainer

LowLevelClassifier = TSKANFISClassifier


def make_simple_input_mfs(n_features=1, n_mfs=2):
    input_mfs = {}
    for i in range(n_features):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)][:n_mfs]
    return input_mfs


def _generate_classification_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(80, 2))
    y = (1.2 * X[:, 0] - 0.8 * X[:, 1] > 0.0).astype(int)
    return X, y


def test_classifier_forward_and_predict_shapes():
    mfs = make_simple_input_mfs(n_features=2, n_mfs=2)
    clf = LowLevelClassifier(mfs, n_classes=3)
    X = np.array([[0.0, 0.5], [1.0, -1.0]])
    logits = clf.forward(X)
    assert logits.shape == (2, 3)
    proba = clf.predict_proba(X)
    assert proba.shape == (2, 3)
    preds = clf.predict(X)
    assert preds.shape == (2,)


def test_classifier_predict_proba_accepts_1d_input_and_repr_and_property():
    mfs = make_simple_input_mfs(n_features=1, n_mfs=2)
    clf = LowLevelClassifier(mfs, n_classes=2)
    x1 = np.array([0.0])
    proba = clf.predict_proba(x1)
    assert proba.shape == (1, 2)
    r = repr(clf)
    assert "TSKANFISClassifier(" in r and "n_classes=2" in r
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
    clf = LowLevelClassifier(mfs, n_classes=2)
    with pytest.raises(ValueError):
        clf.predict_proba(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        clf.predict_proba(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        clf.predict_proba(np.zeros((2, 2, 2)))
    # invalid n_classes
    with pytest.raises(ValueError):
        LowLevelClassifier(make_simple_input_mfs(), n_classes=1)


def test_quickanfis_for_classification_fcm_and_invalid_ndim():
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.3, size=(20, 1)),
            rng.normal(loc=1.0, scale=0.3, size=(20, 1)),
        ]
    )
    model = QuickANFIS.for_classification(X, n_classes=2, n_mfs=2, mf_type="gaussian", init="fcm", random_state=0)
    assert isinstance(model, LowLevelClassifier)
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


def test_classifier_set_parameters_without_consequent_updates_only_membership():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 2))
    model = QuickANFIS.for_classification(X, n_classes=2, n_mfs=2)
    params_before = model.get_parameters()
    # Build membership-only params dict
    memb_only = {"membership": params_before["membership"]}
    # tweak a membership param
    first_name = next(iter(memb_only["membership"]))
    memb_only["membership"][first_name][0]["mean"] += 0.123
    model.set_parameters(memb_only)
    params_after = model.get_parameters()
    # Consequent unchanged
    np.testing.assert_array_equal(params_after["consequent"], params_before["consequent"])
    # Membership changed for the modified entry
    assert np.isclose(params_after["membership"][first_name][0]["mean"], memb_only["membership"][first_name][0]["mean"])


def test_classifier_set_parameters_without_membership_updates_only_consequent():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(8, 1))
    model = QuickANFIS.for_classification(X, n_classes=3, n_mfs=2)
    params_before = model.get_parameters()
    new_consequent = np.ones_like(params_before["consequent"]) * 2.5
    model.set_parameters({"consequent": new_consequent})
    params_after = model.get_parameters()
    np.testing.assert_array_equal(params_after["consequent"], new_consequent)
    assert params_after["membership"] == params_before["membership"]


def test_classifier_set_parameters_membership_missing_name_skips_safely():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(12, 2))
    model = QuickANFIS.for_classification(X, n_classes=2, n_mfs=2)
    params_before = model.get_parameters()
    # Keep only one input's membership params to trigger the continue path
    one_name = list(params_before["membership"].keys())[0]
    partial_membership = {one_name: params_before["membership"][one_name]}
    model.set_parameters({"membership": partial_membership})
    params_after = model.get_parameters()
    # Updated for provided name
    assert params_after["membership"][one_name] == partial_membership[one_name]
    # Unprovided input remains unchanged
    remaining = [n for n in params_before["membership"].keys() if n != one_name]
    for n in remaining:
        assert params_after["membership"][n] == params_before["membership"][n]


def _make_simple_clf(n_inputs: int = 1, n_mfs: int = 2, n_classes: int = 2) -> LowLevelClassifier:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)][:n_mfs]
    return LowLevelClassifier(input_mfs, n_classes=n_classes, random_state=0)


def test_classifier_apply_membership_gradients_private_helper():
    clf = _make_simple_clf(n_inputs=1, n_mfs=2, n_classes=2)
    X = np.array([[-0.5], [0.7]])
    # Create gradients by simulating a backward step with dummy dL/dlogits
    logits = clf.forward(X)
    dL_dlogits = np.ones_like(logits) / logits.shape[0]
    clf.backward(dL_dlogits)
    params_before = clf.get_parameters()
    clf._apply_membership_gradients(learning_rate=0.01)
    params_after = clf.get_parameters()
    # Expect some membership parameter to change
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


def test_classifier_fit_uses_custom_loss_and_default_trainer(monkeypatch):
    mfs = make_simple_input_mfs(n_features=1, n_mfs=2)
    clf = LowLevelClassifier(mfs, n_classes=2)
    X = np.array([[-0.5], [0.2], [0.8]])
    y = np.array([0, 1, 1])

    resolve_calls: list[str] = []

    def fake_resolve(spec):
        resolve_calls.append(spec)
        return _resolve_loss(spec)

    monkeypatch.setattr("anfis_toolbox.model.resolve_loss", fake_resolve)

    class CaptureTrainer:
        created: list["CaptureTrainer"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loss = kwargs.get("loss")
            CaptureTrainer.created.append(self)

        def fit(self, model, X_fit, y_fit):
            self.model = model
            self.X_shape = X_fit.shape
            self.y_shape = y_fit.shape
            return [0.0]

    monkeypatch.setattr("anfis_toolbox.optim.AdamTrainer", CaptureTrainer)

    losses = clf.fit(X, y, epochs=1, learning_rate=0.01, verbose=False, loss="mse")

    assert losses == [0.0]
    assert resolve_calls == ["mse"]
    assert CaptureTrainer.created, "Expected AdamTrainer to be instantiated"
    trainer_instance = CaptureTrainer.created[-1]
    assert isinstance(trainer_instance.loss, LossFunction)
    assert trainer_instance.X_shape == X.shape
    assert trainer_instance.y_shape == y.shape


def test_classifier_fit_updates_existing_trainer_loss():
    class DummyTrainer:
        def __init__(self):
            self.loss = None
            self.fit_called = False

        def fit(self, model, X, y):
            self.fit_called = True
            self.model = model
            self.X_shape = X.shape
            self.y_shape = y.shape
            return [1.0]

    dummy = DummyTrainer()
    mfs = make_simple_input_mfs(n_features=1, n_mfs=2)
    clf = LowLevelClassifier(mfs, n_classes=2)
    X = np.array([[0.1], [-0.3]])
    y = np.array([0, 1])

    losses = clf.fit(X, y, trainer=dummy, loss="cross_entropy")

    assert losses == [1.0]
    assert dummy.fit_called
    assert isinstance(dummy.loss, LossFunction)
    assert dummy.X_shape == X.shape
    assert dummy.y_shape == y.shape


def test_classifier_fit_with_trainer_without_loss_attribute():
    class NoLossTrainer:
        def __init__(self):
            self.fit_called = False

        def fit(self, model, X, y):
            self.fit_called = True
            self.model = model
            self.X_shape = X.shape
            self.y_shape = y.shape
            return [2.0]

    trainer = NoLossTrainer()
    mfs = make_simple_input_mfs(n_features=1, n_mfs=2)
    clf = LowLevelClassifier(mfs, n_classes=2)
    X = np.array([[-0.2], [0.4]])
    y = np.array([0, 1])

    losses = clf.fit(X, y, trainer=trainer)

    assert losses == [2.0]
    assert trainer.fit_called
    assert not hasattr(trainer, "loss")
    assert trainer.X_shape == X.shape
    assert trainer.y_shape == y.shape


def test_anfis_classifier_fit_predict_flow():
    X, y = _generate_classification_data(seed=1)
    clf = ANFISClassifier(
        n_classes=2,
        n_mfs=2,
        optimizer="sgd",
        epochs=3,
        learning_rate=0.05,
        verbose=False,
        random_state=0,
    )

    result = clf.fit(X, y)
    assert result is clf
    assert clf.model_ is not None
    assert isinstance(clf.model_, LowLevelClassifier)
    assert clf.optimizer_ is not None
    assert isinstance(clf.optimizer_, SGDTrainer)
    assert clf.training_history_ is not None
    assert len(clf.training_history_) == clf.optimizer_.epochs
    assert clf.feature_names_in_ == ["x1", "x2"]
    assert clf.n_features_in_ == 2
    assert clf.classes_.shape == (2,)

    proba = clf.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    preds = clf.predict(X[:5])
    assert preds.shape == (5,)
    assert set(np.unique(preds)).issubset(set(clf.classes_))

    metrics = clf.evaluate(X[:5], y[:5])
    assert set(metrics.keys()) == {"accuracy", "log_loss"}
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_anfis_classifier_inputs_config_membership_overrides():
    X, y = _generate_classification_data(seed=2)
    custom_mfs = [GaussianMF(mean=-0.5, sigma=0.4), GaussianMF(mean=0.5, sigma=0.4)]
    inputs_config = {
        "x1": {"mfs": custom_mfs},
        1: {"mf_type": "bell", "n_mfs": 2},
    }

    clf = ANFISClassifier(
        n_classes=2,
        n_mfs=2,
        inputs_config=inputs_config,
        optimizer="sgd",
        epochs=2,
        verbose=False,
        random_state=1,
    )

    clf.fit(X, y)
    assert clf.input_specs_ is not None
    spec = clf.input_specs_[0]
    assert spec["membership_functions"][0] is custom_mfs[0]
    assert clf.model_ is not None
    assert clf.model_.membership_functions["x1"][0] is custom_mfs[0]
    assert type(clf.model_.membership_functions["x2"][0]).__name__ == "BellMF"


def test_anfis_classifier_custom_trainer_instance_overrides():
    X, y = _generate_classification_data(seed=3)
    base_trainer = SGDTrainer(epochs=1, learning_rate=0.01, verbose=True)
    clf = ANFISClassifier(
        n_classes=2,
        n_mfs=2,
        optimizer=base_trainer,
        epochs=4,
        learning_rate=0.2,
        verbose=False,
    )

    clf.fit(X, y)
    assert clf.optimizer_ is not base_trainer
    assert isinstance(clf.optimizer_, SGDTrainer)
    assert clf.optimizer_.epochs == 4
    assert pytest.approx(clf.optimizer_.learning_rate, rel=1e-6) == 0.2
    assert clf.optimizer_.verbose is False
    assert clf.training_history_ is not None
    assert len(clf.training_history_) == clf.optimizer_.epochs


def test_anfis_classifier_invalid_optimizer_string():
    X, y = _generate_classification_data(seed=4)
    clf = ANFISClassifier(n_classes=2, optimizer="invalid", epochs=1)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        clf.fit(X, y)


def test_anfis_classifier_predict_requires_fit():
    clf = ANFISClassifier(n_classes=2)
    with pytest.raises(NotFittedError):
        clf.predict(np.zeros((1, 2)))
