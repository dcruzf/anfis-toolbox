import inspect
import logging
import pickle
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from anfis_toolbox import ANFISClassifier, ANFISRegressor
from anfis_toolbox.builders import ANFISBuilder
from anfis_toolbox.classifier import _ensure_training_logging
from anfis_toolbox.estimator_utils import NotFittedError
from anfis_toolbox.losses import LossFunction
from anfis_toolbox.losses import resolve_loss as _resolve_loss
from anfis_toolbox.membership import GaussianMF, MembershipFunction
from anfis_toolbox.metrics import ANFISMetrics, accuracy
from anfis_toolbox.model import TSKANFISClassifier
from anfis_toolbox.optim import BaseTrainer, HybridAdamTrainer, HybridTrainer, SGDTrainer


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


def _build_classifier_from_data(
    X: np.ndarray,
    *,
    n_classes: int,
    n_mfs: int = 3,
    mf_type: str = "gaussian",
    init: str = "grid",
    random_state: int | None = None,
) -> TSKANFISClassifier:
    if X.ndim != 2:
        raise ValueError("Input data must be 2D (n_samples, n_features)")

    builder = ANFISBuilder()
    init_key = None if init is None else str(init).strip().lower()
    for i in range(X.shape[1]):
        col_data = X[:, i]
        if init_key == "fcm":
            builder.add_input_from_data(
                f"x{i + 1}",
                col_data,
                n_mfs=n_mfs,
                mf_type=mf_type,
                init="fcm",
                random_state=random_state,
            )
        elif init_key == "random":
            builder.add_input_from_data(
                f"x{i + 1}",
                col_data,
                n_mfs=n_mfs,
                mf_type=mf_type,
                init="random",
                random_state=random_state,
            )
        else:
            range_min = float(np.min(col_data))
            range_max = float(np.max(col_data))
            margin = (range_max - range_min) * 0.1
            builder.add_input(
                f"x{i + 1}",
                range_min - margin,
                range_max + margin,
                n_mfs,
                mf_type,
            )

    return TSKANFISClassifier(builder.input_mfs, n_classes=n_classes, random_state=random_state)


def test_classifier_forward_and_predict_shapes():
    mfs = make_simple_input_mfs(n_features=2, n_mfs=2)
    clf = TSKANFISClassifier(mfs, n_classes=3)
    X = np.array([[0.0, 0.5], [1.0, -1.0]])
    logits = clf.forward(X)
    assert logits.shape == (2, 3)
    proba = clf.predict_proba(X)
    assert proba.shape == (2, 3)
    preds = clf.predict(X)
    assert preds.shape == (2,)


def test_ensure_training_logging_behaviour(monkeypatch):
    calls: list[str] = []

    def fake_enable():
        calls.append("enabled")

    dummy_logger = SimpleNamespace(handlers=[])
    monkeypatch.setattr("anfis_toolbox.classifier.enable_training_logs", fake_enable)

    real_get_logger = logging.getLogger

    def fake_get_logger(name: str | None = None):
        if name == "anfis_toolbox":
            return dummy_logger
        return real_get_logger(name)

    monkeypatch.setattr("anfis_toolbox.classifier.logging.getLogger", fake_get_logger)

    _ensure_training_logging(False)
    assert not calls

    _ensure_training_logging(True)
    assert calls == ["enabled"]

    dummy_logger.handlers = [object()]
    _ensure_training_logging(True)
    assert calls == ["enabled"]


def test_classifier_predict_proba_accepts_1d_input_and_repr_and_property():
    mfs = make_simple_input_mfs(n_features=1, n_mfs=2)
    clf = TSKANFISClassifier(mfs, n_classes=2)
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

    model = _build_classifier_from_data(X, n_classes=2, n_mfs=2, mf_type="gaussian", init="fcm", random_state=0)
    history = model.fit(X, y, epochs=10, learning_rate=0.1, verbose=False)
    train_losses = history["train"]
    assert len(train_losses) == 10
    proba = model.predict_proba(X)
    acc = accuracy(y, proba)
    assert acc >= 0.8


def test_classifier_fit_raises_when_trainer_history_invalid():
    class DummyTrainer(BaseTrainer):
        def fit(self, model, X_fit, y_fit, **kwargs):  # noqa: D401 - simple stub
            return []  # Not a dict, should trigger TypeError

        def init_state(self, model, X_fit, y_fit):  # pragma: no cover - not used
            return None

        def train_step(self, model, X_batch, y_batch, state):  # pragma: no cover - not used
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):  # pragma: no cover - not used
            return 0.0

    clf = ANFISClassifier(n_classes=2, n_mfs=2, optimizer="sgd", random_state=0, verbose=False)

    dummy_model = SimpleNamespace(
        rules=[],
        predict=lambda X: np.zeros(X.shape[0]),
        predict_proba=lambda X: np.full((X.shape[0], clf.n_classes), 1.0 / clf.n_classes),
    )

    clf._build_model = lambda X, feature_names: dummy_model  # type: ignore[assignment]
    clf._instantiate_trainer = lambda: DummyTrainer()  # type: ignore[assignment]

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])

    with pytest.raises(TypeError, match="TrainingHistory"):
        clf.fit(X, y)


def test_classifier_fit_raises_when_n_classes_not_inferred(monkeypatch):
    clf = ANFISClassifier(n_mfs=2, optimizer="sgd", random_state=0, verbose=False)

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    def fake_encode(self, targets, n_samples, *, allow_partial_classes=False):
        return np.zeros(n_samples, dtype=int), np.array([0, 1])

    monkeypatch.setattr(clf, "_encode_targets", MethodType(fake_encode, clf))

    with pytest.raises(RuntimeError, match="n_classes could not be inferred"):
        clf.fit(X, y)


def test_classifier_fit_accepts_verbose_override(monkeypatch):
    calls: list[bool] = []

    def fake_logging(flag: bool):
        calls.append(flag)

    monkeypatch.setattr("anfis_toolbox.classifier._ensure_training_logging", fake_logging)

    class DummyTrainer(BaseTrainer):
        def fit(self, model, X_fit, y_fit, **kwargs):
            assert clf.verbose is True
            return {"train": [0.0]}

        def init_state(self, model, X_fit, y_fit):  # pragma: no cover - unused
            return None

        def train_step(self, model, X_batch, y_batch, state):  # pragma: no cover - unused
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):  # pragma: no cover - unused
            return 0.0

    clf = ANFISClassifier(n_classes=2, n_mfs=2, optimizer="sgd", random_state=0, verbose=False)

    dummy_model = SimpleNamespace(
        rules=[],
        predict=lambda X_pred: np.zeros(X_pred.shape[0], dtype=int),
        predict_proba=lambda X_pred: np.full((X_pred.shape[0], clf.n_classes or 1), 1.0),
    )

    monkeypatch.setattr(clf, "_build_model", lambda X, feature_names: dummy_model)
    monkeypatch.setattr(clf, "_instantiate_trainer", lambda: DummyTrainer())

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    clf.fit(X, y, verbose=True)

    assert calls == [True]
    assert clf.verbose is True


def test_classifier_infers_n_classes_from_targets(monkeypatch):
    class DummyTrainer(BaseTrainer):
        def fit(self, model, X_fit, y_fit, **kwargs):
            self.X_shape = X_fit.shape
            self.y_shape = y_fit.shape
            return {"train": [0.0]}

        def init_state(self, model, X_fit, y_fit):  # pragma: no cover - unused stub
            return None

        def train_step(self, model, X_batch, y_batch, state):  # pragma: no cover - unused stub
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):  # pragma: no cover - unused stub
            return 0.0

    clf = ANFISClassifier(n_mfs=2, optimizer="sgd", random_state=0, verbose=False)

    def fake_build_model(X, feature_names):
        assert clf.n_classes == 2
        return SimpleNamespace(
            rules=[],
            predict=lambda X_pred: np.zeros(X_pred.shape[0], dtype=int),
            predict_proba=lambda X_pred: np.full((X_pred.shape[0], clf.n_classes or 1), 1.0),
        )

    monkeypatch.setattr(clf, "_build_model", fake_build_model)
    monkeypatch.setattr(clf, "_instantiate_trainer", lambda: DummyTrainer())

    X = np.array([[0.0], [1.0], [0.5], [1.5]])
    y = np.array([0, 1, 0, 1])

    clf.fit(X, y)

    assert clf.n_classes == 2
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1], dtype=object))


def test_classifier_infers_n_classes_from_one_hot(monkeypatch):
    class DummyTrainer(BaseTrainer):
        def fit(self, model, X_fit, y_fit, **kwargs):
            return {"train": [0.0]}

        def init_state(self, model, X_fit, y_fit):  # pragma: no cover - unused stub
            return None

        def train_step(self, model, X_batch, y_batch, state):  # pragma: no cover - unused stub
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):  # pragma: no cover - unused stub
            return 0.0

    clf = ANFISClassifier(n_mfs=2, optimizer="sgd", random_state=0, verbose=False)

    def fake_build_model(X, feature_names):
        assert clf.n_classes == 3
        return SimpleNamespace(
            rules=[],
            predict=lambda X_pred: np.zeros(X_pred.shape[0], dtype=int),
            predict_proba=lambda X_pred: np.full((X_pred.shape[0], clf.n_classes or 1), 1.0 / (clf.n_classes or 1)),
        )

    monkeypatch.setattr(clf, "_build_model", fake_build_model)
    monkeypatch.setattr(clf, "_instantiate_trainer", lambda: DummyTrainer())

    X = np.array([[0.0], [1.0], [0.5]])
    y = np.zeros((X.shape[0], 3), dtype=float)
    y[np.arange(X.shape[0]), [0, 1, 2]] = 1.0

    clf.fit(X, y)

    assert clf.n_classes == 3
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1, 2]))


def test_classifier_inferred_classes_require_multiple_labels():
    clf = ANFISClassifier(n_mfs=2, optimizer="sgd", random_state=0, verbose=False)
    X = np.array([[0.0], [1.0], [0.5]])
    y = np.zeros(X.shape[0], dtype=int)

    with pytest.raises(ValueError, match="at least two distinct classes"):
        clf.fit(X, y)


def test_get_rules_returns_empty_tuple_when_rules_missing():
    clf = ANFISClassifier(n_classes=2, verbose=False)
    clf._mark_fitted()
    clf.rules_ = []
    assert clf.get_rules() == ()


def test_classifier_fit_with_one_hot_and_invalid_shapes():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    y_int = (X[:, 0] > 0).astype(int)
    y_oh = np.zeros((X.shape[0], 2), dtype=float)
    y_oh[np.arange(X.shape[0]), y_int] = 1.0
    model = _build_classifier_from_data(X, n_classes=2, n_mfs=2)
    history = model.fit(X, y_oh, epochs=2, learning_rate=0.05, verbose=False)
    assert len(history["train"]) == 2
    # invalid one-hot columns
    with pytest.raises(ValueError):
        model.fit(X, np.zeros((X.shape[0], 3)))


def test_classifier_input_validation():
    mfs = make_simple_input_mfs(n_features=2, n_mfs=2)
    clf = TSKANFISClassifier(mfs, n_classes=2)
    with pytest.raises(ValueError):
        clf.predict_proba(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        clf.predict_proba(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        clf.predict_proba(np.zeros((2, 2, 2)))
    # invalid n_classes
    with pytest.raises(ValueError):
        TSKANFISClassifier(make_simple_input_mfs(), n_classes=1)


def test_classifier_builder_helper_fcm_and_invalid_ndim():
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.3, size=(20, 1)),
            rng.normal(loc=1.0, scale=0.3, size=(20, 1)),
        ]
    )
    model = _build_classifier_from_data(X, n_classes=2, n_mfs=2, mf_type="gaussian", init="fcm", random_state=0)
    assert isinstance(model, TSKANFISClassifier)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    # invalid ndim
    with pytest.raises(ValueError):
        _build_classifier_from_data(X.reshape(-1), n_classes=2)


def test_classifier_parameters_and_gradients_management():
    rng = np.random.default_rng(7)
    X = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(10, 1)),
            rng.normal(loc=1.0, scale=0.2, size=(10, 1)),
        ]
    )
    y = np.array([0] * 10 + [1] * 10)
    model = _build_classifier_from_data(X, n_classes=2, n_mfs=2)
    # One epoch to populate gradients
    history = model.fit(X, y, epochs=1, learning_rate=0.01, verbose=False)
    assert len(history["train"]) == 1
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
    model = _build_classifier_from_data(X, n_classes=2, n_mfs=2)
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
    model = _build_classifier_from_data(X, n_classes=3, n_mfs=2)
    params_before = model.get_parameters()
    new_consequent = np.ones_like(params_before["consequent"]) * 2.5
    model.set_parameters({"consequent": new_consequent})
    params_after = model.get_parameters()
    np.testing.assert_array_equal(params_after["consequent"], new_consequent)
    assert params_after["membership"] == params_before["membership"]


def test_classifier_set_parameters_membership_missing_name_skips_safely():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(12, 2))
    model = _build_classifier_from_data(X, n_classes=2, n_mfs=2)
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


def _make_simple_clf(n_inputs: int = 1, n_mfs: int = 2, n_classes: int = 2) -> TSKANFISClassifier:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)][:n_mfs]
    return TSKANFISClassifier(input_mfs, n_classes=n_classes, random_state=0)


def test_classifier_apply_membership_gradients_private_helper():
    clf = _make_simple_clf(n_inputs=1, n_mfs=2, n_classes=2)
    X = np.array([[-0.5], [0.7]])
    # Create gradients by simulating a backward step with dummy dL/dlogits
    logits = clf.forward(X)
    dL_dlogits = np.ones_like(logits) / logits.shape[0]
    clf.backward(dL_dlogits)
    params_before = clf.get_parameters()
    clf._apply_membership_gradients(0.01)
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
    clf = TSKANFISClassifier(mfs, n_classes=2)
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
            return {"train": [0.0]}

    monkeypatch.setattr("anfis_toolbox.optim.AdamTrainer", CaptureTrainer)

    history = clf.fit(X, y, epochs=1, learning_rate=0.01, verbose=False, loss="mse")

    assert history == {"train": [0.0]}
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
            return {"train": [1.0]}

    dummy = DummyTrainer()
    mfs = make_simple_input_mfs(n_features=1, n_mfs=2)
    clf = TSKANFISClassifier(mfs, n_classes=2)
    X = np.array([[0.1], [-0.3]])
    y = np.array([0, 1])

    history = clf.fit(X, y, trainer=dummy, loss="cross_entropy")

    assert history == {"train": [1.0]}
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
            return {"train": [2.0]}

    trainer = NoLossTrainer()
    mfs = make_simple_input_mfs(n_features=1, n_mfs=2)
    clf = TSKANFISClassifier(mfs, n_classes=2)
    X = np.array([[-0.2], [0.4]])
    y = np.array([0, 1])

    history = clf.fit(X, y, trainer=trainer)

    assert history == {"train": [2.0]}
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
    assert isinstance(clf.model_, TSKANFISClassifier)
    assert clf.optimizer_ is not None
    assert isinstance(clf.optimizer_, SGDTrainer)
    assert clf.training_history_ is not None
    assert isinstance(clf.training_history_, dict)
    assert len(clf.training_history_["train"]) == clf.optimizer_.epochs
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
    assert "accuracy" in metrics
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


def test_anfis_classifier_propagates_explicit_rules():
    captured: dict[str, list[tuple[int, ...]]] = {}

    class DummyTrainer(BaseTrainer):
        def fit(self, model, X_fit, y_fit):
            captured["rules"] = model.rules
            return {"train": []}

        def init_state(self, model, X_fit, y_fit):
            return None

        def train_step(self, model, X_batch, y_batch, state):
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):
            return 0.0

    X, y = _generate_classification_data(seed=7)
    explicit_rules = [(0, 0), (1, 1)]
    clf = ANFISClassifier(
        n_classes=2,
        n_mfs=2,
        optimizer=DummyTrainer,
        epochs=1,
        verbose=False,
        rules=explicit_rules,
    )
    clf.fit(X, y)

    expected = [tuple(rule) for rule in explicit_rules]
    assert clf.rules_ == expected
    assert captured["rules"] == expected


def test_anfis_classifier_get_rules_requires_fit_and_returns_tuples():
    X, y = _generate_classification_data(seed=19)
    clf = ANFISClassifier(n_classes=2, n_mfs=2, optimizer="sgd", epochs=1, verbose=False)

    with pytest.raises(NotFittedError):
        clf.get_rules()

    clf.fit(X, y)
    rules = clf.get_rules()

    assert isinstance(rules, tuple)
    assert all(isinstance(rule, tuple) for rule in rules)
    assert tuple(clf.rules_) == rules


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
    assert isinstance(clf.training_history_, dict)
    assert len(clf.training_history_["train"]) == clf.optimizer_.epochs


def test_anfis_classifier_fit_accepts_validation_and_extra_params():
    X, y = _generate_classification_data(seed=41)
    y_one_hot = np.eye(2)[y]
    X_val, y_val = X[:12], y_one_hot[:12]

    class RecordingTrainer(BaseTrainer):
        def __init__(self):
            self.epochs = 3
            self.verbose = False
            self.batch_size = None
            self.received_kwargs: dict[str, Any] | None = None

        def fit(self, model, X_fit, y_fit, **kwargs):
            self.received_kwargs = dict(kwargs)
            kwargs.pop("track_logits", None)
            return super().fit(model, X_fit, y_fit, **kwargs)

        def init_state(self, model, X_fit, y_fit):
            return None

        def train_step(self, model, X_batch, y_batch, state):
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):
            return 0.456

    trainer = RecordingTrainer()
    clf = ANFISClassifier(n_classes=2, optimizer=trainer)

    history = clf.fit(
        X,
        y_one_hot,
        validation_data=(X_val, y_val),
        validation_frequency=2,
        track_logits=True,
    ).training_history_

    fitted_trainer = clf.optimizer_
    assert isinstance(fitted_trainer, RecordingTrainer)
    assert fitted_trainer.received_kwargs is not None
    assert fitted_trainer.received_kwargs["validation_data"] == (X_val, y_val)
    assert fitted_trainer.received_kwargs["validation_frequency"] == 2
    assert fitted_trainer.received_kwargs["track_logits"] is True
    assert history is not None
    assert "val" in history
    assert len(history["val"]) == fitted_trainer.epochs
    assert history["val"] == [None, 0.456, None]


def test_anfis_classifier_invalid_optimizer_string():
    X, y = _generate_classification_data(seed=4)
    clf = ANFISClassifier(n_classes=2, optimizer="invalid", epochs=1)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        clf.fit(X, y)


@pytest.mark.parametrize("optimizer", ["hybrid", "hybrid_adam"])
def test_anfis_classifier_rejects_hybrid_aliases(optimizer):
    X, y = _generate_classification_data(seed=5)
    clf = ANFISClassifier(n_classes=2, optimizer=optimizer, epochs=1, verbose=False)
    with pytest.raises(ValueError, match="Hybrid-style"):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "optimizer",
    [HybridTrainer, HybridAdamTrainer, HybridTrainer(), HybridAdamTrainer()],
)
def test_anfis_classifier_rejects_hybrid_objects(optimizer):
    X, y = _generate_classification_data(seed=6)
    clf = ANFISClassifier(n_classes=2, optimizer=optimizer, epochs=1, verbose=False)
    with pytest.raises(ValueError, match="Hybrid-style"):
        clf.fit(X, y)


def test_anfis_classifier_predict_requires_fit():
    clf = ANFISClassifier(n_classes=2)
    with pytest.raises(NotFittedError):
        clf.predict(np.zeros((1, 2)))


def test_anfis_classifier_requires_minimum_classes():
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        ANFISClassifier(n_classes=1)


def test_anfis_classifier_resolve_input_specs_variants():
    clf = ANFISClassifier(
        n_classes=3,
        inputs_config={
            "x2": "bell",
            "x3": 4,
        },
    )

    specs = clf._resolve_input_specs(["feature0", "feature1", "feature2"])
    assert specs[0]["mf_type"] == clf.mf_type
    assert specs[1]["mf_type"] == "bell"
    assert specs[2]["n_mfs"] == 4

    mf = GaussianMF(mean=0.0, sigma=0.5)
    list_spec = clf._normalize_input_spec([mf])
    assert list_spec["membership_functions"][0] is mf
    single_spec = clf._normalize_input_spec(mf)
    assert single_spec["membership_functions"][0] is mf

    with pytest.raises(TypeError, match="Unsupported input configuration type"):
        clf._normalize_input_spec(3.14)


def test_anfis_classifier_encode_targets_validation():
    clf = ANFISClassifier(n_classes=3)
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="configured for 3"):
        clf._encode_targets(y, n_samples=4)

    encoded, classes = clf._encode_targets(y, n_samples=4, allow_partial_classes=True)
    np.testing.assert_array_equal(encoded, y)
    np.testing.assert_array_equal(classes, np.array([0, 1], dtype=object))

    clf_small = ANFISClassifier(n_classes=2)
    y_excess = np.array([0, 1, 2, 1])
    with pytest.raises(ValueError, match="configured for 2"):
        clf_small._encode_targets(y_excess, n_samples=4)

    y_oh = np.eye(4, 3)
    with pytest.raises(ValueError, match="One-hot targets must have shape"):
        clf_small._encode_targets(y_oh, n_samples=4)

    y_oh_mismatch = np.eye(3, 2)
    with pytest.raises(ValueError, match="same number of samples as X"):
        clf_small._encode_targets(y_oh_mismatch, n_samples=2)

    with pytest.raises(ValueError, match="same number of samples as X"):
        clf_small._encode_targets(np.array([0, 1, 1]), n_samples=4)

    with pytest.raises(ValueError, match="exceeds configured n_classes"):
        clf_small._encode_targets(y_excess, n_samples=4, allow_partial_classes=True)

    with pytest.raises(ValueError, match="must be 1-dimensional or a one-hot encoded 2D"):
        clf._encode_targets(np.zeros((2, 2, 1)), n_samples=2)

    clf_no_n = ANFISClassifier()
    y_single_hot = np.ones((4, 1), dtype=float)
    with pytest.raises(ValueError, match="One-hot targets must encode at least two classes"):
        clf_no_n._encode_targets(y_single_hot, n_samples=4)

    clf_partial = ANFISClassifier()
    y_constant = np.zeros(4, dtype=int)
    with pytest.raises(ValueError, match="at least two distinct classes"):
        clf_partial._encode_targets(y_constant, n_samples=4, allow_partial_classes=True)


def test_anfis_classifier_encode_targets_one_hot_success():
    clf = ANFISClassifier(n_classes=3)
    y_oh = np.eye(3)
    encoded, classes = clf._encode_targets(y_oh, n_samples=3)
    np.testing.assert_array_equal(encoded, np.array([0, 1, 2]))
    np.testing.assert_array_equal(classes, np.arange(3))


def test_anfis_classifier_predict_feature_mismatch_after_fit():
    X, y = _generate_classification_data(seed=10)
    clf = ANFISClassifier(
        n_classes=2,
        n_mfs=2,
        optimizer="sgd",
        epochs=2,
        learning_rate=0.05,
        verbose=False,
        random_state=0,
    )

    clf.fit(X, y)

    with pytest.raises(ValueError, match="Feature mismatch"):
        clf.predict_proba(np.zeros((3, 3)))

    with pytest.raises(ValueError, match="Feature mismatch"):
        clf.predict(np.zeros(3))


def test_anfis_classifier_optimizer_invalid_type():
    X, y = _generate_classification_data(seed=11)
    clf = ANFISClassifier(n_classes=2, optimizer=123, verbose=False)
    with pytest.raises(TypeError, match="optimizer must be a string identifier"):
        clf.fit(X, y)


def test_anfis_classifier_predict_proba_accepts_1d_input():
    X, y = _generate_classification_data(seed=15)
    clf = ANFISClassifier(
        n_classes=2,
        n_mfs=2,
        optimizer="sgd",
        epochs=2,
        learning_rate=0.05,
        verbose=False,
        random_state=0,
    )
    clf.fit(X, y)

    single = X[0]
    proba = clf.predict_proba(single)
    assert proba.shape == (1, 2)


def test_anfis_classifier_predict_runtime_checks():
    X, y = _generate_classification_data(seed=12)
    clf = ANFISClassifier(n_classes=2, n_mfs=2, optimizer="sgd", epochs=2, verbose=False, random_state=0)
    clf.fit(X, y)

    clf.n_features_in_ = None
    with pytest.raises(RuntimeError, match="predict"):
        clf.predict(X[:1])

    clf.n_features_in_ = None
    with pytest.raises(RuntimeError, match="predict_proba"):
        clf.predict_proba(X[:1])


def test_anfis_classifier_evaluate_prints_results(capsys):
    X, y = _generate_classification_data(seed=13)
    clf = ANFISClassifier(n_classes=2, n_mfs=2, optimizer="sgd", epochs=2, verbose=False, random_state=0)
    clf.fit(X, y)

    metrics = clf.evaluate(X[:4], y[:4])
    captured = capsys.readouterr().out
    assert "ANFISClassifier evaluation:" in captured
    assert "accuracy" in captured
    assert "confusion_matrix" in captured
    assert "log_loss" not in captured
    assert "accuracy" in metrics
    assert np.isnan(metrics.get("log_loss", np.nan))


def test_anfis_classifier_evaluate_filters_nan_metrics(monkeypatch, capsys):
    clf = ANFISClassifier(n_classes=3)
    clf.n_classes = 3
    clf.classes_ = np.array([0, 1, 2])
    clf.n_features_in_ = 2
    clf.feature_names_in_ = ["x1", "x2"]
    clf.rules_ = []

    class DummyModel:
        def predict_proba(self, X):
            probs = np.array([0.6, 0.3, 0.1], dtype=float)
            return np.tile(probs, (X.shape[0], 1))

    clf.model_ = DummyModel()
    clf._mark_fitted()

    X = np.zeros((3, 2), dtype=float)
    y = np.array([0, 1, 2], dtype=int)

    metrics_template = {
        "accuracy": 0.75,
        "support": 5,
        "vector": np.array([0.1, 0.9]),
        "matrix": np.arange(4, dtype=float).reshape(2, 2),
        "empty": np.array([], dtype=float),
        "object_array": np.array(["a", "b"], dtype=object),
        "skip_nan": float("nan"),
        "nan_array": np.array([[np.nan, np.nan]]),
        "none_val": None,
        "info": "note",
        "log_loss": float("nan"),
    }

    monkeypatch.setattr(
        ANFISMetrics,
        "classification_metrics",
        lambda y_true, proba: dict(metrics_template),
    )

    result = clf.evaluate(X, y, return_dict=True, print_results=True)
    out = capsys.readouterr().out

    assert "ANFISClassifier evaluation:" in out
    assert "  accuracy: 0.750000" in out
    assert "  support: 5" in out
    assert "  vector: [0.1 0.9]" in out
    assert "  matrix:\n    [[0. 1.]\n     [2. 3.]]" in out
    assert "  empty: []" in out
    assert "  object_array: ['a' 'b']" in out
    assert "  info: note" in out
    assert "skip_nan" not in out
    assert "nan_array" not in out
    assert "none_val" not in out
    assert "log_loss" not in result

    clf.evaluate(X, y, print_results=False)
    suppressed = capsys.readouterr().out
    assert suppressed == ""


def test_anfis_classifier_build_model_respects_range_overrides(monkeypatch):
    X = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]], dtype=float)
    custom_mfs = [GaussianMF(mean=-0.5, sigma=0.3), GaussianMF(mean=0.5, sigma=0.3)]
    inputs_config = {
        "x1": {"membership_functions": custom_mfs, "range": (-2.0, 2.0)},
        "x2": {"range": (-1.0, 1.0), "n_mfs": 2, "mf_type": "gaussian"},
    }
    created_builders: list[object] = []

    class StubBuilder:
        def __init__(self):
            self.input_mfs: dict[str, list[MembershipFunction]] = {}
            self.input_ranges: dict[str, tuple[float, float]] = {}
            self.add_input_calls: list[tuple] = []
            self.add_input_from_data_calls: list[tuple] = []
            created_builders.append(self)

        def add_input(self, name, rmin, rmax, n_mfs, mf_type, *, overlap=None):
            self.add_input_calls.append((name, rmin, rmax, n_mfs, mf_type, overlap))

        def add_input_from_data(self, *args, **kwargs):
            self.add_input_from_data_calls.append((args, kwargs))

    class StubClassifier:
        def __init__(self, input_mfs, n_classes, random_state=None, rules=None):
            self.membership_functions = input_mfs
            self.n_classes = n_classes
            self.random_state = random_state
            self.rules = rules

    monkeypatch.setattr("anfis_toolbox.classifier.ANFISBuilder", StubBuilder)
    monkeypatch.setattr("anfis_toolbox.classifier.TSKANFISClassifier", StubClassifier)

    clf = ANFISClassifier(n_classes=2, n_mfs=2, inputs_config=inputs_config)
    feature_names = ["x1", "x2"]
    clf.input_specs_ = clf._resolve_input_specs(feature_names)

    model = clf._build_model(X, feature_names)
    builder = created_builders[-1]

    assert model.membership_functions["x1"][0] is custom_mfs[0]
    assert builder.input_ranges["x1"] == (-2.0, 2.0)
    assert builder.add_input_calls[0][0] == "x2"
    assert builder.add_input_calls[0][1:3] == (-1.0, 1.0)


def test_anfis_classifier_optimizer_class_params():
    X, y = _generate_classification_data(seed=14)
    clf = ANFISClassifier(
        n_classes=2,
        n_mfs=2,
        optimizer=SGDTrainer,
        optimizer_params={"epochs": 3, "bogus": 99},
        learning_rate=0.05,
        shuffle=False,
        verbose=False,
        random_state=0,
    )

    clf.fit(X, y)

    assert isinstance(clf.optimizer_, SGDTrainer)
    assert clf.optimizer_.epochs == 3
    assert clf.optimizer_.shuffle is False
    assert pytest.approx(clf.optimizer_.learning_rate, rel=1e-6) == 0.05


def test_anfis_classifier_resolved_loss_spec_defaults():
    clf = ANFISClassifier(n_classes=2)
    assert clf._resolved_loss_spec() == "cross_entropy"
    clf.loss = "mse"
    assert clf._resolved_loss_spec() == "mse"


def test_anfis_classifier_collect_trainer_params_skips_self(monkeypatch):
    X, y = _generate_classification_data(seed=16)

    class DummyTrainer(BaseTrainer):
        def __init__(self, alpha=1, beta=2):
            self.alpha = alpha
            self.beta = beta

        def fit(self, model, X_fit, y_fit):
            return {"train": [0.0]}

        def init_state(self, model, X_fit, y_fit):
            return None

        def train_step(self, model, Xb, yb, state):
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):
            return 0.0

    real_signature = inspect.signature

    def fake_signature(obj):
        if obj is DummyTrainer:
            return inspect.Signature(
                parameters=[
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("alpha", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1),
                    inspect.Parameter("beta", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=2),
                ]
            )
        return real_signature(obj)

    monkeypatch.setattr("anfis_toolbox.classifier.inspect.signature", fake_signature)

    clf = ANFISClassifier(
        n_classes=2,
        optimizer=DummyTrainer,
        optimizer_params={"alpha": 5},
        verbose=False,
    )
    clf.fit(X, y)

    assert isinstance(clf.optimizer_, DummyTrainer)
    assert clf.optimizer_.alpha == 5


def test_anfis_classifier_custom_trainer_without_loss():
    X, y = _generate_classification_data(seed=17)

    class NoLossTrainer(BaseTrainer):
        def __init__(self):
            self.epochs = 1
            self.learning_rate = 0.01

        def fit(self, model, X_fit, y_fit):
            return {"train": [0.0]}

        def init_state(self, model, X_fit, y_fit):
            return None

        def train_step(self, model, Xb, yb, state):
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):
            return 0.0

    trainer = NoLossTrainer()
    clf = ANFISClassifier(
        n_classes=2,
        optimizer=trainer,
        epochs=3,
        learning_rate=0.2,
        verbose=False,
        loss="mse",
    )

    clf.fit(X, y)

    assert isinstance(clf.optimizer_, NoLossTrainer)
    assert clf.optimizer_.epochs == 3
    assert not hasattr(clf.optimizer_, "loss")


def test_classifier_fit_forwards_validation_kwargs():
    X, labels = _generate_classification_data(seed=23)
    y = np.eye(2)[labels]
    X_val = X[:4]
    y_val = np.eye(2)[labels[:4]]

    captured: dict[str, object] = {}

    class RecordingTrainer:
        def fit(self, model, X_fit, y_fit, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs
            captured["X"] = X_fit
            captured["y"] = y_fit
            return {"train": [0.0], "val": [0.0]}

    classifier = TSKANFISClassifier(make_simple_input_mfs(n_features=2, n_mfs=2), n_classes=2)
    history = classifier.fit(
        X,
        y,
        trainer=RecordingTrainer(),
        validation_data=(X_val, y_val),
        validation_frequency=5,
    )

    assert history == {"train": [0.0], "val": [0.0]}
    assert captured["model"] is classifier
    np.testing.assert_array_equal(captured["X"], X)
    np.testing.assert_array_equal(captured["y"], y)
    forwarded = captured["kwargs"]
    assert forwarded["validation_frequency"] == 5
    val_X, val_y = forwarded["validation_data"]
    np.testing.assert_array_equal(val_X, X_val)
    np.testing.assert_array_equal(val_y, y_val)


def test_classifier_fit_raises_for_non_dict_history():
    X, labels = _generate_classification_data(seed=24)
    y = np.eye(2)[labels]

    class BadTrainer:
        def fit(self, model, X_fit, y_fit, **kwargs):
            return [0.0]

    classifier = TSKANFISClassifier(make_simple_input_mfs(n_features=2, n_mfs=2), n_classes=2)
    with pytest.raises(TypeError, match="Trainer.fit must return a TrainingHistory dictionary"):
        classifier.fit(X, y, trainer=BadTrainer())


@pytest.mark.parametrize("optimizer", ["sgd", "adam", "pso", "rmsprop"])
def test_moon(optimizer):
    X = np.array(
        [
            [1.73501091, -0.10605525],
            [0.64901897, 0.6545898],
            [-0.05713802, 0.40759172],
            [0.7387451, 0.09503697],
            [1.08164451, -0.6523876],
            [-0.04280461, 0.92575932],
            [1.92966562, 0.28603793],
            [-1.0629475, 0.05977205],
            [-0.45115798, 0.74653008],
            [0.30511514, -0.25865035],
        ]
    )
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1])

    clf = ANFISClassifier(
        n_classes=2,
        n_mfs=2,
        optimizer=optimizer,
        epochs=5,
        learning_rate=0.1,
        verbose=False,
        random_state=0,
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    acc = accuracy(y, proba)
    assert acc >= 0.7


def test_classifier_repr_pre_and_post_fit():
    clf = ANFISClassifier(
        n_classes=3,
        n_mfs=2,
        mf_type="gaussian",
        init="grid",
        optimizer="sgd",
        shuffle=True,
    )

    prefit = repr(clf)
    assert prefit.startswith("ANFISClassifier(")
    assert "n_classes=3" in prefit
    assert "optimizer='sgd'" in prefit
    assert "model_" not in prefit

    class DummyModel:
        n_inputs = 2
        n_rules = 4
        n_classes = 3
        input_names = ["x1", "x2"]
        membership_functions = {
            "x1": [object(), object()],
            "x2": [object(), object(), object()],
        }

    clf.model_ = DummyModel()
    clf.optimizer_ = SimpleNamespace(learning_rate=0.01, epochs=5, batch_size=4)
    clf.training_history_ = {"train": [1.0, 0.3], "val": [0.5, 0.2]}
    clf.rules_ = [(0, 0), (1, 1), (2, 2)]
    clf.feature_names_in_ = ["x1", "x2"]
    clf.classes_ = np.arange(8)
    clf._mark_fitted()

    fitted = repr(clf)
    assert "model_" in fitted
    assert "optimizer_" in fitted
    assert "training_history_" in fitted
    assert "classes_" in fitted
    assert "rules_" in fitted
    assert "feature_names_in_" in fitted
    assert "DummyModel" in fitted
    assert "SimpleNamespace" in fitted
    assert "classes_: 0, 1, 2, 3, 4, ..." in fitted
    assert "├─" in fitted or "|--" in fitted


def test_classifier_save_and_load_roundtrip(tmp_path):
    X, y = _generate_classification_data()
    clf = ANFISClassifier(
        n_classes=2,
        n_mfs=2,
        mf_type="gaussian",
        optimizer="sgd",
        learning_rate=0.05,
        epochs=5,
        random_state=0,
        shuffle=False,
    )
    clf.fit(X, y)
    baseline = clf.predict_proba(X[:5])

    path = tmp_path / "nested" / "classifier.pkl"
    clf.save(path)

    loaded = ANFISClassifier.load(path)
    assert isinstance(loaded, ANFISClassifier)
    assert loaded.is_fitted_
    assert np.array_equal(loaded.classes_, clf.classes_)
    np.testing.assert_allclose(loaded.predict_proba(X[:5]), baseline, atol=1e-6)


def test_classifier_load_rejects_wrong_type(tmp_path):
    path = tmp_path / "wrong.pkl"
    with path.open("wb") as stream:
        pickle.dump(ANFISRegressor(), stream)

    with pytest.raises(TypeError, match="Expected pickled ANFISClassifier"):
        ANFISClassifier.load(path)


def test_classifier_repr_config_pairs_optional_sections():
    clf = ANFISClassifier(
        n_classes=3,
        rules=[(0, 1)],
        optimizer=SGDTrainer,
        optimizer_params={"momentum": 0.9},
    )
    pairs = clf._repr_config_pairs()
    assert ("optimizer", "SGDTrainer") in pairs
    assert ("rules", "preset:1") in pairs
    assert ("optimizer_params", {"momentum": 0.9}) in pairs

    instance_label = ANFISClassifier._describe_optimizer_config(SGDTrainer())
    assert instance_label == "SGDTrainer"

    generic_label = ANFISClassifier._describe_optimizer_config(object())
    assert generic_label.startswith("<")


def test_classifier_summarize_optimizer_history_and_classes_preview():
    class BareTrainer(BaseTrainer):
        def fit(self, model, X_fit, y_fit, **kwargs):  # pragma: no cover - dummy
            return {}

        def init_state(self, model, X_fit, y_fit):  # pragma: no cover - dummy
            return None

        def train_step(self, model, X_batch, y_batch, state):  # pragma: no cover - dummy
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):  # pragma: no cover - dummy
            return 0.0

        def __repr__(self) -> str:  # pragma: no cover - simple repr
            return "BareTrainer(classifier=True)"

    class NoneTrainer(BaseTrainer):
        learning_rate = None
        epochs = None

        def fit(self, model, X_fit, y_fit, **kwargs):  # pragma: no cover - dummy
            return {}

        def init_state(self, model, X_fit, y_fit):  # pragma: no cover - dummy
            return None

        def train_step(self, model, X_batch, y_batch, state):  # pragma: no cover - dummy
            return 0.0, state

        def compute_loss(self, model, X_eval, y_eval):  # pragma: no cover - dummy
            return 0.0

    clf = ANFISClassifier(n_classes=2)
    optimizer_summary = clf._summarize_optimizer(BareTrainer())
    assert optimizer_summary == "BareTrainer(classifier=True)"

    none_summary = clf._summarize_optimizer(NoneTrainer())
    assert "NoneTrainer" in none_summary

    history_zero = ANFISClassifier._summarize_history({"metrics": []})
    assert "metrics=0" in history_zero

    history_unknown = ANFISClassifier._summarize_history({"foo": 1})
    assert history_unknown == "{}"

    clf.classes_ = np.array(["a", "b", "c"])
    clf._mark_fitted()
    children = dict(clf._repr_children_entries())
    assert children["classes_"] == "a, b, c"


def test_classifier_normalize_input_spec_variants():
    clf = ANFISClassifier(n_classes=2)
    gaussian = GaussianMF(mean=0.0, sigma=1.0)

    from_none = clf._normalize_input_spec(None)
    assert from_none["n_mfs"] == clf.n_mfs

    from_list = clf._normalize_input_spec([gaussian])
    assert from_list["membership_functions"] == [gaussian]

    from_str = clf._normalize_input_spec("triangular")
    assert from_str["mf_type"] == "triangular"

    from_int = clf._normalize_input_spec(6)
    assert from_int["n_mfs"] == 6

    from_dict = clf._normalize_input_spec({"mfs": [gaussian], "range": (-1, 1)})
    assert from_dict["membership_functions"] == [gaussian]
    assert from_dict["range"] == (-1, 1)

    with pytest.raises(TypeError):
        clf._normalize_input_spec(1.5)


def test_classifier_repr_children_handles_missing_artifacts():
    clf = ANFISClassifier(n_classes=2)
    clf._mark_fitted()
    clf.model_ = None
    clf.optimizer_ = None
    clf.training_history_ = {}
    clf.classes_ = None
    clf.rules_ = None
    clf.feature_names_in_ = None
    assert clf._repr_children_entries() == []


def test_classifier_summarize_model_and_history_non_numeric_tail():
    clf = ANFISClassifier(n_classes=2)

    class MinimalModel:
        membership_functions: dict[str, list[object]] = {}

    summary = clf._summarize_model(MinimalModel())
    assert summary == "MinimalModel"

    history = ANFISClassifier._summarize_history({"val": [{"loss": 1.0}]})
    assert "val=1" in history

    assert ANFISClassifier._describe_optimizer_config(None) is None


def test_classifier_predict_requires_known_feature_count():
    clf = ANFISClassifier(n_classes=2)
    clf.is_fitted_ = True
    clf.model_ = SimpleNamespace(predict=lambda _X: np.zeros((1, 1)))
    clf.classes_ = np.array([0, 1])
    clf.n_features_in_ = None

    with pytest.raises(RuntimeError, match="Model must be fitted before calling predict."):
        clf.predict(np.array([[0.1]]))


def test_classifier_predict_proba_requires_known_feature_count():
    clf = ANFISClassifier(n_classes=2)
    clf.is_fitted_ = True
    clf.model_ = SimpleNamespace(predict_proba=lambda _X: np.ones((1, 2)))
    clf.n_features_in_ = None

    with pytest.raises(RuntimeError, match="Model must be fitted before calling predict_proba."):
        clf.predict_proba(np.array([[0.1]]))


def test_classifier_predict_requires_model_instance():
    clf = ANFISClassifier(n_classes=2)
    clf.is_fitted_ = True
    clf.model_ = None
    clf.classes_ = np.array([0, 1])
    clf.n_features_in_ = 1

    with pytest.raises(RuntimeError, match="Model must be fitted before calling predict."):
        clf.predict(np.array([[0.1]]))


def test_classifier_predict_proba_requires_model_instance():
    clf = ANFISClassifier(n_classes=2)
    clf.is_fitted_ = True
    clf.model_ = None
    clf.n_features_in_ = 1

    with pytest.raises(RuntimeError, match="Model must be fitted before calling predict_proba."):
        clf.predict_proba(np.array([[0.1]]))


def test_classifier_build_model_requires_input_specs():
    clf = ANFISClassifier(n_classes=2)
    X = np.array([[0.0], [1.0]])
    clf.input_specs_ = None

    with pytest.raises(RuntimeError, match="Input specifications must be resolved"):
        clf._build_model(X, ["x0"])


def test_classifier_build_model_requires_known_class_count():
    clf = ANFISClassifier()
    clf.input_specs_ = [clf._normalize_input_spec(None)]
    X = np.array([[0.0], [1.0]])

    with pytest.raises(RuntimeError, match="Number of classes must be known before constructing"):
        clf._build_model(X, ["x0"])


@pytest.mark.parametrize(
    "spec",
    [
        {"membership_functions": [GaussianMF(mean=0.0, sigma=1.0)], "range": (0.0,)},
        {"range": (0.0,)},
    ],
)
def test_classifier_build_model_range_override_requires_two_values(spec):
    clf = ANFISClassifier(n_classes=2)
    normalized = clf._normalize_input_spec(spec)
    clf.input_specs_ = [normalized]
    X = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="range overrides must contain exactly two values"):
        clf._build_model(X, ["x0"])


def test_low_level_classifier_requires_trainer_protocol():
    mfs = make_simple_input_mfs(n_features=1, n_mfs=2)
    clf = TSKANFISClassifier(mfs, n_classes=2)
    X = np.array([[0.0], [0.5]])
    y = np.array([0, 1])

    with pytest.raises(TypeError, match="trainer must implement fit"):
        clf.fit(X, y, trainer=object())
