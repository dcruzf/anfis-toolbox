import numpy as np
import pytest

from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.model import TSKANFIS, TSKANFISClassifier
from anfis_toolbox.optim import RMSPropTrainer


def _make_regression_model(n_inputs: int = 2) -> TSKANFIS:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return TSKANFIS(input_mfs)


def test_rmsprop_trains_full_batch_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (0.8 * X[:, 0] - 0.2 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = RMSPropTrainer(learning_rate=0.01, epochs=3, batch_size=None, shuffle=False, verbose=False)
    history = trainer.fit(model, X, y)
    train_losses = history["train"]
    assert len(train_losses) == 3
    assert all(np.isfinite(loss) and loss >= 0 for loss in train_losses)


def test_rmsprop_trains_minibatch_regression_and_updates_params():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + 0.5 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    params_before = model.get_parameters()
    trainer = RMSPropTrainer(learning_rate=0.005, epochs=2, batch_size=8, shuffle=True, verbose=False)
    history = trainer.fit(model, X, y)
    assert len(history["train"]) == 2
    params_after = model.get_parameters()
    # Consequent should differ due to RMSProp updates
    assert not np.allclose(params_before["consequent"], params_after["consequent"])


def _make_classifier(n_inputs: int = 1, n_classes: int = 2) -> TSKANFISClassifier:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return TSKANFISClassifier(input_mfs, n_classes=n_classes, random_state=0)


def test_rmsprop_with_classifier_does_not_error_on_forward_backward():
    # RMSProp trainer computes MSE by default; we just exercise mechanics with classifier logits
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 1))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)  # treat as regression target to check plumbing
    clf = _make_classifier(n_inputs=1, n_classes=2)
    trainer = RMSPropTrainer(learning_rate=0.005, epochs=1, batch_size=5, shuffle=False, verbose=False)
    history = trainer.fit(clf, X, y)
    assert len(history["train"]) == 1 and np.isfinite(history["train"][0])


def test_rmsprop_accepts_1d_target_and_reshapes():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(25, 2))
    # 1D target to exercise the internal reshape branch
    y = 0.6 * X[:, 0] + 0.3 * X[:, 1]
    model = _make_regression_model(n_inputs=2)
    trainer = RMSPropTrainer(learning_rate=0.01, epochs=1, batch_size=None, shuffle=False, verbose=False)
    history = trainer.fit(model, X, y)
    assert len(history["train"]) == 1 and np.isfinite(history["train"][0])


def test_rmsprop_classifier_with_cross_entropy_loss():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(24, 1))
    y = (X[:, 0] > 0).astype(int)
    clf = _make_classifier(n_inputs=1, n_classes=2)
    trainer = RMSPropTrainer(
        learning_rate=0.01,
        epochs=2,
        batch_size=None,
        shuffle=False,
        verbose=False,
        loss="cross_entropy",
    )
    history = trainer.fit(clf, X, y)
    assert len(history["train"]) == 2
    assert all(np.isfinite(loss) for loss in history["train"])


def test_rmsprop_fit_raises_when_target_rows_mismatch():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=(5, 1))
    model = _make_regression_model(n_inputs=2)
    trainer = RMSPropTrainer(learning_rate=0.01, epochs=1, batch_size=None, shuffle=False, verbose=False)

    with pytest.raises(ValueError, match="Target array must have same number of rows as X"):
        trainer.fit(model, X, y)


class _ToyOptimModel:
    def __init__(self):
        self.consequent = np.zeros((1, 2), dtype=float)
        self.membership_params = {"x": [{"mean": 0.0, "sigma": 1.0}]}
        self._grads: dict | None = None

    def get_parameters(self):
        return {
            "consequent": self.consequent.copy(),
            "membership": {"x": [self.membership_params["x"][0].copy()]},
        }

    def set_parameters(self, params):
        self.consequent = params["consequent"].copy()
        self.membership_params = {
            name: [mf.copy() for mf in params["membership"][name]] for name in params["membership"]
        }

    def reset_gradients(self):
        self._grads = None

    def forward(self, X):
        return np.zeros((X.shape[0], 1), dtype=float)

    def backward(self, grad):
        self._grads = {
            "consequent": np.full_like(self.consequent, 0.25, dtype=float),
            "membership": {"x": [{"mean": -0.3, "sigma": 0.15}]},
        }

    def get_gradients(self):
        if self._grads is None:
            raise RuntimeError("Gradients not computed")
        return {
            "consequent": self._grads["consequent"].copy(),
            "membership": {"x": [self._grads["membership"]["x"][0].copy()]},
        }


def test_rmsprop_train_step_lazy_initializes_loss():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(12, 2))
    y = (0.4 * X[:, 0] - 0.6 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = RMSPropTrainer(learning_rate=0.01, epochs=1, batch_size=4, shuffle=False, verbose=False)

    X_prepared, y_prepared = trainer._prepare_training_data(model, X, y)
    assert hasattr(trainer, "_loss_fn")
    state = trainer.init_state(model, X_prepared, y_prepared)

    loss, updated_state = trainer.train_step(model, X_prepared[:4], y_prepared[:4], state)

    assert updated_state is state
    assert np.isfinite(loss)
    assert hasattr(trainer, "_loss_fn")

    # Second call should reuse cached loss without re-resolving
    loss2, state_again = trainer.train_step(model, X_prepared[4:8], y_prepared[4:8], state)
    assert state_again is state
    assert np.isfinite(loss2)


def test_rmsprop_apply_step_updates_membership_and_cache():
    trainer = RMSPropTrainer(learning_rate=0.01, epochs=1, batch_size=None, shuffle=False, verbose=False)
    model = _ToyOptimModel()
    X = np.array([[0.0], [1.0]], dtype=float)
    y = np.zeros((2, 1), dtype=float)

    loss_fn = trainer._get_loss_fn()
    assert loss_fn is not None
    assert trainer._get_loss_fn() is loss_fn

    trainer._prepare_training_data(model, X, y)
    state = trainer.init_state(model, X, y)

    loss, updated_state = trainer.train_step(model, X, y, state)

    assert updated_state is state
    assert np.isfinite(loss)
    assert model.membership_params["x"][0]["mean"] != 0.0
    assert state["cache"]["membership"]["x"][0]["mean"] != 0.0
    assert np.all(state["cache"]["consequent"] != 0.0)
    assert trainer.compute_loss(model, X, y) == pytest.approx(0.0, abs=1e-12)


class _CaptureModel:
    def __init__(self):
        self.last_params = None

    def set_parameters(self, params):
        self.last_params = {
            "consequent": params["consequent"].copy(),
            "membership": {name: [mf.copy() for mf in params["membership"][name]] for name in params["membership"]},
        }


def test_rmsprop_apply_step_updates_membership_nested_structure_directly():
    trainer = RMSPropTrainer(learning_rate=0.01, epochs=1, batch_size=None, shuffle=False, verbose=False)
    params = {
        "consequent": np.zeros((1, 1), dtype=float),
        "membership": {"x": [{"a": 0.0}]},
    }
    grads = {
        "consequent": np.full((1, 1), 0.25, dtype=float),
        "membership": {"x": [{"a": -0.4}]},
    }
    cache = {"consequent": np.zeros_like(params["consequent"]), "membership": {"x": [{"a": 0.0}]}}
    model = _CaptureModel()

    trainer._apply_rmsprop_step(model, params, cache, grads)

    assert model.last_params is not None
    assert model.last_params["membership"]["x"][0]["a"] > 0.0
    assert cache["membership"]["x"][0]["a"] != 0.0
    assert np.all(cache["consequent"] != 0.0)


def test_rmsprop_prepare_validation_data_checks_rows():
    trainer = RMSPropTrainer(learning_rate=0.01, epochs=1, batch_size=None, shuffle=False, verbose=False)
    model = _ToyOptimModel()
    X = np.arange(6, dtype=float).reshape(3, 2)
    y = np.array([0.0, 1.0, 2.0])

    trainer._prepare_training_data(model, X, y)

    X_val = X[:2]
    y_val = y[:2]
    X_val_prepared, y_val_prepared = trainer._prepare_validation_data(model, X_val, y_val)
    assert X_val_prepared.shape == (2, 2)
    assert y_val_prepared.shape == (2, 1)

    with pytest.raises(ValueError, match="Target array must have same number of rows"):
        trainer._prepare_validation_data(model, X_val, y)
