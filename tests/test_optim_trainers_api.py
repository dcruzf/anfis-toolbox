import numpy as np
import pytest

from anfis_toolbox.losses import LossFunction
from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.model import TSKANFIS, TSKANFISClassifier
from anfis_toolbox.optim import AdamTrainer, HybridTrainer, RMSPropTrainer, SGDTrainer
from anfis_toolbox.optim.base import BaseTrainer


def _make_regression_model(n_inputs: int = 2) -> TSKANFIS:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return TSKANFIS(input_mfs)


def _make_classifier(n_inputs: int = 1, n_classes: int = 2) -> TSKANFISClassifier:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return TSKANFISClassifier(input_mfs, n_classes=n_classes, random_state=0)


class _DummyModel:
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.scale * X.sum(axis=1, keepdims=True)

    def reset_gradients(self) -> None:
        pass

    def backward(self, grad: np.ndarray) -> None:
        pass

    def update_parameters(self, lr: float) -> None:
        pass


class _SimpleTrainer(BaseTrainer):
    def __init__(self, *, epochs: int = 2, verbose: bool = True):
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = None
        self.shuffle = False

    def init_state(self, model, X: np.ndarray, y: np.ndarray):
        return None

    def train_step(self, model, Xb: np.ndarray, yb: np.ndarray, state):
        preds = model.forward(Xb)
        loss = float(np.mean((preds - yb) ** 2))
        return loss, state

    def compute_loss(self, model, X: np.ndarray, y: np.ndarray) -> float:
        preds = model.forward(X)
        return float(np.mean((preds - y) ** 2))


def test_sgd_train_step_and_init_state():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(10, 2))
    y = (0.5 * X[:, 0] - 0.25 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = SGDTrainer(learning_rate=0.01)
    X_prepared, y_prepared = trainer._prepare_training_data(model, X, y)
    state = trainer.init_state(model, X_prepared, y_prepared)
    assert state is None
    params_before = model.get_parameters()
    loss, state_after = trainer.train_step(model, X_prepared[:5], y_prepared[:5], state)
    assert np.isfinite(loss)
    params_after = model.get_parameters()
    assert not np.allclose(params_before["consequent"], params_after["consequent"])  # parameters updated
    assert state_after is None


def test_adam_train_step_and_state_progress():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(12, 2))
    y = (X[:, 0] + 0.3 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = AdamTrainer(learning_rate=0.01)
    X_prepared, y_prepared = trainer._prepare_training_data(model, X, y)
    state = trainer.init_state(model, X_prepared, y_prepared)
    assert isinstance(state, dict) and set(state.keys()) == {"params", "m", "v", "t"}
    t0 = state["t"]
    params_before = model.get_parameters()
    loss, state = trainer.train_step(model, X_prepared[:6], y_prepared[:6], state)
    assert np.isfinite(loss)
    assert state["t"] == t0 + 1
    params_after = model.get_parameters()
    assert not np.allclose(params_before["consequent"], params_after["consequent"])  # updated by Adam


def test_rmsprop_train_step_and_state():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(12, 2))
    y = (0.1 * X[:, 0] - 0.2 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = RMSPropTrainer(learning_rate=0.01)
    X_prepared, y_prepared = trainer._prepare_training_data(model, X, y)
    state = trainer.init_state(model, X_prepared, y_prepared)
    assert isinstance(state, dict) and set(state.keys()) == {"params", "cache"}
    params_before = model.get_parameters()
    loss, state = trainer.train_step(model, X_prepared[:6], y_prepared[:6], state)
    assert np.isfinite(loss)
    params_after = model.get_parameters()
    assert not np.allclose(params_before["consequent"], params_after["consequent"])  # updated by RMSProp


def test_hybrid_train_step_no_state_and_updates():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(14, 2))
    y = (0.7 * X[:, 0] + 0.2 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = HybridTrainer(learning_rate=0.01)
    state = trainer.init_state(model, X, y)
    assert state is None
    params_before = model.get_parameters()
    loss, state_after = trainer.train_step(model, X[:7], y[:7], state)
    assert np.isfinite(loss)
    params_after = model.get_parameters()
    # Hybrid updates consequents via LSM inside the step
    assert not np.allclose(params_before["consequent"], params_after["consequent"])  # updated by Hybrid
    assert state_after is None


def test_hybrid_train_step_accepts_1d_target_and_reshapes():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(10, 2))
    # 1D target to trigger internal reshape in _prepare_data
    y = 0.4 * X[:, 0] - 0.1 * X[:, 1]
    model = _make_regression_model(n_inputs=2)
    trainer = HybridTrainer(learning_rate=0.01)
    state = trainer.init_state(model, X, y)
    loss, state_after = trainer.train_step(model, X[:5], y[:5], state)
    assert np.isfinite(loss)
    assert state_after is None


def test_base_trainer_fit_with_validation_and_logging(caplog):
    model = _DummyModel()
    trainer = _SimpleTrainer(epochs=2, verbose=True)
    X = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5]], dtype=float)
    y = np.array([0.0, 0.5, 1.5])  # 1D target to trigger reshape
    X_val = X[:2]
    y_val = y[:2]

    caplog.set_level("INFO")
    history = trainer.fit(model, X, y, validation_data=(X_val, y_val), validation_frequency=2)

    assert "train" in history and "val" in history
    assert history["val"][0] is None and history["val"][1] is not None
    messages = [record.message for record in caplog.records if "train_loss" in record.message]
    assert any("val_loss" in msg for msg in messages)


def test_base_trainer_validation_frequency_must_be_positive():
    model = _DummyModel()
    trainer = _SimpleTrainer()
    X = np.zeros((4, 2))
    y = np.zeros(4)

    with pytest.raises(ValueError, match="validation_frequency"):
        trainer.fit(model, X, y, validation_frequency=0)


def test_base_trainer_fit_without_verbose_skips_logging(caplog):
    model = _DummyModel()
    trainer = _SimpleTrainer(epochs=1, verbose=False)
    X = np.array([[0.0, 1.0], [1.0, -1.0]], dtype=float)
    y = np.array([1.0, 0.0])

    caplog.set_level("INFO")
    trainer.fit(model, X, y)

    assert not caplog.records


def test_base_trainer_log_epoch_handles_val_loss(caplog):
    trainer = _SimpleTrainer()

    caplog.set_level("INFO")
    trainer._log_epoch(epoch_idx=0, train_loss=0.1, val_loss=None, verbose=True)
    trainer._log_epoch(epoch_idx=1, train_loss=0.2, val_loss=0.3, verbose=True)

    assert "val_loss" not in caplog.records[0].message
    assert "val_loss" in caplog.records[1].message


def test_base_trainer_prepare_training_data_checks_shapes():
    trainer = _SimpleTrainer()
    X = np.ones((3, 2))
    y = np.array([1.0, 2.0, 3.0])

    X_prepared, y_prepared = trainer._prepare_training_data(_DummyModel(), X, y)
    assert X_prepared.shape == (3, 2)
    assert y_prepared.shape == (3, 1)

    with pytest.raises(ValueError, match="Target array must have same number of rows"):
        trainer._prepare_training_data(_DummyModel(), X, np.array([1.0, 2.0]))


def test_base_trainer_prepare_validation_data_uses_training_logic():
    trainer = _SimpleTrainer()
    X = np.arange(6, dtype=float).reshape(3, 2)
    y = np.array([0.0, 1.0, 2.0])

    X_prepared, y_prepared = trainer._prepare_validation_data(_DummyModel(), X, y)
    assert X_prepared.shape == (3, 2)
    assert y_prepared.shape == (3, 1)

    with pytest.raises(ValueError, match="Target array must have same number of rows"):
        trainer._prepare_validation_data(_DummyModel(), X, np.array([1.0, 2.0]))


def test_sgd_prepare_validation_data_checks_rows():
    trainer = SGDTrainer(epochs=1)
    model = _DummyModel()
    X = np.arange(6, dtype=float).reshape(3, 2)
    y = np.array([0.0, 1.0, 2.0])

    X_train, y_train = trainer._prepare_training_data(model, X, y)
    assert X_train.shape == (3, 2)
    assert y_train.shape == (3, 1)

    X_val = X[:2]
    y_val = y[:2]
    X_val_prepared, y_val_prepared = trainer._prepare_validation_data(model, X_val, y_val)
    assert X_val_prepared.shape == (2, 2)
    assert y_val_prepared.shape == (2, 1)

    loss_value = trainer.compute_loss(model, X_train, y_train)
    assert loss_value >= 0

    with pytest.raises(ValueError, match="Validation targets must match input rows"):
        trainer._prepare_validation_data(model, X_val, y)


def test_hybrid_fit_uses_pinv_on_solve_error(monkeypatch):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(16, 2))
    y = (0.3 * X[:, 0] - 0.6 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = HybridTrainer(learning_rate=0.01, epochs=1)

    original_solve = np.linalg.solve

    def _raise_linalg_error(*args, **kwargs):
        raise np.linalg.LinAlgError

    monkeypatch.setattr(np.linalg, "solve", _raise_linalg_error)
    try:
        history = trainer.fit(model, X, y)
    finally:
        monkeypatch.setattr(np.linalg, "solve", original_solve)
    assert isinstance(history, dict)
    train_losses = history["train"]
    assert len(train_losses) == 1 and np.isfinite(train_losses[0])


def test_hybrid_train_step_uses_pinv_on_solve_error(monkeypatch):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(12, 2))
    y = (0.9 * X[:, 0] - 0.1 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = HybridTrainer(learning_rate=0.01)

    original_solve = np.linalg.solve

    def _raise_linalg_error(*args, **kwargs):
        raise np.linalg.LinAlgError

    monkeypatch.setattr(np.linalg, "solve", _raise_linalg_error)
    try:
        loss, _ = trainer.train_step(model, X[:6], y[:6], None)
    finally:
        monkeypatch.setattr(np.linalg, "solve", original_solve)

    assert np.isfinite(loss)


def test_hybrid_prepare_data_reshapes_1d():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(5, 2))
    y = X[:, 0] - X[:, 1]  # 1D
    Xp, yp = HybridTrainer._prepare_data(X, y)
    assert Xp.shape == X.shape
    assert yp.shape == (5, 1)


def test_sgd_trainer_with_cross_entropy_loss_on_classifier():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(20, 1))
    y = (X[:, 0] > 0).astype(int)
    clf = _make_classifier(n_inputs=1, n_classes=2)
    trainer = SGDTrainer(
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


def test_sgd_fit_raises_when_target_rows_mismatch():
    X = np.zeros((5, 2))
    y = np.zeros(4)  # fewer samples than X
    model = _make_regression_model(n_inputs=2)
    trainer = SGDTrainer(epochs=1)
    with pytest.raises(ValueError, match="Target array must have same number of rows as X"):
        trainer.fit(model, X, y)


def test_sgd_prepare_training_data_sets_loss_fn():
    trainer = SGDTrainer()
    model = _make_regression_model(n_inputs=1)
    X = np.linspace(-1, 1, 4).reshape(-1, 1)
    y = 0.5 * X[:, 0]
    Xt, yt = trainer._prepare_training_data(model, X, y)
    assert hasattr(trainer, "_loss_fn")
    assert isinstance(trainer._loss_fn, LossFunction)
    assert Xt.shape == (4, 1)
    assert yt.shape == (4, 1)
