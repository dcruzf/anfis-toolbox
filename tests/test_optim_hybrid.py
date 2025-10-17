import numpy as np

from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.model import ANFIS
from anfis_toolbox.optim import HybridAdamTrainer, HybridTrainer


def _make_regression_model(n_inputs: int = 1) -> ANFIS:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return ANFIS(input_mfs)


def test_hybrid_prepare_data_reshapes_targets():
    trainer = HybridTrainer(learning_rate=0.01, epochs=1, verbose=False)
    X = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([1.0, 2.0], dtype=float)

    X_prepared, y_prepared = trainer._prepare_training_data(None, X, y)
    assert X_prepared.shape == (2, 1)
    assert y_prepared.shape == (2, 1)

    X_val, y_val = trainer._prepare_validation_data(None, X, y)
    assert X_val.shape == (2, 1)
    assert y_val.shape == (2, 1)


def test_hybrid_train_step_uses_pseudoinverse_on_singular_system(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 1))
    y = (0.5 * X[:, 0]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=1)
    trainer = HybridTrainer(learning_rate=0.01, epochs=1, verbose=True)

    trainer._prepare_training_data(model, X, y)

    def _raise_lin_alg_error(a, b):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", _raise_lin_alg_error)
    pinv_calls: list[np.ndarray] = []
    original_pinv = np.linalg.pinv

    def _track_pinv(matrix):
        pinv_calls.append(matrix)
        return original_pinv(matrix)

    monkeypatch.setattr(np.linalg, "pinv", _track_pinv)

    loss, state = trainer.train_step(model, X, y, None)

    assert state is None
    assert np.isfinite(loss)
    assert pinv_calls

    val_loss = trainer.compute_loss(model, X, y)
    assert np.isfinite(val_loss)


def test_hybrid_adam_trainer_runs_and_reduces_loss():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 2))
    y = 1.2 * X[:, 0] - 0.8 * X[:, 1] + rng.normal(scale=0.1, size=30)
    model = _make_regression_model(n_inputs=2)
    trainer = HybridAdamTrainer(learning_rate=0.01, epochs=10, verbose=False)
    trainer._prepare_training_data(model, X, y)
    initial_loss = trainer.compute_loss(model, X, y)
    state = trainer.init_state(model, X, y)
    for _ in range(10):
        loss, state = trainer.train_step(model, X, y, state)
    final_loss = trainer.compute_loss(model, X, y)
    assert final_loss < initial_loss


def test_hybrid_adam_trainer_pseudoinverse_path(monkeypatch):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(6, 1))
    y = (0.4 * X[:, 0]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=1)
    trainer = HybridAdamTrainer(learning_rate=0.01, epochs=1, verbose=False)
    state = trainer.init_state(model, X, y)

    def _raise_lin_alg_error(a, b):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", _raise_lin_alg_error)
    original_pinv = np.linalg.pinv
    pinv_calls: list[np.ndarray] = []

    def _track_pinv(matrix):
        pinv_calls.append(matrix)
        return original_pinv(matrix)

    monkeypatch.setattr(np.linalg, "pinv", _track_pinv)

    loss, state = trainer.train_step(model, X, y, state)
    assert np.isfinite(loss)
    assert state["t"] >= 0
    assert pinv_calls


def test_hybrid_adam_prepare_data_reshapes_1d_targets():
    trainer = HybridAdamTrainer()
    X = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([0.5, -0.2], dtype=float)
    X_prep, y_prep = trainer._prepare_data(X, y)
    assert X_prep.shape == (2, 1)
    assert y_prep.shape == (2, 1)
