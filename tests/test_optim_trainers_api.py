import numpy as np

from anfis_toolbox import ANFIS
from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.optim import AdamTrainer, HybridTrainer, RMSPropTrainer, SGDTrainer


def _make_regression_model(n_inputs: int = 2) -> ANFIS:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return ANFIS(input_mfs)


def test_sgd_train_step_and_init_state():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(10, 2))
    y = (0.5 * X[:, 0] - 0.25 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = SGDTrainer(learning_rate=0.01)
    state = trainer.init_state(model, X, y)
    assert state is None
    params_before = model.get_parameters()
    loss, state_after = trainer.train_step(model, X[:5], y[:5], state)
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
    state = trainer.init_state(model, X, y)
    assert isinstance(state, dict) and set(state.keys()) == {"params", "m", "v", "t"}
    t0 = state["t"]
    params_before = model.get_parameters()
    loss, state = trainer.train_step(model, X[:6], y[:6], state)
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
    state = trainer.init_state(model, X, y)
    assert isinstance(state, dict) and set(state.keys()) == {"params", "cache"}
    params_before = model.get_parameters()
    loss, state = trainer.train_step(model, X[:6], y[:6], state)
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
        losses = trainer.fit(model, X, y)
    finally:
        monkeypatch.setattr(np.linalg, "solve", original_solve)

    assert isinstance(losses, list) and len(losses) == 1 and np.isfinite(losses[0])


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
