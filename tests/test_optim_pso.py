import numpy as np
import pytest

from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.model import ANFIS, TSKANFISClassifier
from anfis_toolbox.optim import PSOTrainer
from anfis_toolbox.optim.pso import _flatten_params, _unflatten_params


def _make_regression_model(n_inputs: int = 2) -> ANFIS:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return ANFIS(input_mfs)


def _make_classifier(n_inputs: int = 1, n_classes: int = 2) -> TSKANFISClassifier:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return TSKANFISClassifier(input_mfs, n_classes=n_classes, random_state=0)


def _flatten_model_params(model: ANFIS) -> np.ndarray:
    theta, _ = _flatten_params(model.get_parameters())
    return theta


def test_pso_trains_and_updates_regression_params():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] - 0.5 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    params_before = model.get_parameters()
    trainer = PSOTrainer(swarm_size=10, epochs=3, random_state=0, init_sigma=0.05, verbose=False)
    history = trainer.fit(model, X, y)
    train_losses = history["train"]
    assert len(train_losses) == 3
    assert all(np.isfinite(loss) and loss >= 0 for loss in train_losses)
    params_after = model.get_parameters()
    assert not np.allclose(params_before["consequent"], params_after["consequent"])  # updated


def test_pso_train_step_api_progression():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    y = (0.3 * X[:, 0] + 0.1 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = PSOTrainer(swarm_size=8, epochs=1, random_state=1, init_sigma=0.05, verbose=False)
    params_before = _flatten_model_params(model)
    X_prepared, y_prepared = trainer._prepare_training_data(model, X, y)
    assert hasattr(trainer, "_loss_fn")
    state = trainer.init_state(model, X_prepared, y_prepared)
    # init_state should not mutate the caller's model parameters
    np.testing.assert_allclose(_flatten_model_params(model), params_before)
    best0 = state["gbest_val"]
    loss1, state = trainer.train_step(model, X_prepared[:10], y_prepared[:10], state)
    assert np.isfinite(loss1)
    # Global best should be finite and typically non-increasing after a step
    assert state["gbest_val"] <= best0 or np.isfinite(state["gbest_val"])  # not strict monotonic in stochastic PSO
    # After train_step the model should be set to the current global best parameters
    best_params = _unflatten_params(state["gbest_pos"], state["meta"], state["template"])
    theta_best, _ = _flatten_params(best_params)
    np.testing.assert_allclose(_flatten_model_params(model), theta_best)


def test_pso_init_state_preserves_model_parameters():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(12, 2))
    y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    params_before = _flatten_model_params(model)
    trainer = PSOTrainer(swarm_size=6, epochs=1, random_state=5, init_sigma=0.05, verbose=False)
    X_prepared, y_prepared = trainer._prepare_training_data(model, X, y)
    trainer.init_state(model, X_prepared, y_prepared)
    params_after = _flatten_model_params(model)
    np.testing.assert_allclose(params_before, params_after)


def test_pso_runs_with_classifier_logits():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 1))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    clf = _make_classifier(n_inputs=1, n_classes=2)
    trainer = PSOTrainer(swarm_size=6, epochs=2, random_state=2, init_sigma=0.05, verbose=False)
    history = trainer.fit(clf, X, y)
    train_losses = history["train"]
    assert len(train_losses) == 2 and np.isfinite(train_losses[0]) and np.isfinite(train_losses[1])


def test_pso_flatten_handles_no_membership():
    # Construct a minimal params dict with no membership parameters
    params = {"consequent": np.zeros((2, 3)), "membership": {}}
    theta, meta = _flatten_params(params)
    assert theta.shape[0] == 6  # only consequent flattened
    assert meta["membership_info"] == []
    # Round-trip via unflatten should preserve consequent and empty membership
    out = _unflatten_params(theta, meta, params)
    assert np.allclose(out["consequent"], params["consequent"]) and out["membership"] == {}


def test_pso_fit_applies_velocity_and_position_clamps():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 2))
    y = (X[:, 0] - X[:, 1]).reshape(-1)
    model = _make_regression_model(n_inputs=2)
    # Tight clamps to force clipping; 1 epoch is enough to hit the clamp lines
    trainer = PSOTrainer(
        swarm_size=6,
        epochs=1,
        random_state=3,
        init_sigma=0.2,
        clamp_velocity=(-0.01, 0.01),
        clamp_position=(-0.1, 0.1),
        verbose=False,
    )
    history = trainer.fit(model, X, y)  # y is 1D to exercise reshape branch as well
    assert len(history["train"]) == 1 and np.isfinite(history["train"][0])


def test_pso_train_step_with_clamps_and_no_improvement_path():
    # Configure PSO so particles don't move: zero inertia and coefficients
    rng = np.random.default_rng(4)
    X = rng.normal(size=(16, 2))
    y = (0.2 * X[:, 0]).astype(float)  # 1D target to hit reshape in train_step
    model = _make_regression_model(n_inputs=2)
    trainer = PSOTrainer(
        swarm_size=5,
        epochs=1,
        random_state=4,
        init_sigma=0.05,
        inertia=0.0,
        cognitive=0.0,
        social=0.0,
        # Use extremely wide clamps to execute the clamp branches without changing values
        clamp_velocity=(-1e9, 1e9),
        clamp_position=(-1e9, 1e9),
        verbose=False,
    )
    X_prepared, y_prepared = trainer._prepare_training_data(model, X, y)
    state = trainer.init_state(model, X_prepared, y_prepared)
    prev_pbest = state["pbest_val"].copy()
    loss, state = trainer.train_step(model, X_prepared, y_prepared, state)
    # No movement implies no improvement; bests stay the same, loss finite
    assert np.isfinite(loss)
    assert np.allclose(state["pbest_val"], prev_pbest)


def test_pso_classifier_with_cross_entropy_loss():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(18, 1))
    y = (X[:, 0] > 0).astype(int)
    clf = _make_classifier(n_inputs=1, n_classes=2)
    trainer = PSOTrainer(
        swarm_size=8,
        epochs=2,
        random_state=7,
        init_sigma=0.05,
        verbose=False,
        loss="cross_entropy",
    )
    history = trainer.fit(clf, X, y)
    assert len(history["train"]) == 2
    assert all(np.isfinite(loss) for loss in history["train"])


def test_pso_fit_raises_when_target_rows_mismatch():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(12, 2))
    y = rng.normal(size=(6, 1))
    model = _make_regression_model(n_inputs=2)
    trainer = PSOTrainer(swarm_size=5, epochs=1, random_state=8, init_sigma=0.05, verbose=False)

    with pytest.raises(ValueError, match="Target array must have same number of rows as X"):
        trainer.fit(model, X, y)


def test_pso_prepare_validation_data_matches_training_batch():
    trainer = PSOTrainer(swarm_size=4, epochs=1, random_state=0, verbose=False)
    model = _make_regression_model(n_inputs=1)
    X = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)

    X_train, y_train = trainer._prepare_training_data(model, X, y)
    X_val, y_val = trainer._prepare_validation_data(model, X, y)

    np.testing.assert_array_equal(X_val, X_train)
    np.testing.assert_array_equal(y_val, y_train)


def test_pso_compute_loss_and_ensure_loss_fn_initializes():
    trainer = PSOTrainer(swarm_size=4, epochs=1, random_state=1, verbose=False)
    model = _make_regression_model(n_inputs=1)
    X = np.array([[0.5], [-0.5]], dtype=float)
    y = (0.3 * X[:, 0]).reshape(-1, 1)

    loss_fn = trainer._ensure_loss_fn()
    assert loss_fn is trainer._loss_fn

    trainer._prepare_training_data(model, X, y)
    loss = trainer.compute_loss(model, X, y)
    assert np.isfinite(loss)
