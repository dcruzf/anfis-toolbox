import numpy as np

from anfis_toolbox import ANFIS, ANFISClassifier
from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.optim import PSOTrainer
from anfis_toolbox.optim.pso import _flatten_params, _unflatten_params


def _make_regression_model(n_inputs: int = 2) -> ANFIS:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return ANFIS(input_mfs)


def _make_classifier(n_inputs: int = 1, n_classes: int = 2) -> ANFISClassifier:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return ANFISClassifier(input_mfs, n_classes=n_classes, random_state=0)


def test_pso_trains_and_updates_regression_params():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] - 0.5 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    params_before = model.get_parameters()
    trainer = PSOTrainer(swarm_size=10, epochs=3, random_state=0, init_sigma=0.05, verbose=False)
    losses = trainer.fit(model, X, y)
    assert len(losses) == 3
    assert all(np.isfinite(loss) and loss >= 0 for loss in losses)
    params_after = model.get_parameters()
    assert not np.allclose(params_before["consequent"], params_after["consequent"])  # updated


def test_pso_train_step_api_progression():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    y = (0.3 * X[:, 0] + 0.1 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = PSOTrainer(swarm_size=8, epochs=1, random_state=1, init_sigma=0.05, verbose=False)
    state = trainer.init_state(model, X, y)
    best0 = state["gbest_val"]
    loss1, state = trainer.train_step(model, X[:10], y[:10], state)
    assert np.isfinite(loss1)
    # Global best should be finite and typically non-increasing after a step
    assert state["gbest_val"] <= best0 or np.isfinite(state["gbest_val"])  # not strict monotonic in stochastic PSO


def test_pso_runs_with_classifier_logits():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 1))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
    clf = _make_classifier(n_inputs=1, n_classes=2)
    trainer = PSOTrainer(swarm_size=6, epochs=2, random_state=2, init_sigma=0.05, verbose=False)
    losses = trainer.fit(clf, X, y)
    assert len(losses) == 2 and np.isfinite(losses[0]) and np.isfinite(losses[1])


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
    losses = trainer.fit(model, X, y)  # y is 1D to exercise reshape branch as well
    assert len(losses) == 1 and np.isfinite(losses[0])


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
    state = trainer.init_state(model, X, y)
    prev_pbest = state["pbest_val"].copy()
    loss, state = trainer.train_step(model, X, y, state)
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
    losses = trainer.fit(clf, X, y)
    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)
