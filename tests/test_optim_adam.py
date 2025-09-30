from copy import deepcopy

import numpy as np
import pytest

from anfis_toolbox import ANFIS, TSKANFISClassifier
from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.optim import AdamTrainer


def _make_regression_model(n_inputs: int = 2) -> ANFIS:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return ANFIS(input_mfs)


def test_adam_trains_full_batch_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (0.8 * X[:, 0] - 0.2 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = AdamTrainer(learning_rate=0.01, epochs=3, batch_size=None, shuffle=False, verbose=False)
    losses = trainer.fit(model, X, y)
    assert len(losses) == 3
    assert all(np.isfinite(loss) and loss >= 0 for loss in losses)


def test_adam_trains_minibatch_regression_and_updates_params():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + 0.5 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    params_before = model.get_parameters()
    trainer = AdamTrainer(learning_rate=0.005, epochs=2, batch_size=8, shuffle=True, verbose=False)
    losses = trainer.fit(model, X, y)
    assert len(losses) == 2
    params_after = model.get_parameters()
    # Consequent should differ due to Adam updates
    assert not np.allclose(params_before["consequent"], params_after["consequent"])


def _make_classifier(n_inputs: int = 1, n_classes: int = 2) -> TSKANFISClassifier:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return TSKANFISClassifier(input_mfs, n_classes=n_classes, random_state=0)


def test_adam_with_classifier_does_not_error_on_forward_backward():
    # Adam trainer computes MSE by default; we just exercise mechanics with classifier logits
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 1))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)  # treat as regression target to check plumbing
    clf = _make_classifier(n_inputs=1, n_classes=2)
    trainer = AdamTrainer(learning_rate=0.005, epochs=1, batch_size=5, shuffle=False, verbose=False)
    losses = trainer.fit(clf, X, y)
    assert len(losses) == 1 and np.isfinite(losses[0])


def test_adam_accepts_1d_target_and_reshapes():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 2))
    # 1D target to exercise the internal reshape branch
    y = 0.7 * X[:, 0] - 0.1 * X[:, 1]
    model = _make_regression_model(n_inputs=2)
    trainer = AdamTrainer(learning_rate=0.01, epochs=1, batch_size=None, shuffle=False, verbose=False)
    losses = trainer.fit(model, X, y)
    assert len(losses) == 1 and np.isfinite(losses[0])


def test_adam_classifier_with_cross_entropy_loss():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(24, 1))
    y = (X[:, 0] > 0).astype(int)
    clf = _make_classifier(n_inputs=1, n_classes=2)
    trainer = AdamTrainer(
        learning_rate=0.01,
        epochs=2,
        batch_size=None,
        shuffle=False,
        verbose=False,
        loss="cross_entropy",
    )
    losses = trainer.fit(clf, X, y)
    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)


def test_adam_fit_raises_when_target_rows_mismatch():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=(5, 1))
    model = _make_regression_model(n_inputs=2)
    trainer = AdamTrainer(learning_rate=0.01, epochs=1, batch_size=None, shuffle=False, verbose=False)

    with pytest.raises(ValueError, match="Target array must have same number of rows as X"):
        trainer.fit(model, X, y)


def test_adam_train_step_lazy_initializes_loss_and_updates_membership():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(12, 2))
    y = (0.3 * X[:, 0] - 0.4 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = AdamTrainer(learning_rate=0.01, epochs=1, batch_size=4, shuffle=False, verbose=False)

    state = trainer.init_state(model, X, y)
    assert not hasattr(trainer, "_loss_fn")
    membership_before = deepcopy(state["params"]["membership"])

    loss, updated_state = trainer.train_step(model, X[:4], y[:4], state)

    assert updated_state is state
    assert np.isfinite(loss)
    assert hasattr(trainer, "_loss_fn")

    membership_after = state["params"]["membership"]
    assert any(
        not np.isclose(membership_before[name][i][key], membership_after[name][i][key])
        for name in membership_after
        for i in range(len(membership_after[name]))
        for key in membership_after[name][i]
    )

    loss2, state_again = trainer.train_step(model, X[4:8], y[4:8], state)
    assert state_again is state
    assert np.isfinite(loss2)
