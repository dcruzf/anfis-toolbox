import numpy as np

from anfis_toolbox import ANFIS, ANFISClassifier
from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.optim import RMSPropTrainer


def _make_regression_model(n_inputs: int = 2) -> ANFIS:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return ANFIS(input_mfs)


def test_rmsprop_trains_full_batch_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (0.8 * X[:, 0] - 0.2 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    trainer = RMSPropTrainer(learning_rate=0.01, epochs=3, batch_size=None, shuffle=False, verbose=False)
    losses = trainer.fit(model, X, y)
    assert len(losses) == 3
    assert all(np.isfinite(loss) and loss >= 0 for loss in losses)


def test_rmsprop_trains_minibatch_regression_and_updates_params():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + 0.5 * X[:, 1]).reshape(-1, 1)
    model = _make_regression_model(n_inputs=2)
    params_before = model.get_parameters()
    trainer = RMSPropTrainer(learning_rate=0.005, epochs=2, batch_size=8, shuffle=True, verbose=False)
    losses = trainer.fit(model, X, y)
    assert len(losses) == 2
    params_after = model.get_parameters()
    # Consequent should differ due to RMSProp updates
    assert not np.allclose(params_before["consequent"], params_after["consequent"])


def _make_classifier(n_inputs: int = 1, n_classes: int = 2) -> ANFISClassifier:
    input_mfs = {}
    for i in range(n_inputs):
        input_mfs[f"x{i + 1}"] = [GaussianMF(mean=-1.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    return ANFISClassifier(input_mfs, n_classes=n_classes, random_state=0)


def test_rmsprop_with_classifier_does_not_error_on_forward_backward():
    # RMSProp trainer computes MSE by default; we just exercise mechanics with classifier logits
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 1))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)  # treat as regression target to check plumbing
    clf = _make_classifier(n_inputs=1, n_classes=2)
    trainer = RMSPropTrainer(learning_rate=0.005, epochs=1, batch_size=5, shuffle=False, verbose=False)
    losses = trainer.fit(clf, X, y)
    assert len(losses) == 1 and np.isfinite(losses[0])


def test_rmsprop_accepts_1d_target_and_reshapes():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(25, 2))
    # 1D target to exercise the internal reshape branch
    y = 0.6 * X[:, 0] + 0.3 * X[:, 1]
    model = _make_regression_model(n_inputs=2)
    trainer = RMSPropTrainer(learning_rate=0.01, epochs=1, batch_size=None, shuffle=False, verbose=False)
    losses = trainer.fit(model, X, y)
    assert len(losses) == 1 and np.isfinite(losses[0])
