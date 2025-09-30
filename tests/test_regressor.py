import inspect

import numpy as np
import pytest

from anfis_toolbox import ANFISRegressor
from anfis_toolbox.builders import ANFISBuilder
from anfis_toolbox.estimator_utils import NotFittedError
from anfis_toolbox.membership import GaussianMF
from anfis_toolbox.optim import BaseTrainer, SGDTrainer


def _generate_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(60, 2))
    # Simple smooth target with mild noise
    y = 1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.2 * np.sin(np.pi * X[:, 0])
    y += rng.normal(scale=0.05, size=X.shape[0])
    return X, y


def test_anfis_regressor_fit_predict_and_metrics():
    X, y = _generate_dataset()
    reg = ANFISRegressor(
        n_mfs=3,
        mf_type="gaussian",
        optimizer="hybrid",
        epochs=5,
        learning_rate=0.05,
        random_state=42,
    )

    reg.fit(X, y)

    preds = reg.predict(X[:5])
    assert preds.shape == (5,)
    assert reg.training_history_ is not None
    metrics = reg.evaluate(X, y)
    assert {"mse", "rmse", "mae", "r2"}.issubset(metrics.keys())


def test_input_config_overrides_membership_counts():
    X, y = _generate_dataset()
    inputs_config = {
        "x1": {"mf_type": "triangular", "n_mfs": 4},
        1: {"range": (-1.5, 1.5), "n_mfs": 4, "mf_type": "gaussian"},
    }
    reg = ANFISRegressor(
        inputs_config=inputs_config,
        optimizer="hybrid",
        epochs=2,
        learning_rate=0.02,
    )
    reg.fit(X, y)

    assert reg.model_ is not None
    mf_counts = {name: len(mfs) for name, mfs in reg.model_.membership_functions.items()}
    assert mf_counts["x1"] == 4
    assert mf_counts["x2"] == 4

    # Check MF classes were overridden as requested
    x1_classes = {type(mf).__name__ for mf in reg.model_.membership_functions["x1"]}
    x2_classes = {type(mf).__name__ for mf in reg.model_.membership_functions["x2"]}
    assert x1_classes == {"TriangularMF"}
    assert x2_classes == {"GaussianMF"}


def test_get_set_params_roundtrip():
    reg = ANFISRegressor(
        n_mfs=5,
        optimizer="sgd",
        optimizer_params={"epochs": 10, "learning_rate": 0.01},
        shuffle=False,
    )
    params = reg.get_params()

    clone = ANFISRegressor(**params)
    assert clone.get_params()["n_mfs"] == 5
    assert clone.get_params()["optimizer"] == "sgd"
    assert clone.get_params()["optimizer_params"]["epochs"] == 10

    # Exercise set_params for coverage
    reg2 = ANFISRegressor()
    reg2.set_params(**params)
    assert reg2.optimizer == "sgd"
    assert reg2.optimizer_params["learning_rate"] == 0.01
    assert reg2.shuffle is False


def test_optimizer_instance_overrides_learning_rate_and_loss():
    X, y = _generate_dataset()
    trainer = SGDTrainer(learning_rate=0.1, epochs=1, loss=None)
    reg = ANFISRegressor(
        optimizer=trainer,
        learning_rate=0.03,
        epochs=2,
        loss="mse",
    )
    reg.fit(X, y)
    assert isinstance(reg.optimizer_, SGDTrainer)
    assert pytest.approx(reg.optimizer_.learning_rate, rel=1e-6) == 0.03
    assert reg.optimizer_.epochs == 2
    assert reg.optimizer_.loss == "mse"


def test_optimizer_class_resolves_and_receives_overrides():
    X, y = _generate_dataset()
    reg = ANFISRegressor(
        optimizer=SGDTrainer,
        learning_rate=0.02,
        epochs=3,
        loss="mse",
        shuffle=False,
    )
    reg.fit(X, y)
    assert isinstance(reg.optimizer_, SGDTrainer)
    assert reg.optimizer_.epochs == 3


def test_membership_list_configuration_and_predict_guard(capsys):
    X, y = _generate_dataset()
    mfs = [GaussianMF(-1.0, 0.4), GaussianMF(0.0, 0.4), GaussianMF(1.0, 0.4)]

    with pytest.raises(NotFittedError):
        # Predict before fit should trigger NotFittedError
        ANFISRegressor().predict([[0.0, 0.0]])

    reg = ANFISRegressor(
        inputs_config={"x1": mfs},
        optimizer="hybrid",
        epochs=2,
        learning_rate=0.02,
    )
    reg.fit(X, y)

    assert reg.model_ is not None
    assert reg.model_.membership_functions["x1"][0] is mfs[0]

    reg.evaluate(X, y, print_results=True)
    output = capsys.readouterr().out
    assert "ANFISRegressor evaluation" in output


def test_fit_validates_sample_alignment():
    X, y = _generate_dataset()
    reg = ANFISRegressor(optimizer="hybrid", epochs=1)
    with pytest.raises(ValueError, match="same number of samples"):
        reg.fit(X, y[:-1])


def test_predict_handles_1d_and_feature_checks():
    X, y = _generate_dataset()
    reg = ANFISRegressor(optimizer="hybrid", epochs=2)
    reg.fit(X, y)

    single = reg.predict(X[0])
    assert single.shape == (1,)

    with pytest.raises(ValueError):
        reg.predict(np.random.randn(5, 3))

    reg.n_features_in_ = None
    with pytest.raises(RuntimeError):
        reg.predict(X[0])


def test_inputs_config_alt_key_and_membership_instance():
    X, y = _generate_dataset()
    inputs_config = {
        "x1": GaussianMF(0.0, 0.3),
        "x2": {"mf_type": "bell", "n_mfs": 1, "range": (-1.5, 1.5)},
    }
    reg = ANFISRegressor(n_mfs=1, inputs_config=inputs_config, optimizer="hybrid", epochs=2)
    reg.fit(X, y)

    assert reg.model_ is not None
    assert len(reg.model_.membership_functions["x1"]) == 1
    assert type(reg.model_.membership_functions["x1"][0]).__name__ == "GaussianMF"
    assert type(reg.model_.membership_functions["x2"][0]).__name__ == "BellMF"


def test_membership_list_with_range_override():
    X, y = _generate_dataset()
    mfs = [GaussianMF(-1.0, 0.4), GaussianMF(0.0, 0.4), GaussianMF(1.0, 0.4)]
    inputs_config = {
        "x1": {"membership_functions": mfs, "range": (-2.0, 2.0)},
        1: {"n_mfs": 3, "mf_type": "triangular"},
    }
    reg = ANFISRegressor(inputs_config=inputs_config, optimizer="hybrid", epochs=2, learning_rate=0.02)
    reg.fit(X, y)


def test_optimizer_validation_errors():
    X, y = _generate_dataset()
    with pytest.raises(ValueError, match="Unknown optimizer"):
        ANFISRegressor(optimizer="does-not-exist").fit(X, y)

    with pytest.raises(TypeError):
        ANFISRegressor(optimizer=123).fit(X, y)


def test_optimizer_params_forwarded():
    X, y = _generate_dataset()
    reg = ANFISRegressor(
        optimizer="sgd",
        optimizer_params={"epochs": 2, "learning_rate": 0.05},
        shuffle=False,
    )
    reg.fit(X, y)
    assert isinstance(reg.optimizer_, SGDTrainer)
    assert reg.optimizer_.epochs == 2
    assert pytest.approx(reg.optimizer_.learning_rate, rel=1e-6) == 0.05


def test_inputs_config_mfs_alias_applies_memberships():
    X, y = _generate_dataset()
    mfs = [GaussianMF(-1.0, 0.4), GaussianMF(0.0, 0.4), GaussianMF(1.0, 0.4)]
    inputs_config = {
        "x1": {"mfs": mfs},
        1: {"n_mfs": 3, "mf_type": "triangular"},
    }
    reg = ANFISRegressor(inputs_config=inputs_config, optimizer="hybrid", epochs=2)
    reg.fit(X, y)
    assert reg.model_ is not None
    assert reg.model_.membership_functions["x1"][0] is mfs[0]


def test_custom_trainer_class_triggers_self_parameter_handling():
    class MinimalTrainer(BaseTrainer):
        def __init__(self, scale: float = 1.0):
            self.scale = scale

        def fit(self, model, X, y):
            return []

        def init_state(self, model, X, y):
            return None

        def train_step(self, model, Xb, yb, state):
            return 0.0, state

    X, y = _generate_dataset()
    reg = ANFISRegressor(optimizer=MinimalTrainer, optimizer_params={"scale": 2.0}, epochs=1)
    reg.fit(X, y)
    assert isinstance(reg.optimizer_, MinimalTrainer)
    assert reg.optimizer_.scale == 2.0


def test_regressor_collect_trainer_params_skips_self(monkeypatch):
    class DummyTrainer(BaseTrainer):
        def __init__(self, alpha=1, beta=2):
            self.alpha = alpha
            self.beta = beta

        def fit(self, model, X_fit, y_fit):
            return [0.0]

        def init_state(self, model, X_fit, y_fit):
            return None

        def train_step(self, model, Xb, yb, state):
            return 0.0, state

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

    monkeypatch.setattr("anfis_toolbox.regressor.inspect.signature", fake_signature)

    X, y = _generate_dataset(seed=21)
    reg = ANFISRegressor(
        optimizer=DummyTrainer,
        optimizer_params={"alpha": 5},
        epochs=1,
    )
    reg.fit(X, y)

    assert isinstance(reg.optimizer_, DummyTrainer)
    assert reg.optimizer_.alpha == 5
    assert reg.optimizer_.beta == 2


def test_regressor_apply_runtime_overrides_skips_verbose_when_none():
    class VerboseTrainer(BaseTrainer):
        def __init__(self):
            self.verbose = True

        def fit(self, model, X_fit, y_fit):
            return [0.0]

        def init_state(self, model, X_fit, y_fit):
            return None

        def train_step(self, model, Xb, yb, state):
            return 0.0, state

    trainer = VerboseTrainer()
    X, y = _generate_dataset(seed=22)
    reg = ANFISRegressor(optimizer=trainer, verbose=None, epochs=1)
    reg.fit(X, y)

    assert isinstance(reg.optimizer_, VerboseTrainer)
    assert reg.optimizer_ is not trainer
    assert reg.optimizer_.verbose is True


def test_invalid_input_spec_type_triggers_type_error():
    X, y = _generate_dataset()
    inputs_config = {"x1": 3.14}
    with pytest.raises(TypeError):
        ANFISRegressor(inputs_config=inputs_config, optimizer="hybrid", epochs=1).fit(X, y)


def test_inputs_config_alt_key_with_dataframe():
    class SimpleFrame:
        def __init__(self, data):
            self._data = np.asarray(data)
            self.columns = ["f1", "f2"]

        def to_numpy(self, dtype=float):
            return np.asarray(self._data, dtype=dtype)

    X, y = _generate_dataset()
    frame = SimpleFrame(X)
    inputs_config = {
        "x1": "triangular",
        "x2": {"mf_type": "bell", "n_mfs": 3},
    }
    reg = ANFISRegressor(n_mfs=3, inputs_config=inputs_config, optimizer="hybrid", epochs=2)
    reg.fit(frame, y)

    assert reg.model_ is not None
    assert type(reg.model_.membership_functions["f1"][0]).__name__ == "TriangularMF"
    assert type(reg.model_.membership_functions["f2"][0]).__name__ == "BellMF"


def test_inputs_config_integer_override():
    X, y = _generate_dataset()
    inputs_config = {0: 2, 1: 2}
    reg = ANFISRegressor(n_mfs=2, inputs_config=inputs_config, optimizer="hybrid", epochs=1)
    reg.fit(X, y)
    assert all(len(mfs) == 2 for mfs in reg.model_.membership_functions.values())


def test_init_none_defaults_to_grid_behavior():
    X, y = _generate_dataset()
    reg = ANFISRegressor(init=None, optimizer="hybrid", epochs=1)
    reg.fit(X, y)

    assert reg.input_specs_ is not None
    assert all(spec["init"] is None for spec in reg.input_specs_)


def test_init_random_matches_builder_layout():
    X, y = _generate_dataset()
    reg = ANFISRegressor(init="random", random_state=123, epochs=1)
    reg.fit(X, y)

    assert reg.input_specs_ is not None
    assert all(spec["init"] == "random" for spec in reg.input_specs_)

    builder = ANFISBuilder()
    builder.add_input_from_data("x1", X[:, 0], n_mfs=reg.n_mfs, mf_type=reg.mf_type, init="random", random_state=123)
    builder.add_input_from_data("x2", X[:, 1], n_mfs=reg.n_mfs, mf_type=reg.mf_type, init="random", random_state=123)

    centers_reg_x1 = np.array([mf.parameters["mean"] for mf in reg.model_.membership_functions["x1"]])
    centers_builder_x1 = np.array([mf.parameters["mean"] for mf in builder.input_mfs["x1"]])
    assert np.allclose(centers_reg_x1, centers_builder_x1, atol=1e-4)

    centers_reg_x2 = np.array([mf.parameters["mean"] for mf in reg.model_.membership_functions["x2"]])
    centers_builder_x2 = np.array([mf.parameters["mean"] for mf in builder.input_mfs["x2"]])
    assert np.allclose(centers_reg_x2, centers_builder_x2, atol=1e-4)


def test_invalid_init_strategy_raises_error():
    X, y = _generate_dataset()
    reg = ANFISRegressor(init="invalid", epochs=1)
    with pytest.raises(ValueError, match="Unknown init strategy"):
        reg.fit(X, y)
