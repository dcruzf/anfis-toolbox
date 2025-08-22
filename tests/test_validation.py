import numpy as np
import pytest

from anfis_toolbox import QuickANFIS
from anfis_toolbox.validation import ANFISMetrics, ANFISValidator


def make_data(n=60, d=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    y = X[:, 0] ** 2 + (X[:, 1] if d > 1 else 0)
    return X, y


def test_cross_validate_runs_and_returns_stats():
    X, y = make_data(n=40, d=2)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    validator = ANFISValidator(model)

    res = validator.cross_validate(X, y, cv=3, epochs=3, learning_rate=0.01, n_mfs=2, random_state=123)

    assert set(res.keys()) == {
        "mse_mean",
        "mse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
        "training_time_mean",
        "training_time_std",
    }
    # Basic sanity ranges
    assert np.isfinite(res["mse_mean"]) and res["mse_mean"] >= 0
    # R2 can be very negative on poor models; only upper-bound at 1 and check finite
    assert np.isfinite(res["r2_mean"]) and res["r2_mean"] <= 1.0


def test_cross_validate_backprop_branch():
    X, y = make_data(n=30, d=2)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    validator = ANFISValidator(model)
    res = validator.cross_validate(X, y, cv=2, epochs=1, learning_rate=0.01, n_mfs=2, method="backprop", random_state=0)
    assert "mse_mean" in res and "mae_mean" in res


def test_train_test_split_evaluate_shapes_and_metrics():
    X, y = make_data(n=50, d=2)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    validator = ANFISValidator(model)

    out = validator.train_test_split_evaluate(X, y, test_size=0.25, epochs=2, learning_rate=0.01, random_state=0)

    assert "train_metrics" in out and "test_metrics" in out
    assert "losses" in out and isinstance(out["losses"], list)
    tm = out["train_metrics"]
    assert set(tm.keys()) == {"mse", "mae", "r2"}


def test_train_test_split_evaluate_backprop_branch():
    X, y = make_data(n=40, d=2)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    validator = ANFISValidator(model)
    out = validator.train_test_split_evaluate(
        X, y, test_size=0.3, epochs=1, learning_rate=0.01, method="backprop", random_state=0
    )
    assert "final_loss" in out and out["final_loss"] is not None


def test_learning_curve_progresses_with_sizes():
    X, y = make_data(n=80, d=2)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    validator = ANFISValidator(model)

    res = validator.learning_curve(X, y, train_sizes=[0.2, 0.5], epochs=2, learning_rate=0.01, n_mfs=2, random_state=0)

    # learning_curve uses an 80/20 split first; sizes are fractions of the 80% subset
    n_lc = int(len(X) * 0.8)
    assert res["train_sizes"] == [int(n_lc * 0.2), int(n_lc * 0.5)]
    assert len(res["train_scores"]) == 2 and len(res["val_scores"]) == 2
    assert len(res["training_times"]) == 2


def test_learning_curve_backprop_and_default_sizes():
    X, y = make_data(n=60, d=2)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    validator = ANFISValidator(model)
    # default train_sizes path and backprop branch
    res = validator.learning_curve(X, y, epochs=1, learning_rate=0.01, method="backprop", n_mfs=2, random_state=0)
    assert len(res["train_sizes"]) == 5
    assert len(res["train_scores"]) == 5 and len(res["val_scores"]) == 5


def test_metrics_helper_is_consistent_with_module():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    m = ANFISMetrics.regression_metrics(y_true, y_pred)
    assert set(m.keys()) == {"mse", "rmse", "mae", "r2", "mape", "max_error", "std_error"}
    assert m["max_error"] == pytest.approx(np.max(np.abs(y_true - y_pred)))


def test_metrics_mape_zero_true_returns_inf():
    y_true = np.zeros(5)
    y_pred = np.ones(5)
    m = ANFISMetrics.regression_metrics(y_true, y_pred)
    assert m["mape"] == np.inf


def test_model_complexity_metrics_and_quick_evaluate_print(capsys):
    X, y = make_data(n=20, d=1)
    from anfis_toolbox import QuickANFIS

    model = QuickANFIS.for_regression(X, n_mfs=2)
    # complexity metrics
    from anfis_toolbox.validation import ANFISMetrics as _M

    cm = _M.model_complexity_metrics(model)
    assert cm["n_inputs"] == 1 and cm["n_rules"] == model.n_rules
    assert cm["total_parameters"] == cm["n_premise_parameters"] + cm["n_consequent_parameters"]
    # quick_evaluate print path
    _ = model.fit_hybrid(X, y, epochs=1, learning_rate=0.01, verbose=False)
    from anfis_toolbox.validation import quick_evaluate

    metrics = quick_evaluate(model, X, y, print_results=True)
    captured = capsys.readouterr()
    assert "ANFIS Model Evaluation Results" in captured.out
    assert "Mean Squared Error" in captured.out
    assert "R-squared" in captured.out
    assert "mse" in metrics and "rmse" in metrics


def test_quick_evaluate_no_print_branch():
    X, y = make_data(n=10, d=1)
    model = QuickANFIS.for_regression(X, n_mfs=2)
    _ = model.fit_hybrid(X, y, epochs=1, learning_rate=0.01, verbose=False)
    from anfis_toolbox.validation import quick_evaluate

    metrics = quick_evaluate(model, X, y, print_results=False)
    assert isinstance(metrics, dict) and "mse" in metrics
