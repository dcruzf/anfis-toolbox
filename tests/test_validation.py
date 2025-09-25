import numpy as np
import pytest

from anfis_toolbox import QuickANFIS
from anfis_toolbox.validation import ANFISMetrics


def make_data(n=60, d=2, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    y = X[:, 0] ** 2 + (X[:, 1] if d > 1 else 0)
    return X, y


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
    _ = model.fit(X, y, epochs=1, learning_rate=0.01, verbose=False)
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
    _ = model.fit(X, y, epochs=1, learning_rate=0.01, verbose=False)
    from anfis_toolbox.validation import quick_evaluate

    metrics = quick_evaluate(model, X, y, print_results=False)
    assert isinstance(metrics, dict) and "mse" in metrics
