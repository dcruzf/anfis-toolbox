"""Tests for metrics utilities."""

from types import SimpleNamespace

import numpy as np
import pytest

from anfis_toolbox.metrics import (
    ANFISMetrics,
    accuracy,
    classification_entropy,
    cross_entropy,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
    partition_coefficient,
    pearson_correlation,
    quick_evaluate,
    r2_score,
    root_mean_squared_error,
    softmax,
    symmetric_mean_absolute_percentage_error,
    xie_beni_index,
)


def test_mse_1d_arrays():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 2.0])
    # Squared errors: [0, 1, 1] -> mean = 2/3
    mse = mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, 2.0 / 3.0)


def test_mse_2d_arrays():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[0.0], [2.0], [4.0]])
    # Squared errors: [1, 0, 1] -> mean = 2/3
    mse = mean_squared_error(y_true, y_pred)
    assert np.isclose(mse, 2.0 / 3.0)


def test_mse_python_lists_and_casting():
    # Accepts lists, casts to float arrays
    mse = mean_squared_error([0, 1, 2], [1, 1, 2])
    assert np.isclose(mse, (1 + 0 + 0) / 3.0)


def test_mse_shape_mismatch_raises():
    # Rely on numpy broadcasting to raise on incompatible shapes
    with pytest.raises(ValueError):
        # (3,) and (2,) cannot be broadcast together
        mean_squared_error(np.array([1, 2, 3]), np.array([1, 2]))


def test_mae_1d_arrays():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 2.0])
    # Abs errors: [0,1,1] -> mean = 2/3
    mae = mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 2.0 / 3.0)


def test_mae_2d_arrays():
    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[0.0], [2.0], [4.0]])
    # Abs errors: [1,0,1] -> mean = 2/3
    mae = mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 2.0 / 3.0)


def test_mae_python_lists_and_casting():
    mae = mean_absolute_error([0, 1, 2], [1, 1, 2])
    assert np.isclose(mae, (1 + 0 + 0) / 3.0)


def test_mae_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mean_absolute_error(np.array([1, 2, 3]), np.array([1, 2]))


def test_rmse_consistency_with_mse():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.0, 1.0, 3.0, 6.0])
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(mse))


def test_rmse_2d_arrays():
    y_true = np.array([[1.0], [2.0]])
    y_pred = np.array([[2.0], [2.0]])
    # Errors: [1,0] -> MSE=0.5 -> RMSE=sqrt(0.5)
    rmse = root_mean_squared_error(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(0.5))


def test_mape_basic():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    # Percent errors: [10/100, 10/200, 30/300] = [0.1, 0.05, 0.1] -> mean=0.08333.. *100
    mape = mean_absolute_percentage_error(y_true, y_pred)
    assert np.isclose(mape, (0.1 + 0.05 + 0.1) / 3.0 * 100.0)


def test_mape_with_zeros_uses_epsilon():
    y_true = np.array([0.0, 0.0, 10.0])
    y_pred = np.array([0.0, 1.0, 8.0])
    # For zeros, denominator becomes epsilon; just check it's finite and non-negative
    mape = mean_absolute_percentage_error(y_true, y_pred)
    assert np.isfinite(mape) and mape >= 0.0


def test_mape_lists_and_casting():
    mape = mean_absolute_percentage_error([100, 200], [110, 190])
    assert np.isclose(mape, ((10 / 100) + (10 / 200)) / 2 * 100)


def test_mape_shape_mismatch_raises():
    with pytest.raises(ValueError):
        # Numpy will raise on incompatible shapes during subtraction/division
        mean_absolute_percentage_error(np.array([1, 2, 3]), np.array([1, 2]))


def test_mape_ignore_zero_targets_skips_zero_entries():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([10.0, 110.0])
    mape = mean_absolute_percentage_error(y_true, y_pred, ignore_zero_targets=True)
    # Only the non-zero target contributes: |100-110|/100 = 0.1 -> 10%
    assert np.isclose(mape, 10.0)


def test_mape_ignore_zero_targets_all_zero_returns_inf():
    y_true = np.zeros(3)
    y_pred = np.array([1.0, -1.0, 0.5])
    mape = mean_absolute_percentage_error(y_true, y_pred, ignore_zero_targets=True)
    assert np.isinf(mape)


def test_smape_basic():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    # SMAPE per element: 200*10/(100+110)=2000/210 ≈ 9.5238, 200*10/(200+190)=2000/390 ≈ 5.1282
    expected = (2000 / 210 + 2000 / 390) / 2
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    assert np.isclose(smape, expected)


def test_smape_with_zeros_safe():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 1.0])
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    # First term: 0, second term: 200*1/(0+1)=200
    assert np.isclose(smape, (0.0 + 200.0) / 2.0)


def test_smape_lists_and_casting():
    smape = symmetric_mean_absolute_percentage_error([100, 200], [110, 190])
    expected = (2000 / 210 + 2000 / 390) / 2
    assert np.isclose(smape, expected)


def test_smape_shape_mismatch_raises():
    with pytest.raises(ValueError):
        symmetric_mean_absolute_percentage_error(np.array([1, 2, 3]), np.array([1, 2]))


def test_r2_perfect_and_poor_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat_perfect = y.copy()
    y_hat_poor = np.array([4.0, 3.0, 2.0, 1.0])
    assert np.isclose(r2_score(y, y_hat_perfect), 1.0)
    # R2 can be negative for poor fits
    assert r2_score(y, y_hat_poor) < 0.0


def test_r2_constant_target_cases():
    y_const = np.array([5.0, 5.0, 5.0])
    # Perfect prediction on constant target -> 1.0
    assert np.isclose(r2_score(y_const, np.array([5.0, 5.0, 5.0])), 1.0)
    # Non-perfect prediction on constant target -> 0.0 by definition here
    assert np.isclose(r2_score(y_const, np.array([5.0, 5.0, 6.0])), 0.0)


def test_pearson_positive_negative_and_constant():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y_pos = np.array([10.0, 20.0, 30.0, 40.0])  # perfectly correlated
    y_neg = np.array([40.0, 30.0, 20.0, 10.0])  # perfectly anti-correlated
    y_const = np.array([5.0, 5.0, 5.0, 5.0])  # zero variance
    assert np.isclose(pearson_correlation(x, y_pos), 1.0)
    assert np.isclose(pearson_correlation(x, y_neg), -1.0)
    assert np.isclose(pearson_correlation(x, y_const), 0.0)


def test_pearson_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pearson_correlation(np.array([1, 2, 3]), np.array([1, 2]))


def test_msle_basic_and_lists():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 2.0])
    # Use direct definition with log1p
    expected = np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
    msle = mean_squared_logarithmic_error(y_true, y_pred)
    assert np.isclose(msle, expected)
    # Lists also work
    msle2 = mean_squared_logarithmic_error([0, 1, 2], [0, 1, 1])
    expected2 = np.mean((np.log1p(np.array([0, 1, 2])) - np.log1p(np.array([0, 1, 1]))) ** 2)
    assert np.isclose(msle2, expected2)


def test_msle_negative_inputs_raise():
    with pytest.raises(ValueError):
        mean_squared_logarithmic_error(np.array([-1.0, 0.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        mean_squared_logarithmic_error(np.array([0.0, 0.0]), np.array([0.0, -1.0]))


def test_partition_coefficient_invalid_ndim_and_empty():
    # Invalid ndim
    with pytest.raises(ValueError, match="U must be a 2D membership matrix"):
        partition_coefficient(np.array([0.5, 0.5]))
    # Empty returns 0.0
    U_empty = np.zeros((0, 3))
    assert np.isclose(partition_coefficient(U_empty), 0.0)


def test_classification_entropy_invalid_ndim_and_empty():
    # Invalid ndim
    with pytest.raises(ValueError, match="U must be a 2D membership matrix"):
        classification_entropy(np.array([0.5, 0.5]))
    # Empty returns 0.0
    U_empty = np.zeros((0, 2))
    assert np.isclose(classification_entropy(U_empty), 0.0)


def test_xb_invalid_dims_and_shape_mismatches_and_k_lt_2():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    U = np.array([[0.6, 0.4], [0.5, 0.5]])
    C = np.array([[0.0, 0.0], [1.0, 0.0]])

    # X 3D invalid
    with pytest.raises(ValueError, match="X must be 1D or 2D"):
        xie_beni_index(np.zeros((2, 2, 1)), U, C)
    # U invalid ndim
    with pytest.raises(ValueError, match="U must be a 2D membership matrix"):
        xie_beni_index(X, np.array([0.5, 0.5]), C)
    # C invalid ndim
    with pytest.raises(ValueError, match="C must be a 2D centers matrix"):
        xie_beni_index(X, U, np.array([0.0, 0.0]))
    # X and U sample mismatch
    with pytest.raises(ValueError, match="X and U must have the same number of samples"):
        xie_beni_index(X[:1], U, C)
    # C and X feature mismatch
    with pytest.raises(ValueError, match="C and X must have the same number of features"):
        xie_beni_index(X, U, np.array([[0.0], [1.0]]))
    # k < 2 -> inf
    assert np.isinf(xie_beni_index(X, U[:, :1], C[:1]))


def test_xb_denominator_epsilon_guard_and_1d_X_path():
    # Centers identical -> min inter-center dist = 0, epsilon path engaged
    X = np.array([[1.0, 0.0], [2.0, 0.0]])
    U = np.array([[0.7, 0.3], [0.4, 0.6]])
    C = np.array([[0.0, 0.0], [0.0, 0.0]])  # identical centers
    m = 2.0
    # Compute expected numerator
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    num = np.sum((U**m) * d2)
    epsilon = 1e-12
    expected = num / (X.shape[0] * epsilon)
    xb = xie_beni_index(X, U, C, m=m, epsilon=epsilon)
    assert np.isclose(xb, expected)

    # 1D X should be reshaped internally and produce finite XB
    X1d = np.array([0.0, 1.0, 2.0])
    U1d = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    C1d = np.array([[0.0], [2.0]])
    xb1d = xie_beni_index(X1d, U1d, C1d)
    assert np.isfinite(xb1d) and xb1d >= 0.0


def test_classification_metrics_returns_scores():
    y_true = np.array([0, 1, 1])
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
        ]
    )

    metrics = ANFISMetrics.classification_metrics(y_true, y_proba)

    assert np.isclose(metrics["accuracy"], accuracy(y_true, y_proba))
    assert np.isclose(metrics["log_loss"], log_loss(y_true, y_proba))


def test_classification_metrics_validates_inputs():
    with pytest.raises(ValueError, match="y_proba must be a 2D array"):
        ANFISMetrics.classification_metrics(np.array([0, 1]), np.array([0.8, 0.2]))

    with pytest.raises(ValueError, match="same number of samples"):
        ANFISMetrics.classification_metrics(np.array([0, 1, 1]), np.full((2, 2), 0.5))


def test_model_complexity_metrics_counts_parameters():
    membership_functions = {
        "x1": [SimpleNamespace(parameters=np.array([0.1, 0.2]))],
        "x2": [SimpleNamespace(parameters=np.array([0.3, 0.4, 0.5]))],
    }
    consequent_params = np.zeros((3, 2))

    model = SimpleNamespace(
        n_inputs=2,
        n_rules=3,
        membership_layer=SimpleNamespace(membership_functions=membership_functions),
        consequent_layer=SimpleNamespace(parameters=consequent_params),
    )

    stats = ANFISMetrics.model_complexity_metrics(model)
    assert stats == {
        "n_inputs": 2,
        "n_rules": 3,
        "n_premise_parameters": 5,  # 2 + 3 from membership parameters
        "n_consequent_parameters": consequent_params.size,
        "total_parameters": 5 + consequent_params.size,
    }


def test_quick_evaluate_prints_full_summary(capsys):
    class DummyModel:
        def __init__(self):
            self._pred = np.array([0.0, 0.5])

        def predict(self, X):
            assert np.allclose(X, np.array([[0.0], [1.0]]))
            return self._pred

    model = DummyModel()
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    result = quick_evaluate(model, X, y, print_results=True)
    captured = capsys.readouterr().out.strip().splitlines()

    expected = ANFISMetrics.regression_metrics(y, model.predict(X))
    assert result == expected
    assert captured[0] == "=" * 50
    assert captured[1] == "ANFIS Model Evaluation Results"
    assert captured[2] == "=" * 50
    assert captured[3].startswith("Mean Squared Error (MSE):")
    assert captured[4].startswith("Root Mean Squared Error:")
    assert captured[5].startswith("Mean Absolute Error (MAE):")
    assert captured[6].startswith("R-squared (R²):")
    assert captured[7].startswith("Mean Abs. Percentage Error:")
    assert captured[8].startswith("Maximum Error:")
    assert captured[9].startswith("Standard Deviation of Error:")
    assert captured[10] == "=" * 50


def test_quick_evaluate_without_prints_returns_metrics_silently(capsys):
    class DummyModel:
        def predict(self, X):
            return np.array([1.0, 2.0, 3.0])

    model = DummyModel()
    X = np.zeros((3, 1))
    y = np.array([1.0, 2.0, 3.0])

    result = quick_evaluate(model, X, y, print_results=False)
    output = capsys.readouterr()

    expected = ANFISMetrics.regression_metrics(y, model.predict(X))
    assert result == expected
    assert output.out == ""


def test_softmax_rows_sum_to_one_and_invariant_to_shift():
    z = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    p = softmax(z, axis=1)
    np.testing.assert_allclose(np.sum(p, axis=1), np.ones(p.shape[0]))
    # Shift invariance
    shift = 5.0
    p2 = softmax(z + shift, axis=1)
    np.testing.assert_allclose(p, p2)


def test_cross_entropy_with_int_labels_and_one_hot_and_empty():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    y_int = np.array([0, 1])
    y_oh = np.array([[1.0, 0.0], [0.0, 1.0]])
    ce_int = cross_entropy(y_int, logits)
    ce_oh = cross_entropy(y_oh, logits)
    np.testing.assert_allclose(ce_int, ce_oh)
    # Empty batch
    assert np.isclose(cross_entropy([], np.zeros((0, 2))), 0.0)


def test_cross_entropy_mismatch_errors():
    # Integer labels length mismatch with logits batch size
    logits = np.array([[1.0, -1.0], [0.5, 0.1]])  # n=2
    y_int_bad = np.array([0])  # length 1 != 2
    with pytest.raises(ValueError, match="y_true length must match logits batch size"):
        cross_entropy(y_int_bad, logits)

    # One-hot shape mismatch with logits
    y_oh_bad = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (2,3) vs logits (2,2)
    with pytest.raises(ValueError, match="shape must match logits"):
        cross_entropy(y_oh_bad, logits)


def test_log_loss_int_and_one_hot_and_mismatch_errors():
    P = np.array([[0.8, 0.2], [0.1, 0.9]])
    y_int = np.array([0, 1])
    y_oh = np.array([[1.0, 0.0], [0.0, 1.0]])
    ll_int = log_loss(y_int, P)
    ll_oh = log_loss(y_oh, P)
    np.testing.assert_allclose(ll_int, ll_oh)
    # Mismatch shapes
    with pytest.raises(ValueError):
        log_loss(np.array([[1.0, 0.0]]), P)


def test_accuracy_from_logits_probs_and_indices_and_length_mismatch():
    logits = np.array([[2.0, 0.0], [0.1, 0.2], [0.0, 3.0]])
    probs = softmax(logits, axis=1)
    y = np.array([0, 1, 1])
    acc1 = accuracy(y, logits)
    acc2 = accuracy(y, probs)
    acc3 = accuracy(y, np.array([0, 1, 1]))
    assert acc1 == acc2 == acc3
    with pytest.raises(ValueError):
        accuracy(np.array([0, 1]), logits)


def test_accuracy_with_one_hot_y_true():
    # One-hot y_true; y_pred as class indices
    y_oh = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    preds = np.array([0, 1, 1])
    assert np.isclose(accuracy(y_oh, preds), 1.0)
