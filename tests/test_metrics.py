"""Tests for metrics utilities."""

import numpy as np
import pytest

from anfis_toolbox.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
    pearson_correlation,
    r2_score,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
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
