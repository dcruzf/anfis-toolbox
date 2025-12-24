"""Tests for parameter utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from anfis_toolbox.optim._utils import (
    flatten_membership_params,
    get_membership_param,
    iterate_membership_params,
    iterate_membership_params_with_state,
    unflatten_membership_params,
    update_membership_param,
)


@pytest.fixture
def sample_params():
    """Create a sample parameter dictionary."""
    return {
        "consequent": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "membership": {
            "bell": [
                {"center": 0.5, "width": 1.0},
                {"center": 1.5, "width": 2.0},
            ],
            "gaussian": [
                {"mean": 0.0, "sigma": 0.5},
            ],
        },
    }


@pytest.fixture
def sample_grads(sample_params):
    """Create gradient dictionary matching sample_params structure."""
    grads = {
        "consequent": np.array([[0.01, 0.02], [0.03, 0.04]]),
        "membership": {},
    }
    for name in sample_params["membership"].keys():
        grads["membership"][name] = []
        for mf_dict in sample_params["membership"][name]:
            grads["membership"][name].append(dict.fromkeys(mf_dict.keys(), 0.001))
    return grads


@pytest.fixture
def sample_state(sample_params):
    """Create state dictionary (for momentum, cache, etc)."""
    state = {
        "consequent": np.zeros_like(sample_params["consequent"]),
        "membership": {},
    }
    for name in sample_params["membership"].keys():
        state["membership"][name] = []
        for mf_dict in sample_params["membership"][name]:
            state["membership"][name].append(dict.fromkeys(mf_dict.keys(), 0.0))
    return state


def test_iterate_membership_params_without_grads(sample_params):
    """Test iterating over membership params without gradients."""
    results = list(iterate_membership_params(sample_params))

    assert len(results) == 6  # 2 bell MFs × 2 params each + 1 gaussian MF × 2 params

    # Check bell params
    assert results[0][0] == ("bell", 0, "center")
    assert results[0][1] == 0.5
    assert results[0][2] is None

    assert results[1][0] == ("bell", 0, "width")
    assert results[1][1] == 1.0

    assert results[2][0] == ("bell", 1, "center")
    assert results[2][1] == 1.5

    # Check gaussian params
    assert results[4][0] == ("gaussian", 0, "mean")
    assert results[4][1] == 0.0


def test_iterate_membership_params_with_grads(sample_params, sample_grads):
    """Test iterating over membership params with gradients."""
    results = list(iterate_membership_params(sample_params, sample_grads))

    assert len(results) == 6

    # First param should have gradient
    path, param_val, grad_val = results[0]
    assert path == ("bell", 0, "center")
    assert param_val == 0.5
    assert grad_val == 0.001


def test_iterate_membership_params_with_state(sample_params, sample_state, sample_grads):
    """Test iterating with state and gradients."""
    results = list(iterate_membership_params_with_state(sample_params, sample_state, sample_grads))

    assert len(results) == 6

    path, param_val, state_val, grad_val = results[0]
    assert path == ("bell", 0, "center")
    assert param_val == 0.5
    assert state_val == 0.0  # Initial state
    assert grad_val == 0.001


def test_iterate_membership_params_with_state_no_grads(sample_params, sample_state):
    """Test iterating with state but without gradients."""
    results = list(iterate_membership_params_with_state(sample_params, sample_state, grads_dict=None))

    assert len(results) == 6

    path, param_val, state_val, grad_val = results[0]
    assert path == ("bell", 0, "center")
    assert param_val == 0.5
    assert state_val == 0.0
    assert grad_val is None  # No gradients provided


def test_update_membership_param(sample_params):
    """Test updating a single parameter."""
    path = ("bell", 0, "center")
    new_value = 0.75

    update_membership_param(sample_params, path, new_value)

    assert sample_params["membership"]["bell"][0]["center"] == 0.75
    # Other params unchanged
    assert sample_params["membership"]["bell"][1]["center"] == 1.5


def test_get_membership_param(sample_params):
    """Test retrieving a single parameter."""
    path = ("bell", 1, "width")
    value = get_membership_param(sample_params, path)

    assert value == 2.0


def test_flatten_membership_params(sample_params):
    """Test flattening membership parameters to 1D."""
    flat, paths = flatten_membership_params(sample_params)

    expected_values = [0.5, 1.0, 1.5, 2.0, 0.0, 0.5]
    np.testing.assert_array_almost_equal(flat, expected_values)

    expected_paths = [
        ("bell", 0, "center"),
        ("bell", 0, "width"),
        ("bell", 1, "center"),
        ("bell", 1, "width"),
        ("gaussian", 0, "mean"),
        ("gaussian", 0, "sigma"),
    ]
    assert paths == expected_paths


def test_unflatten_membership_params(sample_params):
    """Test unflattening from 1D back to nested structure."""
    # First flatten
    flat, paths = flatten_membership_params(sample_params)

    # Modify the flat array
    flat_modified = flat * 2.0

    # Unflatten
    unflatten_membership_params(flat_modified, paths, sample_params)

    # Check values doubled
    assert sample_params["membership"]["bell"][0]["center"] == 1.0
    assert sample_params["membership"]["bell"][1]["width"] == 4.0
    assert sample_params["membership"]["gaussian"][0]["mean"] == 0.0


def test_flatten_unflatten_roundtrip(sample_params):
    """Test that flatten/unflatten preserves values."""

    # Get flat representation
    flat, paths = flatten_membership_params(sample_params)

    # Modify params
    sample_params["membership"]["bell"][0]["center"] = 999.0

    # Restore via unflatten
    unflatten_membership_params(flat, paths, sample_params)

    # Should match original
    assert sample_params["membership"]["bell"][0]["center"] == 0.5
    assert sample_params["membership"]["gaussian"][0]["mean"] == 0.0


def test_iterate_empty_membership():
    """Test iteration with empty membership dict."""
    params = {"consequent": np.array([1.0]), "membership": {}}
    results = list(iterate_membership_params(params))
    assert results == []


def test_iterator_modification_pattern(sample_params, sample_grads):
    """Test a realistic optimization pattern using the iterator."""
    learning_rate = 0.01

    for path, param_val, grad_val in iterate_membership_params(sample_params, sample_grads):
        new_val = param_val - learning_rate * grad_val
        update_membership_param(sample_params, path, new_val)

    # Check that updates were applied
    assert sample_params["membership"]["bell"][0]["center"] < 0.5
    assert sample_params["membership"]["gaussian"][0]["mean"] < 0.0


def test_unflatten_with_empty_paths():
    """Test unflattening with empty paths list."""
    params = {"consequent": np.array([1.0]), "membership": {}}
    flat = np.array([], dtype=float)
    paths: list[tuple[str, int, str]] = []

    # Should not raise
    unflatten_membership_params(flat, paths, params)
    # Membership should remain unchanged
    assert params["membership"] == {}


def test_unflatten_single_param():
    """Test unflattening a single parameter."""
    params = {
        "consequent": np.array([1.0]),
        "membership": {
            "bell": [{"center": 0.5}],
        },
    }
    flat = np.array([0.75])
    paths = [("bell", 0, "center")]

    unflatten_membership_params(flat, paths, params)

    assert params["membership"]["bell"][0]["center"] == 0.75


def test_unflatten_partial_update():
    """Test that unflatten only updates specified paths."""
    params = {
        "consequent": np.array([1.0]),
        "membership": {
            "bell": [
                {"center": 0.5, "width": 1.0},
                {"center": 1.5, "width": 2.0},
            ],
        },
    }
    # Only flatten/update first two params
    flat = np.array([0.6, 1.1])
    paths = [("bell", 0, "center"), ("bell", 0, "width")]

    unflatten_membership_params(flat, paths, params)

    # Updated values
    assert params["membership"]["bell"][0]["center"] == 0.6
    assert params["membership"]["bell"][0]["width"] == 1.1
    # Unchanged values
    assert params["membership"]["bell"][1]["center"] == 1.5
    assert params["membership"]["bell"][1]["width"] == 2.0


def test_unflatten_with_all_paths():
    """Test unflattening all parameters from a flatten roundtrip."""
    params = {
        "consequent": np.array([1.0]),
        "membership": {
            "bell": [
                {"center": 0.5, "width": 1.0},
                {"center": 1.5, "width": 2.0},
            ],
            "gaussian": [{"mean": 0.0}],
        },
    }

    flat_orig, paths = flatten_membership_params(params)
    # Scale by 3
    flat_scaled = flat_orig * 3.0
    unflatten_membership_params(flat_scaled, paths, params)

    # All values should be tripled
    assert params["membership"]["bell"][0]["center"] == 1.5
    assert params["membership"]["bell"][0]["width"] == 3.0
    assert params["membership"]["bell"][1]["center"] == 4.5
    assert params["membership"]["gaussian"][0]["mean"] == 0.0  # 0 * 3 = 0
