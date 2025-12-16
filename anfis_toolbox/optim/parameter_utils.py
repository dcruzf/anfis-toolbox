"""Utilities for traversing and manipulating nested parameter structures.

This module provides clean abstractions over the ANFIS model's parameter hierarchy,
reducing the need for deeply nested loops and improving maintainability.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


def iterate_membership_params(
    params_dict: dict[str, Any],
    grads_dict: dict[str, Any] | None = None,
) -> Iterator[tuple[tuple[str, int, str], float, float | None]]:
    """Iterate over membership parameters with their gradients.

    Yields tuples of ((name, mf_index, param_key), param_value, grad_value) where:
        - name: Name of the membership function type (e.g., 'bell', 'gaussian')
        - mf_index: Index of the membership function in that group
        - param_key: Name of the parameter (e.g., 'center', 'width')
        - param_value: Current parameter value (float)
        - grad_value: Current gradient value (float) or None if grads_dict not provided

    This iterator hides the 3-level nesting and provides a clean interface for
    parameter updates in optimizers.

    Parameters:
        params_dict: Parameter dictionary with structure:
            {'consequent': np.ndarray, 'membership': {name: [{key: val, ...}, ...]}}
        grads_dict: Optional gradients dictionary with same structure as params_dict.

    Yields:
        Tuples enabling parameter updates without explicit nested loops.

    Examples:
        Update parameters with a simple loop:
        >>> for (name, i, key), param_val, grad_val in iterate_membership_params(params, grads):
        ...     new_val = param_val - learning_rate * grad_val
        ...     params['membership'][name][i][key] = new_val
    """
    for name in params_dict["membership"].keys():
        for i, mf_dict in enumerate(params_dict["membership"][name]):
            for key in mf_dict.keys():
                param_val = float(params_dict["membership"][name][i][key])
                grad_val = None
                if grads_dict is not None:
                    grad_val = float(grads_dict["membership"][name][i][key])
                yield (name, i, key), param_val, grad_val


def iterate_membership_params_with_state(
    params_dict: dict[str, Any],
    state_dict: dict[str, Any],
    grads_dict: dict[str, Any] | None = None,
) -> Iterator[tuple[tuple[str, int, str], float, float, float | None]]:
    """Iterate over membership parameters with state (for momentum-based optimizers).

    Yields tuples of ((name, mf_index, param_key), param_value, state_value, grad_value) where:
        - name, mf_index, param_key: Same as iterate_membership_params
        - param_value: Current parameter value (float)
        - state_value: State value (e.g., momentum, velocity, cache) (float)
        - grad_value: Current gradient value (float) or None

    Useful for Adam, RMSProp, and similar optimizers that maintain auxiliary state.

    Parameters:
        params_dict: Parameter dictionary with structure above.
        state_dict: State dictionary with same structure as params_dict.
        grads_dict: Optional gradients dictionary.

    Yields:
        Tuples enabling parameter updates with state tracking.
    """
    for name in params_dict["membership"].keys():
        for i, mf_dict in enumerate(params_dict["membership"][name]):
            for key in mf_dict.keys():
                param_val = float(params_dict["membership"][name][i][key])
                state_val = float(state_dict["membership"][name][i][key])
                grad_val = None
                if grads_dict is not None:
                    grad_val = float(grads_dict["membership"][name][i][key])
                yield (name, i, key), param_val, state_val, grad_val


def update_membership_param(
    params_dict: dict[str, Any],
    path: tuple[str, int, str],
    value: float,
) -> None:
    """Update a single membership parameter by path.

    Parameters:
        params_dict: Parameter dictionary to update.
        path: Tuple of (name, mf_index, param_key) identifying the parameter.
        value: New value to set.
    """
    name, i, key = path
    params_dict["membership"][name][i][key] = float(value)


def get_membership_param(
    params_dict: dict[str, Any],
    path: tuple[str, int, str],
) -> float:
    """Retrieve a single membership parameter by path.

    Parameters:
        params_dict: Parameter dictionary to query.
        path: Tuple of (name, mf_index, param_key) identifying the parameter.

    Returns:
        The parameter value as a float.
    """
    name, i, key = path
    return float(params_dict["membership"][name][i][key])


def flatten_membership_params(params_dict: dict[str, Any]) -> tuple[np.ndarray, list[tuple[str, int, str]]]:
    """Flatten membership parameters into a 1D array.

    Returns:
        Tuple of (flat_array, path_list) where:
        - flat_array: 1D numpy array of all membership parameter values in order
        - path_list: List of (name, mf_index, param_key) tuples in same order
    """
    paths: list[tuple[str, int, str]] = []
    values: list[float] = []
    for name in params_dict["membership"].keys():
        for i, mf_dict in enumerate(params_dict["membership"][name]):
            for key in mf_dict.keys():
                paths.append((name, i, key))
                values.append(float(params_dict["membership"][name][i][key]))
    return np.asarray(values, dtype=float), paths


def unflatten_membership_params(
    flat_array: np.ndarray,
    paths: list[tuple[str, int, str]],
    params_dict: dict[str, Any],
) -> None:
    """Unflatten membership parameters from a 1D array back into nested structure.

    Modifies params_dict in-place.

    Parameters:
        flat_array: 1D array of membership parameter values.
        paths: List of (name, mf_index, param_key) tuples in same order as flat_array.
        params_dict: Parameter dictionary to update.
    """
    for idx, (name, i, key) in enumerate(paths):
        params_dict["membership"][name][i][key] = float(flat_array[idx])


__all__ = [
    "iterate_membership_params",
    "iterate_membership_params_with_state",
    "update_membership_param",
    "get_membership_param",
    "flatten_membership_params",
    "unflatten_membership_params",
]
