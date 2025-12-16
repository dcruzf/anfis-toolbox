from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


def zeros_like_structure(params: dict[str, Any]) -> dict[str, Any]:
    """Create a zero-structure matching model.get_parameters() format.

    Returns a dict with:
      - 'consequent': np.zeros_like(params['consequent'])
      - 'membership': { name: [ {param_name: 0.0, ...} ] }
    """
    out: dict[str, Any] = {"consequent": np.zeros_like(params["consequent"]), "membership": {}}
    for name, mf_list in params["membership"].items():
        out["membership"][name] = []
        for mf_params in mf_list:
            out["membership"][name].append(dict.fromkeys(mf_params.keys(), 0.0))
    return out


def iterate_membership_params(
    params_dict: dict[str, Any],
    grads_dict: dict[str, Any] | None = None,
) -> Iterator[tuple[tuple[str, int, str], float, float | None]]:
    """Iterate over membership parameters with their gradients.

    See optim/parameter_utils.py for structure details.
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
    """Iterate over membership parameters with state (for momentum-based optimizers)."""
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
    name, i, key = path
    params_dict["membership"][name][i][key] = float(value)


def get_membership_param(
    params_dict: dict[str, Any],
    path: tuple[str, int, str],
) -> float:
    name, i, key = path
    return float(params_dict["membership"][name][i][key])


def flatten_membership_params(params_dict: dict[str, Any]) -> tuple[np.ndarray, list[tuple[str, int, str]]]:
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
    for idx, (name, i, key) in enumerate(paths):
        params_dict["membership"][name][i][key] = float(flat_array[idx])


__all__ = [
    "zeros_like_structure",
    "iterate_membership_params",
    "iterate_membership_params_with_state",
    "update_membership_param",
    "get_membership_param",
    "flatten_membership_params",
    "unflatten_membership_params",
]
