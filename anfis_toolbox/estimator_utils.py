"""Lightweight utilities for scikit-learn style estimators without external dependencies.

This module provides a minimal subset of the scikit-learn estimator contract so that
high-level ANFIS interfaces can expose familiar methods (`fit`, `predict`,
`get_params`, `set_params`, etc.) without requiring scikit-learn as a runtime
dependency. The helpers here intentionally implement only the pieces we need
and keep them Numpy-centric for portability.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    from sklearn.utils._tags import default_tags as _sklearn_default_tags

    _SKLEARN_TAGS_AVAILABLE = True
except Exception:  # pragma: no cover - sklearn not installed
    _sklearn_default_tags = None
    _SKLEARN_TAGS_AVAILABLE = False

__all__ = [
    "BaseEstimatorLike",
    "RegressorMixinLike",
    "ClassifierMixinLike",
    "FittedMixin",
    "RuleInspectorMixin",
    "NotFittedError",
    "check_is_fitted",
    "ensure_2d_array",
    "ensure_vector",
    "infer_feature_names",
    "format_estimator_repr",
]


class NotFittedError(RuntimeError):
    """Exception raised when an estimator is used before fitting."""


class BaseEstimatorLike:
    """Mixin implementing scikit-learn style parameter inspection.

    Parameters are assumed to live on the instance `__dict__` and be declared in
    `__init__`. This matches the common sklearn design pattern and enables
    cloning/grid-search like workflows without relying on sklearn itself.
    """

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters following sklearn conventions."""

        def clone_param(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: clone_param(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(clone_param(v) for v in value)
            # Primitive / numpy scalars
            if isinstance(value, (str, int, float, bool, type(None), np.generic)):
                return value
            # Fallback to deepcopy for custom objects
            return deepcopy(value)

        return {key: clone_param(value) for key, value in self.__dict__.items() if not key.endswith("_")}

    def set_params(self, **params: Any) -> BaseEstimatorLike:
        """Set estimator parameters and return self."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for {type(self).__name__}.")
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # scikit-learn compatibility hooks
    # ------------------------------------------------------------------
    def __sklearn_tags__(self) -> dict[str, Any] | Any:
        """Return estimator capability tags expected by scikit-learn."""
        merged: dict[str, Any] = {}
        more_tags = getattr(self, "_more_tags", None)
        if callable(more_tags):
            merged.update(more_tags())
        extra_tags = getattr(self, "_sklearn_tags", None)
        if isinstance(extra_tags, dict):
            merged.update(extra_tags)

        if _SKLEARN_TAGS_AVAILABLE:
            tags = _sklearn_default_tags(self)

            direct_updates = {}
            if "estimator_type" in merged:
                direct_updates["estimator_type"] = merged.pop("estimator_type")
            if "non_deterministic" in merged:
                direct_updates["non_deterministic"] = merged.pop("non_deterministic")
            if "requires_fit" in merged:
                direct_updates["requires_fit"] = merged.pop("requires_fit")
            if direct_updates:
                tags = replace(tags, **direct_updates)

            if "requires_y" in merged:
                required = bool(merged.pop("requires_y"))
                target_tags = replace(tags.target_tags, required=required)
                tags = replace(tags, target_tags=target_tags)

            # Remaining keys are not recognised by the public Tags API; ignore gracefully.
            return tags

        # Fallback lightweight representation when scikit-learn is not available.
        fallback = {
            "estimator_type": merged.pop("estimator_type", None),
            "non_deterministic": merged.pop("non_deterministic", False),
            "requires_y": merged.pop("requires_y", True),
            "requires_fit": merged.pop("requires_fit", True),
        }
        fallback.update(merged)
        return fallback

    def __sklearn_is_fitted__(self) -> bool:
        """Expose estimator fitted state to scikit-learn utilities."""
        return bool(getattr(self, "is_fitted_", False))


class FittedMixin:
    """Mixin providing utility to guard against using estimators pre-fit."""

    def _mark_fitted(self) -> None:
        self.is_fitted_ = True

    def _require_is_fitted(self, attributes: Iterable[str] | None = None) -> None:
        if not getattr(self, "is_fitted_", False):
            raise NotFittedError(f"{type(self).__name__} instance is not fitted yet.")
        if attributes:
            missing = [attr for attr in attributes if not hasattr(self, attr)]
            if missing:
                raise NotFittedError(
                    f"Estimator {type(self).__name__} is missing fitted attribute(s): {', '.join(missing)}"
                )


def check_is_fitted(estimator: FittedMixin, attributes: Iterable[str] | None = None) -> None:
    """Check if the estimator is fitted by verifying `is_fitted_` and optional attributes."""
    estimator._require_is_fitted(attributes)


def format_estimator_repr(
    name: str,
    config_pairs: Iterable[tuple[str, Any]],
    children: Iterable[tuple[str, str]],
    *,
    ascii_only: bool = False,
) -> str:
    r"""Compose a tree-style ``__repr__`` string for estimators.

    Parameters
    ----------
    name : str
        The display name for the estimator (typically ``type(self).__name__``).
    config_pairs : Iterable[tuple[str, Any]]
        Sequence of ``(key, value)`` pairs describing configuration values. ``None``
        values are omitted automatically. The values are rendered with ``repr`` to
        preserve type information (strings quoted, etc.).
    children : Iterable[tuple[str, str]]
        Sequence of ``(label, description)`` items describing child artefacts, such as
        fitted submodels or optimizers. Descriptions may contain newlines; they will
        be indented appropriately beneath the child label.
    ascii_only : bool, default=False
        When ``True`` use ASCII connectors (``|--``/``\--``) instead of box drawing
        characters. Automatically useful for environments that do not render Unicode
        well.

    Returns:
    -------
    str
        Multi-line string representation combining the header and child sections.
    """
    config_fragments = [f"{key}={value!r}" for key, value in config_pairs if value is not None]
    header = f"{name}({', '.join(config_fragments)})" if config_fragments else f"{name}()"

    child_list = list(children)
    if not child_list:
        return header

    branch_mid, branch_last, pad_mid, pad_last = (
        ("|--", "\\--", "|  ", "   ") if ascii_only else ("├─", "└─", "│ ", "  ")
    )

    lines = [header]
    total = len(child_list)
    for index, (label, description) in enumerate(child_list):
        is_last = index == total - 1
        branch = branch_last if is_last else branch_mid
        pad = pad_last if is_last else pad_mid
        desc_lines = str(description).splitlines() or [""]
        first = desc_lines[0]
        lines.append(f"{branch} {label}: {first}")
        if len(desc_lines) > 1:
            padding = f"{pad}    "
            for extra in desc_lines[1:]:
                lines.append(f"{padding}{extra}")

    return "\n".join(lines)


class RuleInspectorMixin(FittedMixin):
    """Mixin that exposes ANFIS rule descriptors for fitted estimators."""

    def get_rules(
        self,
        *,
        include_membership_functions: bool = False,
    ) -> list[tuple[int, ...]] | list[dict[str, Any]]:
        """Return the fuzzy rules learned by the estimator.

        Parameters
        ----------
        include_membership_functions : bool, default=False
            When ``False`` (default), return a list of tuples with the membership-function
            indices per input. When ``True``, return a list of dictionaries describing each
            rule with input names, membership-function indices, and their corresponding
            membership function instances (when available).

        Returns:
        -------
        list
            Rule definitions either as tuples of integers (default) or as dictionaries with a
            rich description if ``include_membership_functions`` is ``True``.
        """
        check_is_fitted(self, attributes=["rules_"])

        raw_rules = getattr(self, "rules_", None) or []
        rule_tuples = [tuple(rule) for rule in raw_rules]

        if not include_membership_functions:
            return rule_tuples

        model = getattr(self, "model_", None)
        if model is None:
            raise NotFittedError(f"{type(self).__name__} instance is not fitted yet.")

        membership_map = getattr(model, "membership_functions", {})
        feature_names = list(getattr(self, "feature_names_in_", []) or membership_map.keys())

        descriptors: list[dict[str, Any]] = []
        for rule_index, rule in enumerate(rule_tuples):
            antecedents: list[dict[str, Any]] = []
            for input_index, mf_index in enumerate(rule):
                if input_index < len(feature_names):
                    input_name = feature_names[input_index]
                else:  # Fallback to positional naming
                    input_name = f"x{input_index + 1}"

                mf_list = membership_map.get(input_name, [])
                membership_fn = None
                if 0 <= mf_index < len(mf_list):
                    membership_fn = mf_list[mf_index]

                antecedents.append(
                    {
                        "input": input_name,
                        "mf_index": int(mf_index),
                        "membership_function": membership_fn,
                    }
                )

            descriptors.append(
                {
                    "index": rule_index,
                    "rule": rule,
                    "antecedents": antecedents,
                }
            )

        return descriptors


class RegressorMixinLike:
    """Mixin implementing a default `score` method for regressors."""

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        """Return predicted targets for ``X``."""
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R^2 of the prediction."""
        y_true = np.asarray(y, dtype=float).reshape(-1)
        y_pred = np.asarray(self.predict(X), dtype=float).reshape(-1)
        if y_true.shape != y_pred.shape:
            raise ValueError("Predicted values have a different shape than y.")
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        y_mean = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - y_mean) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - ss_res / ss_tot


class ClassifierMixinLike:
    """Mixin implementing default `score` via simple accuracy."""

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        """Return predicted class labels for ``X``."""
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given test data and labels."""
        y_true = np.asarray(y)
        y_pred = np.asarray(self.predict(X))
        if y_true.shape != y_pred.shape:
            raise ValueError("Predicted values have a different shape than y.")
        if y_true.size == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))


@dataclass
class ValidationResult:
    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]


def ensure_2d_array(X: Any) -> tuple[np.ndarray, list[str]]:
    """Validate and convert input data to a 2D float64 numpy array."""
    if hasattr(X, "to_numpy"):
        values = X.to_numpy(dtype=float)
        names = getattr(X, "columns", None)
        feature_names = [str(col) for col in names] if names is not None else None
    else:
        values = np.asarray(X, dtype=float)
        feature_names = None

    if values.ndim != 2:
        raise ValueError("Input data must be 2-dimensional (n_samples, n_features).")
    if feature_names is None:
        feature_names = [f"x{i + 1}" for i in range(values.shape[1])]

    return values, feature_names


def ensure_vector(y: Any, *, allow_2d_column: bool = True) -> np.ndarray:
    """Validate and convert target data to a 1D numpy array."""
    array = np.asarray(y)
    if array.ndim == 2:
        if array.shape[1] == 1 and allow_2d_column:
            array = array.reshape(-1)
        else:
            raise ValueError("Target array must be 1-dimensional or a column vector.")
    elif array.ndim != 1:
        raise ValueError("Target array must be 1-dimensional.")
    return array


# Backwards compatibility aliases (deprecated; prefer the public variants above)
_ensure_2d_array = ensure_2d_array
_ensure_vector = ensure_vector


def infer_feature_names(X: Any) -> list[str]:
    """Return feature names inferred from the input data structure."""
    if hasattr(X, "columns"):
        return [str(col) for col in X.columns]
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("Expected 2D array-like input to infer feature names.")
    return [f"x{i + 1}" for i in range(X_arr.shape[1])]
