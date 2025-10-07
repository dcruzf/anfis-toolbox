"""High-level regression estimator facade for ANFIS.

The :class:`ANFISRegressor` provides a scikit-learn style interface that wires
up membership-function generation, model construction, and optimizer selection
at instantiation time. It reuses the low-level :mod:`anfis_toolbox` components
under the hood without introducing an external dependency on scikit-learn.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from .builders import ANFISBuilder
from .estimator_utils import (
    BaseEstimatorLike,
    FittedMixin,
    RegressorMixinLike,
    _ensure_2d_array,
    _ensure_vector,
    check_is_fitted,
)
from .losses import LossFunction
from .membership import MembershipFunction
from .metrics import ANFISMetrics
from .model import ANFIS as LowLevelANFIS
from .optim import (
    AdamTrainer,
    BaseTrainer,
    HybridTrainer,
    PSOTrainer,
    RMSPropTrainer,
    SGDTrainer,
)
from .optim.base import TrainingHistory

TRAINER_REGISTRY: dict[str, type[BaseTrainer]] = {
    "hybrid": HybridTrainer,
    "sgd": SGDTrainer,
    "adam": AdamTrainer,
    "rmsprop": RMSPropTrainer,
    "pso": PSOTrainer,
}


class ANFISRegressor(BaseEstimatorLike, FittedMixin, RegressorMixinLike):
    """Adaptive Neuro-Fuzzy regressor with a scikit-learn style API.

    Parameters
    ----------
    n_mfs : int, default=3
        Default number of membership functions per input.
    mf_type : str, default="gaussian"
        Default membership function family (see :class:`ANFISBuilder`).
    init : {"grid", "fcm", "random", None}, default="grid"
        Strategy used when inferring membership functions from data. ``None``
        falls back to ``"grid"``.
    overlap : float, default=0.5
        Controls overlap when generating membership functions via the builder.
    margin : float, default=0.10
        Margin added around observed data ranges during grid initialization.
    inputs_config : Mapping, optional
        Per-input overrides. Keys may be feature names (when ``X`` is a
        :class:`pandas.DataFrame`) or integer indices. Values may be:

        * ``dict`` with keys among ``{"n_mfs", "mf_type", "init", "overlap",
          "margin", "range", "membership_functions", "mfs"}``.
        * A list/tuple of :class:`MembershipFunction` instances for full control.
        * ``None`` for defaults.
    random_state : int, optional
        Random state forwarded to FCM-based initialization and any stochastic
        optimizers.
    optimizer : str, BaseTrainer, type[BaseTrainer], or None, default="hybrid"
        Trainer identifier or instance used for fitting. Strings map to entries
        in :data:`TRAINER_REGISTRY`. ``None`` defaults to "hybrid".
    optimizer_params : Mapping, optional
        Additional keyword arguments forwarded to the trainer constructor.
    learning_rate, epochs, batch_size, shuffle, verbose : optional scalars
        Common trainer hyper-parameters provided for convenience. When the
        selected trainer supports the parameter it is included automatically.
    loss : str or LossFunction, optional
        Custom loss forwarded to trainers that expose a ``loss`` parameter.
    rules : Sequence[Sequence[int]] | None, optional
        Explicit fuzzy rule indices to use instead of the full Cartesian product. Each
        rule lists the membership-function index per input. ``None`` keeps the default
        exhaustive rule set.
    """

    def __init__(
        self,
        *,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        init: str | None = "grid",
        overlap: float = 0.5,
        margin: float = 0.10,
        inputs_config: Mapping[Any, Any] | None = None,
        random_state: int | None = None,
        optimizer: str | BaseTrainer | type[BaseTrainer] | None = "hybrid",
        optimizer_params: Mapping[str, Any] | None = None,
        learning_rate: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        shuffle: bool | None = None,
        verbose: bool = True,
        loss: LossFunction | str | None = None,
        rules: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Initialize the ANFIS regressor."""
        self.n_mfs = int(n_mfs)
        self.mf_type = str(mf_type)
        self.init = None if init is None else str(init)
        self.overlap = float(overlap)
        self.margin = float(margin)
        self.inputs_config = dict(inputs_config) if inputs_config is not None else None
        self.random_state = random_state
        self.optimizer = optimizer
        self.optimizer_params = dict(optimizer_params) if optimizer_params is not None else None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.loss = loss
        self.rules = None if rules is None else tuple(tuple(int(idx) for idx in rule) for rule in rules)

        # Fitted attributes (initialised later)
        self.model_: LowLevelANFIS | None = None
        self.optimizer_: BaseTrainer | None = None
        self.feature_names_in_: list[str] | None = None
        self.n_features_in_: int | None = None
        self.training_history_: TrainingHistory | None = None
        self.input_specs_: list[dict[str, Any]] | None = None
        self.rules_: list[tuple[int, ...]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        *,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        validation_frequency: int = 1,
        **fit_params: Any,
    ):
        """Fit the ANFIS regressor on data.

        Parameters
        ----------
        X, y : array-like
            Training data and targets.
        validation_data : tuple[np.ndarray, np.ndarray], optional
            Optional validation split forwarded to the underlying trainer.
        validation_frequency : int, default=1
            Evaluate validation loss every N epochs when validation data is provided.
        **fit_params : Any
            Additional keyword arguments forwarded to the trainer ``fit`` method.
        """
        X_arr, feature_names = _ensure_2d_array(X)
        y_vec = _ensure_vector(y)
        if X_arr.shape[0] != y_vec.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        self.feature_names_in_ = feature_names
        self.n_features_in_ = X_arr.shape[1]
        self.input_specs_ = self._resolve_input_specs(feature_names)

        self.model_ = self._build_model(X_arr, feature_names)
        trainer = self._instantiate_trainer()
        self.optimizer_ = trainer
        trainer_kwargs: dict[str, Any] = dict(fit_params)
        if validation_data is not None:
            trainer_kwargs.setdefault("validation_data", validation_data)
        if validation_data is not None or validation_frequency != 1:
            trainer_kwargs.setdefault("validation_frequency", validation_frequency)

        history = trainer.fit(self.model_, X_arr, y_vec, **trainer_kwargs)
        if not isinstance(history, dict):
            raise TypeError("Trainer.fit must return a TrainingHistory dictionary")
        self.training_history_ = history
        self.rules_ = self.model_.rules

        self._mark_fitted()
        return self

    def predict(self, X):
        """Predict regression targets for input samples."""
        check_is_fitted(self, attributes=["model_"])
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            feature_names = self.feature_names_in_ or [f"x{i + 1}" for i in range(X_arr.shape[1])]
        else:
            X_arr, feature_names = _ensure_2d_array(X)

        if self.n_features_in_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Feature mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}.")

        preds = self.model_.predict(X_arr)  # type: ignore[operator]
        return np.asarray(preds, dtype=float).reshape(-1)

    def evaluate(self, X, y, *, return_dict: bool = True, print_results: bool = False):
        """Evaluate predictions against targets using :class:`ANFISMetrics`."""
        check_is_fitted(self, attributes=["model_"])
        X_arr, _ = _ensure_2d_array(X)
        y_vec = _ensure_vector(y)
        preds = self.predict(X_arr)
        metrics = ANFISMetrics.regression_metrics(y_vec, preds)
        if print_results:
            quick = [
                ("MSE", metrics["mse"]),
                ("RMSE", metrics["rmse"]),
                ("MAE", metrics["mae"]),
                ("R2", metrics["r2"]),
            ]
            print("ANFISRegressor evaluation:")  # noqa: T201
            for name, value in quick:
                print(f"  {name:>6}: {value:.6f}")  # noqa: T201
        return metrics if return_dict else None

    def get_rules(self) -> tuple[tuple[int, ...], ...]:
        """Return the fuzzy rule index combinations of the fitted model."""
        check_is_fitted(self, attributes=["rules_"])
        if not self.rules_:
            return ()
        return tuple(tuple(rule) for rule in self.rules_)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_input_specs(self, feature_names: list[str]) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for idx, name in enumerate(feature_names):
            spec = self._fetch_input_config(name, idx)
            resolved.append(self._normalize_input_spec(spec))
        return resolved

    def _fetch_input_config(self, name: str, index: int):
        if self.inputs_config is None:
            return None
        if name in self.inputs_config:
            return self.inputs_config[name]
        if index in self.inputs_config:
            return self.inputs_config[index]
        alt_key = f"x{index + 1}"
        if alt_key in self.inputs_config:
            return self.inputs_config[alt_key]
        return None

    def _normalize_input_spec(self, spec) -> dict[str, Any]:
        config: dict[str, Any] = {
            "n_mfs": self.n_mfs,
            "mf_type": self.mf_type,
            "init": self.init,
            "overlap": self.overlap,
            "margin": self.margin,
            "range": None,
            "membership_functions": None,
        }
        if spec is None:
            return config
        if isinstance(spec, (list, tuple)) and all(isinstance(mf, MembershipFunction) for mf in spec):
            config["membership_functions"] = list(spec)
            return config
        if isinstance(spec, MembershipFunction):
            config["membership_functions"] = [spec]
            return config
        if isinstance(spec, dict):
            if "mfs" in spec and "membership_functions" not in spec:
                spec = {**spec, "membership_functions": spec["mfs"]}
            for key in ("n_mfs", "mf_type", "init", "overlap", "margin", "range", "membership_functions"):
                if key in spec and (spec[key] is not None or key == "init"):
                    config[key] = spec[key]
            return config
        if isinstance(spec, str):
            config["mf_type"] = spec
            return config
        if isinstance(spec, int):
            config["n_mfs"] = int(spec)
            return config
        raise TypeError(f"Unsupported input configuration type: {type(spec)!r}")

    def _build_model(self, X: np.ndarray, feature_names: list[str]) -> LowLevelANFIS:
        builder = ANFISBuilder()
        for idx, name in enumerate(feature_names):
            column = X[:, idx]
            spec = self.input_specs_[idx]
            mf_list = spec.get("membership_functions")
            range_override = spec.get("range")
            if mf_list is not None:
                builder.input_mfs[name] = list(mf_list)
                if range_override is not None:
                    builder.input_ranges[name] = tuple(float(v) for v in range_override)
                else:
                    builder.input_ranges[name] = (float(np.min(column)), float(np.max(column)))
                continue
            if range_override is not None:
                rmin, rmax = range_override
                builder.add_input(
                    name,
                    float(rmin),
                    float(rmax),
                    int(spec["n_mfs"]),
                    str(spec["mf_type"]),
                    overlap=float(spec["overlap"]),
                )
            else:
                init_strategy = spec.get("init")
                init_arg = None if init_strategy is None else str(init_strategy)
                builder.add_input_from_data(
                    name,
                    column,
                    n_mfs=int(spec["n_mfs"]),
                    mf_type=str(spec["mf_type"]),
                    overlap=float(spec["overlap"]),
                    margin=float(spec["margin"]),
                    init=init_arg,
                    random_state=self.random_state,
                )
        builder.set_rules(self.rules)
        return builder.build()

    def _instantiate_trainer(self) -> BaseTrainer:
        optimizer = self.optimizer if self.optimizer is not None else "hybrid"
        if isinstance(optimizer, BaseTrainer):
            trainer = deepcopy(optimizer)
            self._apply_runtime_overrides(trainer)
            return trainer
        if inspect.isclass(optimizer) and issubclass(optimizer, BaseTrainer):
            params = self._collect_trainer_params(optimizer)
            return optimizer(**params)
        if isinstance(optimizer, str):
            key = optimizer.lower()
            if key not in TRAINER_REGISTRY:
                supported = ", ".join(sorted(TRAINER_REGISTRY.keys()))
                raise ValueError(f"Unknown optimizer '{optimizer}'. Supported: {supported}")
            trainer_cls = TRAINER_REGISTRY[key]
            params = self._collect_trainer_params(trainer_cls)
            return trainer_cls(**params)
        raise TypeError("optimizer must be a string identifier, BaseTrainer instance, or BaseTrainer subclass")

    def _collect_trainer_params(self, trainer_cls: type[BaseTrainer]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.optimizer_params is not None:
            params.update(self.optimizer_params)

        overrides = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "verbose": self.verbose,
            "loss": self.loss,
        }
        for key, value in overrides.items():
            if value is not None and key not in params:
                params[key] = value
        # Ensure boolean defaults propagate when value could be False
        if self.shuffle is not None:
            params.setdefault("shuffle", self.shuffle)
        params.setdefault("verbose", self.verbose)

        sig = inspect.signature(trainer_cls)
        filtered: dict[str, Any] = {}
        for name in sig.parameters:
            if name == "self":
                continue
            if name in params:
                filtered[name] = params[name]
        return filtered

    def _apply_runtime_overrides(self, trainer: BaseTrainer) -> None:
        for attr, value in (
            ("learning_rate", self.learning_rate),
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("shuffle", self.shuffle),
            ("verbose", self.verbose),
            ("loss", self.loss),
        ):
            if value is not None and hasattr(trainer, attr):
                setattr(trainer, attr, value)
        if hasattr(trainer, "verbose") and self.verbose is not None:
            trainer.verbose = self.verbose
