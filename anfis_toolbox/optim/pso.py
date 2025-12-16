from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np

from ..losses import LossFunction, resolve_loss
from .base import BaseTrainer


class _FlattenMeta(TypedDict):
    consequent_shape: tuple[int, ...]
    n_consequent: int
    membership_info: list[tuple[str, int, str]]


def _flatten_params(params: dict[str, Any]) -> tuple[np.ndarray, _FlattenMeta]:
    """Flatten model parameters into a 1D vector and return meta for reconstruction.

    The expected structure matches model.get_parameters():
      { 'consequent': np.ndarray, 'membership': { name: [ {param: val, ...}, ... ] } }
    """
    cons = params["consequent"].ravel()
    memb_info: list[tuple[str, int, str]] = []
    memb_vals: list[float] = []
    for name in params["membership"].keys():
        for i, mf_params in enumerate(params["membership"][name]):
            for key in mf_params.keys():
                memb_info.append((name, i, key))
                memb_vals.append(float(mf_params[key]))
    memb = np.asarray(memb_vals, dtype=float)
    if memb.size:
        theta = np.concatenate([cons, memb])
    else:
        theta = cons.copy()
    meta: _FlattenMeta = {
        "consequent_shape": params["consequent"].shape,
        "n_consequent": cons.size,
        "membership_info": memb_info,
    }
    return theta, meta


def _unflatten_params(theta: np.ndarray, meta: _FlattenMeta, template: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct parameter dictionary from theta using meta and template structure."""
    n_cons = meta["n_consequent"]
    cons = theta[:n_cons].reshape(meta["consequent_shape"])
    out: dict[str, Any] = {"consequent": cons.copy(), "membership": {}}
    offset = n_cons
    # Copy structure from template membership dict
    for name in template["membership"].keys():
        out["membership"][name] = []
        for _ in range(len(template["membership"][name])):
            out["membership"][name].append({})
    # Assign values in the same order used in flatten
    for name, i, key in meta["membership_info"]:
        out["membership"][name][i][key] = float(theta[offset])
        offset += 1
    return out


@dataclass
class PSOTrainer(BaseTrainer):
    """Particle Swarm Optimization (PSO) trainer for ANFIS.

    Parameters:
        swarm_size: Number of particles.
        inertia: Inertia weight (w).
        cognitive: Cognitive coefficient (c1).
        social: Social coefficient (c2).
        epochs: Number of iterations of the swarm update.
        init_sigma: Std-dev for initializing particle positions around current params.
        clamp_velocity: Optional (min, max) to clip velocities element-wise.
        clamp_position: Optional (min, max) to clip positions element-wise.
        random_state: Seed for RNG to ensure determinism.
        verbose: Unused here; kept for API parity.

    Notes:
        Optimizes the loss specified by ``loss`` (defaulting to mean squared error) by searching
        directly in parameter space without gradients. With ``ANFISClassifier`` you can set
        ``loss="cross_entropy"`` to optimize categorical cross-entropy on logits.
    """

    swarm_size: int = 20
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    epochs: int = 100
    init_sigma: float = 0.1
    clamp_velocity: None | tuple[float, float] = None
    clamp_position: None | tuple[float, float] = None
    random_state: None | int = None
    verbose: bool = False
    loss: LossFunction | str | None = None
    _loss_fn: LossFunction = field(init=False, repr=False)

    def _prepare_training_data(self, model: Any, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._loss_fn = resolve_loss(self.loss)
        return self._prepare_batch(X, y, model)

    def _prepare_validation_data(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._prepare_batch(X_val, y_val, model)

    def init_state(self, model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Initialize PSO swarm state and return as a dict."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        base_params = model.get_parameters()
        theta0, meta = _flatten_params(base_params)
        D = theta0.size
        positions = theta0[None, :] + self.init_sigma * rng.normal(size=(self.swarm_size, D))
        velocities = np.zeros((self.swarm_size, D), dtype=float)
        # Initialize personal/global bests on provided data
        personal_best_pos = positions.copy()
        personal_best_val = np.empty(self.swarm_size, dtype=float)
        for i in range(self.swarm_size):
            params_i = _unflatten_params(positions[i], meta, base_params)
            with self._temporary_parameters(model, params_i):
                personal_best_val[i] = self._evaluate_loss(model, X, y)
        g_idx = int(np.argmin(personal_best_val))
        global_best_pos = personal_best_pos[g_idx].copy()
        global_best_val = float(personal_best_val[g_idx])
        return {
            "meta": meta,
            "template": base_params,
            "positions": positions,
            "velocities": velocities,
            "pbest_pos": personal_best_pos,
            "pbest_val": personal_best_val,
            "gbest_pos": global_best_pos,
            "gbest_val": global_best_val,
            "rng": rng,
        }

    def train_step(
        self, model: Any, Xb: np.ndarray, yb: np.ndarray, state: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        """Perform one PSO iteration over the swarm on a batch and return (best_loss, state)."""
        positions = state["positions"]
        velocities = state["velocities"]
        personal_best_pos = state["pbest_pos"]
        personal_best_val = state["pbest_val"]
        global_best_pos = state["gbest_pos"]
        global_best_val = state["gbest_val"]
        meta = state["meta"]
        template = state["template"]
        rng = state["rng"]

        D = positions.shape[1]
        r1 = rng.random(size=(self.swarm_size, D))
        r2 = rng.random(size=(self.swarm_size, D))
        cognitive_term = self.cognitive * r1 * (personal_best_pos - positions)
        social_term = self.social * r2 * (global_best_pos[None, :] - positions)
        velocities = self.inertia * velocities + cognitive_term + social_term
        if self.clamp_velocity is not None:
            vmin, vmax = self.clamp_velocity
            velocities = np.clip(velocities, vmin, vmax)
        positions = positions + velocities
        if self.clamp_position is not None:
            pmin, pmax = self.clamp_position
            positions = np.clip(positions, pmin, pmax)

        # Evaluate swarm and update bests
        for i in range(self.swarm_size):
            params_i = _unflatten_params(positions[i], meta, template)
            with self._temporary_parameters(model, params_i):
                val = self._evaluate_loss(model, Xb, yb)
            if val < personal_best_val[i]:
                personal_best_val[i] = val
                personal_best_pos[i] = positions[i].copy()
                if val < global_best_val:
                    global_best_val = float(val)
                    global_best_pos = positions[i].copy()

        # Update state and set model to global best
        state.update(
            {
                "positions": positions,
                "velocities": velocities,
                "pbest_pos": personal_best_pos,
                "pbest_val": personal_best_val,
                "gbest_pos": global_best_pos,
                "gbest_val": global_best_val,
            }
        )
        best_params = _unflatten_params(global_best_pos, meta, template)
        model.set_parameters(best_params)
        return float(global_best_val), state

    @contextmanager
    def _temporary_parameters(self, model: Any, params: dict[str, Any]) -> Generator[None, None, None]:
        original = model.get_parameters()
        model.set_parameters(params)
        try:
            yield
        finally:
            model.set_parameters(original)

    def _prepare_batch(self, X: np.ndarray, y: np.ndarray, model: Any) -> tuple[np.ndarray, np.ndarray]:
        loss_fn = self._ensure_loss_fn()
        X_arr = np.asarray(X, dtype=float)
        y_arr = loss_fn.prepare_targets(y, model=model)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("Target array must have same number of rows as X")
        return X_arr, y_arr

    def _evaluate_loss(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        preds = model.forward(X)
        return float(self._loss_fn.loss(y, preds))

    def compute_loss(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the swarm's current parameters on ``(X, y)`` without mutation."""
        return self._evaluate_loss(model, X, y)

    def _ensure_loss_fn(self) -> LossFunction:
        if not hasattr(self, "_loss_fn"):
            self._loss_fn = resolve_loss(self.loss)
        return self._loss_fn
