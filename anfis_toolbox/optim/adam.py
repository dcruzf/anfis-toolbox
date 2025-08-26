from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseTrainer


def _zeros_like_structure(params):
    """Create a zero-structure matching model.get_parameters() format.

    Returns a dict with:
      - 'consequent': np.zeros_like(params['consequent'])
      - 'membership': { name: [ {param_name: 0.0, ...} ] }
    """
    out = {"consequent": np.zeros_like(params["consequent"]), "membership": {}}
    for name, mf_list in params["membership"].items():
        out["membership"][name] = []
        for mf_params in mf_list:
            out["membership"][name].append(dict.fromkeys(mf_params.keys(), 0.0))
    return out


def _adam_update(param, grad, m, v, lr, beta1, beta2, eps, t):
    """Compute Adam update for numpy arrays (param, grad, m, v)."""
    m[:] = beta1 * m + (1.0 - beta1) * grad
    v[:] = beta2 * v + (1.0 - beta2) * (grad * grad)
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    param[:] = param - lr * m_hat / (np.sqrt(v_hat) + eps)


@dataclass
class AdamTrainer(BaseTrainer):
    """Adam optimizer-based trainer for ANFIS.

    Parameters:
        learning_rate: Base step size (alpha).
        beta1: Exponential decay rate for the first moment estimates.
        beta2: Exponential decay rate for the second moment estimates.
        epsilon: Small constant for numerical stability.
        epochs: Number of passes over the dataset.
        batch_size: If None, use full-batch; otherwise mini-batches of this size.
        shuffle: Whether to shuffle the data at each epoch when using mini-batches.
        verbose: Unused here; kept for API parity.

    Notes:
                - Optimizes mean squared error (MSE) between ``model.forward(X)`` and ``y``.
                    For regression ANFIS this is the intended objective.
                - With ``ANFISClassifier``, this trainer will still execute (treating integer
                    labels reshaped to ``(n,1)`` or one-hot targets) but it will minimize MSE on
                    logits/probabilities. For classification tasks, prefer ``ANFISClassifier.fit``
                    which uses cross-entropy with softmax.
    """

    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    epochs: int = 100
    batch_size: None | int = None
    shuffle: bool = True
    verbose: bool = True

    def fit(self, model, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train the model using Adam optimization.

        This involves computing the forward pass, loss, backward pass, and applying
        the Adam update step for each training iteration.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        # Initialize Adam state structures matching parameter shapes
        params = model.get_parameters()
        m = _zeros_like_structure(params)
        v = _zeros_like_structure(params)
        t = 0  # time step

        def _apply_adam_step(grads):
            nonlocal t
            t += 1
            # Consequent (numpy array)
            _adam_update(
                params["consequent"],
                grads["consequent"],
                m["consequent"],
                v["consequent"],
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
                t,
            )
            # Membership (nested scalars)
            for name in params["membership"].keys():
                for i in range(len(params["membership"][name])):
                    for key in params["membership"][name][i].keys():
                        # Scalars: use simple Adam update formula
                        g = float(grads["membership"][name][i][key])
                        m_val = m["membership"][name][i][key]
                        v_val = v["membership"][name][i][key]
                        m_val = self.beta1 * m_val + (1.0 - self.beta1) * g
                        v_val = self.beta2 * v_val + (1.0 - self.beta2) * (g * g)
                        m_hat = m_val / (1.0 - self.beta1**t)
                        v_hat = v_val / (1.0 - self.beta2**t)
                        step = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                        params["membership"][name][i][key] -= float(step)
                        m["membership"][name][i][key] = m_val
                        v["membership"][name][i][key] = v_val

            # Push updated params back into the model
            model.set_parameters(params)

        losses: list[float] = []
        for _ in range(self.epochs):
            if self.batch_size is None:
                # Full-batch Adam step
                model.reset_gradients()
                y_pred = model.forward(X)
                loss = float(np.mean((y_pred - y) ** 2))
                dL_dy = 2.0 * (y_pred - y) / y.shape[0]
                model.backward(dL_dy)
                grads = model.get_gradients()
                _apply_adam_step(grads)
                losses.append(loss)
            else:
                indices = np.arange(n_samples)
                if self.shuffle:
                    np.random.shuffle(indices)
                batch_losses: list[float] = []
                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    batch_idx = indices[start:end]
                    model.reset_gradients()
                    y_pred_b = model.forward(X[batch_idx])
                    batch_loss = float(np.mean((y_pred_b - y[batch_idx]) ** 2))
                    dL_dy_b = 2.0 * (y_pred_b - y[batch_idx]) / y[batch_idx].shape[0]
                    model.backward(dL_dy_b)
                    grads_b = model.get_gradients()
                    _apply_adam_step(grads_b)
                    batch_losses.append(batch_loss)
                losses.append(float(np.mean(batch_losses)))

        return losses
