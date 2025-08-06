from itertools import product

import numpy as np


class RuleLayer:
    """Rule layer for ANFIS (Adaptive Neuro-Fuzzy Inference System).

    This layer computes the rule strengths (firing strengths) by applying
    the T-norm (typically product) operation to the membership degrees of
    all input variables for each rule.

    Attributes:
        input_mfs (dict): Dictionary mapping input names to lists of membership functions.
        input_names (list): List of input variable names.
        n_inputs (int): Number of input variables.
        mf_per_input (list): Number of membership functions per input.
        rules (list): List of all possible rule combinations.
        last (dict): Cache of last forward pass computations for backward pass.
    """

    def __init__(self, input_mfs: dict):
        """Initializes the rule layer with input membership functions.

        Parameters:
            input_mfs (dict): Dictionary mapping input names to lists of membership functions.
                             Format: {input_name: [MembershipFunction, ...]}
        """
        self.input_mfs = input_mfs
        self.input_names = list(input_mfs.keys())
        self.n_inputs = len(input_mfs)
        self.mf_per_input = [len(mfs) for mfs in input_mfs.values()]
        # Generate all possible rule combinations (Cartesian product)
        self.rules = list(product(*[range(n) for n in self.mf_per_input]))
        self.last = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs forward pass to compute rule strengths.

        Parameters:
            x (np.ndarray): Input data with shape (batch_size, n_inputs).

        Returns:
            np.ndarray: Rule strengths with shape (batch_size, n_rules).
        """
        mu = []  # Membership degrees for each MF of each input

        # Compute membership degrees for each input variable
        for i, name in enumerate(self.input_names):
            mfs = self.input_mfs[name]
            # Stack membership values for all MFs of this input
            mu_i = np.stack([mf(x[:, i]) for mf in mfs], axis=-1)  # (batch, n_mfs)
            mu.append(mu_i)
        mu = np.stack(mu, axis=1)  # (batch, n_inputs, n_mfs)

        # Compute rule activations (firing strengths)
        rule_activations = []
        for rule in self.rules:
            rule_mu = []
            # Get membership degree for each input in this rule
            for input_idx, mf_idx in enumerate(rule):
                rule_mu.append(mu[:, input_idx, mf_idx])  # (batch,)
            # Apply T-norm (product) to get rule strength
            rule_strength = np.prod(rule_mu, axis=0)  # (batch,)
            rule_activations.append(rule_strength)

        rule_activations = np.stack(rule_activations, axis=1)  # (batch, n_rules)

        # Cache values for backward pass
        self.last = {"x": x, "mu": mu, "rule_activations": rule_activations}

        return rule_activations

    def backward(self, dL_dw: np.ndarray):
        """Performs backward pass to compute gradients for membership functions.

        Parameters:
            dL_dw (np.ndarray): Gradient of loss with respect to rule strengths.
                               Shape: (batch_size, n_rules)

        Returns:
            None: Gradients are accumulated in the membership functions.
        """
        batch_size = dL_dw.shape[0]
        mu = self.last["mu"]  # (batch, n_inputs, n_mfs)

        # Initialize gradient accumulators for each membership function
        dmu = {name: [np.zeros(batch_size) for _ in self.input_mfs[name]] for name in self.input_names}

        # Compute gradients for each rule
        for rule_idx, rule in enumerate(self.rules):
            for input_idx, mf_idx in enumerate(rule):
                name = self.input_names[input_idx]

                # Compute partial derivative: d(rule_strength)/d(mu_ij)
                # This is the product of all other membership degrees in the rule
                other_factors = []
                for j, j_mf in enumerate(rule):
                    if j == input_idx:
                        continue  # Skip the current input
                    other_factors.append(mu[:, j, j_mf])

                # Product of other factors (or 1 if no other factors)
                partial = np.prod(other_factors, axis=0) if other_factors else np.ones(batch_size)

                # Apply chain rule: dL/dmu = dL/dw * dw/dmu
                dmu[name][mf_idx] += dL_dw[:, rule_idx] * partial

        # Propagate gradients to each membership function
        for name in self.input_names:
            for idx, mf in enumerate(self.input_mfs[name]):
                mf.backward(dmu[name][idx])


class NormalizationLayer:
    """Normalization layer for ANFIS (Adaptive Neuro-Fuzzy Inference System).

    This layer normalizes the rule strengths (firing strengths) to ensure
    they sum to 1.0 for each sample in the batch. This is a crucial step
    in ANFIS as it converts rule strengths to normalized rule weights.

    The normalization formula is: norm_w_i = w_i / sum(w_j for all j)

    Attributes:
        last (dict): Cache of last forward pass computations for backward pass.
    """

    def __init__(self):
        """Initializes the normalization layer."""
        self.last = {}

    def forward(self, w: np.ndarray) -> np.ndarray:
        """Performs forward pass to normalize rule weights.

        Parameters:
            w (np.ndarray): Rule strengths with shape (batch_size, n_rules).

        Returns:
            np.ndarray: Normalized rule weights with shape (batch_size, n_rules).
                       Each row sums to 1.0.
        """
        # Add small epsilon to avoid division by zero
        sum_w = np.sum(w, axis=1, keepdims=True) + 1e-8
        norm_w = w / sum_w

        # Cache values for backward pass
        self.last = {"w": w, "sum_w": sum_w, "norm_w": norm_w}
        return norm_w

    def backward(self, dL_dnorm_w: np.ndarray) -> np.ndarray:
        """Performs backward pass to compute gradients for original rule weights.

        The gradient computation uses the quotient rule for derivatives:
        If norm_w_i = w_i / sum_w, then:
        - d(norm_w_i)/d(w_i) = (sum_w - w_i) / sum_w²
        - d(norm_w_i)/d(w_j) = -w_j / sum_w² for j ≠ i

        Parameters:
            dL_dnorm_w (np.ndarray): Gradient of loss with respect to normalized weights.
                                    Shape: (batch_size, n_rules)

        Returns:
            np.ndarray: Gradient of loss with respect to original weights.
                       Shape: (batch_size, n_rules)
        """
        w = self.last["w"]
        sum_w = self.last["sum_w"]

        batch_size, n_rules = w.shape
        dL_dw = np.zeros_like(w)

        # Compute gradients using the quotient rule
        for b in range(batch_size):
            for i in range(n_rules):
                for j in range(n_rules):
                    if i == j:
                        # Derivative of w_i / sum_w with respect to w_i
                        grad = (sum_w[b, 0] - w[b, i]) / (sum_w[b, 0] ** 2)
                    else:
                        # Derivative of w_i / sum_w with respect to w_j (j ≠ i)
                        grad = -w[b, j] / (sum_w[b, 0] ** 2)

                    # Apply chain rule: dL/dw = dL/dnorm_w * dnorm_w/dw
                    dL_dw[b, i] += dL_dnorm_w[b, j] * grad

        return dL_dw


class ConsequentLayer:
    """Consequent layer for ANFIS (Adaptive Neuro-Fuzzy Inference System).

    This layer implements the consequent part of fuzzy rules in ANFIS.
    Each rule has a linear consequent function of the form:
    f_i(x) = p_i * x_1 + q_i * x_2 + ... + r_i (TSK model)

    The final output is computed as a weighted sum:
    y = Σ(w_i * f_i(x)) where w_i are normalized rule weights

    Attributes:
        n_rules (int): Number of fuzzy rules.
        n_inputs (int): Number of input variables.
        parameters (np.ndarray): Linear parameters for each rule with shape (n_rules, n_inputs + 1).
                                Each row contains [p_i, q_i, ..., r_i] for rule i.
        gradients (np.ndarray): Accumulated gradients for parameters.
        last (dict): Cache of last forward pass computations for backward pass.
    """

    def __init__(self, n_rules: int, n_inputs: int):
        """Initializes the consequent layer with random linear parameters.

        Parameters:
            n_rules (int): Number of fuzzy rules.
            n_inputs (int): Number of input variables.
        """
        # Each rule has (n_inputs + 1) parameters: p_i, q_i, ..., r_i (including bias)
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.parameters = np.random.randn(n_rules, n_inputs + 1)
        self.gradients = np.zeros_like(self.parameters)
        self.last = {}

    def forward(self, x: np.ndarray, norm_w: np.ndarray) -> np.ndarray:
        """Performs forward pass to compute the final ANFIS output.

        Parameters:
            x (np.ndarray): Input data with shape (batch_size, n_inputs).
            norm_w (np.ndarray): Normalized rule weights with shape (batch_size, n_rules).

        Returns:
            np.ndarray: Final ANFIS output with shape (batch_size, 1).
        """
        batch_size = x.shape[0]

        # Augment input with bias term (column of ones)
        X_aug = np.hstack([x, np.ones((batch_size, 1))])  # (batch_size, n_inputs + 1)

        # Compute consequent function f_i(x) for each rule
        # f[b, i] = p_i * x[b, 0] + q_i * x[b, 1] + ... + r_i
        f = X_aug @ self.parameters.T  # (batch_size, n_rules)

        # Compute final output as weighted sum: y = Σ(w_i * f_i(x))
        y_hat = np.sum(norm_w * f, axis=1, keepdims=True)  # (batch_size, 1)

        # Cache values for backward pass
        self.last = {"X_aug": X_aug, "norm_w": norm_w, "f": f}

        return y_hat

    def backward(self, dL_dy: np.ndarray):
        """Performs backward pass to compute gradients for parameters and inputs.

        Parameters:
            dL_dy (np.ndarray): Gradient of loss with respect to layer output.
                               Shape: (batch_size, 1)

        Returns:
            tuple: (dL_dnorm_w, dL_dx) where:
                - dL_dnorm_w: Gradient w.r.t. normalized weights, shape (batch_size, n_rules)
                - dL_dx: Gradient w.r.t. input x, shape (batch_size, n_inputs)
        """
        X_aug = self.last["X_aug"]  # (batch_size, n_inputs + 1)
        norm_w = self.last["norm_w"]  # (batch_size, n_rules)
        f = self.last["f"]  # (batch_size, n_rules)

        batch_size = X_aug.shape[0]

        # Compute gradients for consequent parameters
        self.gradients = np.zeros_like(self.parameters)

        for i in range(self.n_rules):
            # Gradient of y_hat w.r.t. parameters of rule i: norm_w_i * x_aug
            for b in range(batch_size):
                self.gradients[i] += dL_dy[b, 0] * norm_w[b, i] * X_aug[b]

        # Compute gradient of loss w.r.t. normalized weights
        # dy/dnorm_w_i = f_i(x), so dL/dnorm_w_i = dL/dy * f_i(x)
        dL_dnorm_w = dL_dy * f  # (batch_size, n_rules)

        # Compute gradient of loss w.r.t. input x (for backpropagation to previous layers)
        dL_dx = np.zeros((batch_size, self.n_inputs))

        for b in range(batch_size):
            for i in range(self.n_rules):
                # dy/dx = norm_w_i * parameters_i[:-1] (excluding bias term)
                dL_dx[b] += dL_dy[b, 0] * norm_w[b, i] * self.parameters[i, :-1]

        return dL_dnorm_w, dL_dx

    def reset(self):
        """Resets gradients and cached values.

        Returns:
            None
        """
        self.gradients = np.zeros_like(self.parameters)
        self.last = {}
