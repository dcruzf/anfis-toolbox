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
