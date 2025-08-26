from itertools import product

import numpy as np


class MembershipLayer:
    """Membership layer for ANFIS (Adaptive Neuro-Fuzzy Inference System).

    This is the first layer of ANFIS that applies membership functions to
    input variables. Each input variable has multiple membership functions
    that transform crisp input values into fuzzy membership degrees.

    This layer serves as the fuzzification stage, converting crisp inputs
    into fuzzy sets that can be processed by subsequent ANFIS layers.

    Attributes:
        input_mfs (dict): Dictionary mapping input names to lists of membership functions.
        input_names (list): List of input variable names.
        n_inputs (int): Number of input variables.
        mf_per_input (list): Number of membership functions per input.
        last (dict): Cache of last forward pass computations for backward pass.
    """

    def __init__(self, input_mfs: dict):
        """Initializes the membership layer with input membership functions.

        Parameters:
            input_mfs (dict): Dictionary mapping input names to lists of membership functions.
                             Format: {input_name: [MembershipFunction, ...]}
        """
        self.input_mfs = input_mfs
        self.input_names = list(input_mfs.keys())
        self.n_inputs = len(input_mfs)
        self.mf_per_input = [len(mfs) for mfs in input_mfs.values()]
        self.last = {}

    @property
    def membership_functions(self) -> dict:
        """Alias for input_mfs to provide a standardized interface.

        Returns:
            dict: Dictionary mapping input names to lists of membership functions.
        """
        return self.input_mfs

    def forward(self, x: np.ndarray) -> dict:
        """Performs forward pass to compute membership degrees for all inputs.

        Parameters:
            x (np.ndarray): Input data with shape (batch_size, n_inputs).

        Returns:
            dict: Dictionary mapping input names to membership degree arrays.
                 Format: {input_name: np.ndarray with shape (batch_size, n_mfs)}
        """
        _batch_size = x.shape[0]
        membership_outputs = {}

        # Compute membership degrees for each input variable
        for i, name in enumerate(self.input_names):
            mfs = self.input_mfs[name]
            # Apply each membership function to the i-th input
            mu_values = []
            for mf in mfs:
                mu = mf(x[:, i])  # (batch_size,)
                mu_values.append(mu)

            # Stack membership values for all MFs of this input
            membership_outputs[name] = np.stack(mu_values, axis=-1)  # (batch_size, n_mfs)

        # Cache values for backward pass
        self.last = {"x": x, "membership_outputs": membership_outputs}

        return membership_outputs

    def backward(self, gradients: dict):
        """Performs backward pass to compute gradients for membership functions.

        Parameters:
            gradients (dict): Dictionary mapping input names to gradient arrays.
                             Format: {input_name: np.ndarray with shape (batch_size, n_mfs)}

        Returns:
            None: Gradients are accumulated in the membership functions.
        """
        # Propagate gradients to each membership function
        for name in self.input_names:
            mfs = self.input_mfs[name]
            grad_array = gradients[name]  # (batch_size, n_mfs)

            for mf_idx, mf in enumerate(mfs):
                # Extract gradient for this specific membership function
                mf_gradient = grad_array[:, mf_idx]  # (batch_size,)
                # Propagate gradient to the membership function
                mf.backward(mf_gradient)

    def reset(self):
        """Resets all membership functions to their initial state.

        Returns:
            None
        """
        for name in self.input_names:
            for mf in self.input_mfs[name]:
                mf.reset()
        self.last = {}


class RuleLayer:
    """Rule layer for ANFIS (Adaptive Neuro-Fuzzy Inference System).

    This layer computes the rule strengths (firing strengths) by applying
    the T-norm (typically product) operation to the membership degrees of
    all input variables for each rule.

    This is the second layer of ANFIS that takes membership degrees from
    the MembershipLayer and computes rule activations.

    Attributes:
        input_names (list): List of input variable names.
        n_inputs (int): Number of input variables.
        mf_per_input (list): Number of membership functions per input.
        rules (list): List of all possible rule combinations.
        last (dict): Cache of last forward pass computations for backward pass.
    """

    def __init__(self, input_names: list, mf_per_input: list):
        """Initializes the rule layer with input configuration.

        Parameters:
            input_names (list): List of input variable names.
            mf_per_input (list): Number of membership functions per input variable.
        """
        self.input_names = input_names
        self.n_inputs = len(input_names)
        self.mf_per_input = mf_per_input
        # Generate all possible rule combinations (Cartesian product)
        self.rules = list(product(*[range(n) for n in self.mf_per_input]))
        self.last = {}

    def forward(self, membership_outputs: dict) -> np.ndarray:
        """Performs forward pass to compute rule strengths.

        Parameters:
            membership_outputs (dict): Dictionary mapping input names to membership degree arrays.
                                     Format: {input_name: np.ndarray with shape (batch_size, n_mfs)}

        Returns:
            np.ndarray: Rule strengths with shape (batch_size, n_rules).
        """
        # Convert membership outputs to array format for easier processing
        mu_list = []
        for name in self.input_names:
            mu_list.append(membership_outputs[name])  # (batch_size, n_mfs)
        mu = np.stack(mu_list, axis=1)  # (batch_size, n_inputs, n_mfs)

        _batch_size = mu.shape[0]

        # Compute rule activations (firing strengths)
        rule_activations = []
        for rule in self.rules:
            rule_mu = []
            # Get membership degree for each input in this rule
            for input_idx, mf_idx in enumerate(rule):
                rule_mu.append(mu[:, input_idx, mf_idx])  # (batch_size,)
            # Apply T-norm (product) to get rule strength
            rule_strength = np.prod(rule_mu, axis=0)  # (batch_size,)
            rule_activations.append(rule_strength)

        rule_activations = np.stack(rule_activations, axis=1)  # (batch_size, n_rules)

        # Cache values for backward pass
        self.last = {"membership_outputs": membership_outputs, "mu": mu, "rule_activations": rule_activations}

        return rule_activations

    def backward(self, dL_dw: np.ndarray) -> dict:
        """Performs backward pass to compute gradients for membership functions.

        Parameters:
            dL_dw (np.ndarray): Gradient of loss with respect to rule strengths.
                               Shape: (batch_size, n_rules)

        Returns:
            dict: Dictionary mapping input names to gradient arrays for membership functions.
                 Format: {input_name: np.ndarray with shape (batch_size, n_mfs)}
        """
        batch_size = dL_dw.shape[0]
        mu = self.last["mu"]  # (batch_size, n_inputs, n_mfs)

        # Initialize gradient accumulators for each input's membership functions
        gradients = {}
        for i, name in enumerate(self.input_names):
            n_mfs = self.mf_per_input[i]
            gradients[name] = np.zeros((batch_size, n_mfs))

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
                gradients[name][:, mf_idx] += dL_dw[:, rule_idx] * partial

        return gradients


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
        w = self.last["w"]  # (batch_size, n_rules)
        sum_w = self.last["sum_w"]  # (batch_size, 1)

        # Jacobian-vector product without building the full Jacobian:
        # (J^T g)_j = (sum_w * g_j - (g · w)) / sum_w^2
        g = dL_dnorm_w  # (batch_size, n_rules)
        s = sum_w  # (batch_size, 1)
        gw_dot = np.sum(g * w, axis=1, keepdims=True)  # (batch_size, 1)
        dL_dw = (s * g - gw_dot) / (s**2)  # (batch_size, n_rules)

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


class ClassificationConsequentLayer:
    """Consequent layer that produces per-class logits for classification.

    Each rule i has a vector of class logits with a linear function of inputs:
    f_i(x) = W_i x + b_i, where W_i has shape (n_classes, n_inputs) and b_i (n_classes,).
    We store parameters as a single array of shape (n_rules, n_classes, n_inputs + 1).
    """

    def __init__(self, n_rules: int, n_inputs: int, n_classes: int):
        """Initializes the layer with the specified number of rules, inputs, and classes.

        Args:
            n_rules (int): Number of fuzzy rules in the layer.
            n_inputs (int): Number of input features.
            n_classes (int): Number of output classes.

        Attributes:
            n_rules (int): Stores the number of fuzzy rules.
            n_inputs (int): Stores the number of input features.
            n_classes (int): Stores the number of output classes.
            parameters (np.ndarray): Randomly initialized parameters for each rule, class, and input (including bias).
            gradients (np.ndarray): Gradient values initialized to zeros, matching the shape of parameters.
            last (dict): Dictionary for storing intermediate results or state.
        """
        self.n_rules = n_rules
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.parameters = np.random.randn(n_rules, n_classes, n_inputs + 1)
        self.gradients = np.zeros_like(self.parameters)
        self.last = {}

    def forward(self, x: np.ndarray, norm_w: np.ndarray) -> np.ndarray:
        """Computes the forward pass for the classification consequent layer."""
        batch = x.shape[0]
        X_aug = np.hstack([x, np.ones((batch, 1))])  # (b, d+1)
        # Compute per-rule class logits: (b, r, k)
        f = np.einsum("bd,rkd->brk", X_aug, self.parameters)
        # Weighted sum over rules -> logits (b, k)
        logits = np.einsum("br,brk->bk", norm_w, f)
        self.last = {"X_aug": X_aug, "norm_w": norm_w, "f": f}
        return logits

    def backward(self, dL_dlogits: np.ndarray):
        """Computes the backward pass for the classification consequent layer."""
        X_aug = self.last["X_aug"]  # (b, d+1)
        norm_w = self.last["norm_w"]  # (b, r)
        f = self.last["f"]  # (b, r, k)

        # Gradients w.r.t. per-rule parameters
        self.gradients = np.zeros_like(self.parameters)
        # dL/df_{brk} = dL/dlogits_{bk} * norm_w_{br}
        dL_df = dL_dlogits[:, None, :] * norm_w[:, :, None]  # (b, r, k)
        # Accumulate over batch: grad[r,k,d] = sum_b dL_df[b,r,k] * X_aug[b,d]
        self.gradients = np.einsum("brk,bd->rkd", dL_df, X_aug)

        # dL/dnorm_w: sum_k dL/dlogits_{bk} * f_{brk}
        dL_dnorm_w = np.einsum("bk,brk->br", dL_dlogits, f)

        # dL/dx: sum_r sum_k dL/dlogits_{bk} * norm_w_{br} * W_{r,k,:}
        W = self.parameters[:, :, :-1]  # (r,k,d)
        dL_dx = np.einsum("bk,br,rkd->bd", dL_dlogits, norm_w, W)
        return dL_dnorm_w, dL_dx

    def reset(self):
        """Resets the gradients and cached values."""
        self.gradients = np.zeros_like(self.parameters)
        self.last = {}
