import numpy as np
import pytest

from anfis_toolbox.layers import ConsequentLayer, NormalizationLayer, RuleLayer
from anfis_toolbox.membership import GaussianMF


@pytest.fixture
def simple_rule_layer():
    # Duas entradas com duas funções de pertinência cada
    mf_x1 = [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]
    mf_x2 = [GaussianMF(mean=0.0, sigma=1.0), GaussianMF(mean=1.0, sigma=1.0)]

    input_mfs = {"x1": mf_x1, "x2": mf_x2}

    return RuleLayer(input_mfs)


def test_rule_layer_forward(simple_rule_layer):
    x = np.array([[0.0, 0.0]])  # shape (1, 2)

    rule_strengths = simple_rule_layer.forward(x)

    # Devem existir 4 regras (2x2 combinações)
    assert rule_strengths.shape == (1, 4)

    # Verificação manual do produto das pertinências
    mu_x1 = [mf(x[:, 0])[0] for mf in simple_rule_layer.input_mfs["x1"]]
    mu_x2 = [mf(x[:, 1])[0] for mf in simple_rule_layer.input_mfs["x2"]]

    expected_strengths = [
        mu_x1[0] * mu_x2[0],
        mu_x1[0] * mu_x2[1],
        mu_x1[1] * mu_x2[0],
        mu_x1[1] * mu_x2[1],
    ]

    np.testing.assert_allclose(rule_strengths[0], expected_strengths, rtol=1e-5)


def test_rule_layer_backward(simple_rule_layer):
    x = np.array([[0.0, 0.0]])
    output = simple_rule_layer.forward(x)

    dL_dw = np.ones_like(output)  # dL/dw = 1 para todas as regras
    simple_rule_layer.backward(dL_dw)

    # Verifica se todos os gradientes de cada MF estão definidos
    for _name, mfs in simple_rule_layer.input_mfs.items():
        for mf in mfs:
            assert "mean" in mf.gradients
            assert "sigma" in mf.gradients
            assert np.isscalar(mf.gradients["mean"])
            assert np.isscalar(mf.gradients["sigma"])


def test_normalization_forward():
    layer = NormalizationLayer()
    w = np.array([[1.0, 2.0, 3.0]])
    norm = layer.forward(w)

    expected = w / np.sum(w, axis=1, keepdims=True)
    np.testing.assert_allclose(norm, expected, rtol=1e-6)


def test_normalization_backward():
    layer = NormalizationLayer()
    w = np.array([[1.0, 2.0, 3.0]])
    layer.forward(w)

    dL_dnorm = np.array([[1.0, 0.0, 0.0]])  # só a primeira saída tem gradiente
    dL_dw = layer.backward(dL_dnorm)

    # Verifica numericamente o gradiente com diferença finita
    epsilon = 1e-5
    numerical = np.zeros_like(w)

    for i in range(w.shape[1]):
        w_pos = w.copy()
        w_neg = w.copy()
        w_pos[0, i] += epsilon
        w_neg[0, i] -= epsilon

        out_pos = w_pos / np.sum(w_pos, axis=1, keepdims=True)
        out_neg = w_neg / np.sum(w_neg, axis=1, keepdims=True)

        loss_pos = out_pos[0, 0]  # como dL/dnorm = [1, 0, 0]
        loss_neg = out_neg[0, 0]

        numerical[0, i] = (loss_pos - loss_neg) / (2 * epsilon)

    np.testing.assert_allclose(dL_dw, numerical, rtol=1e-4, atol=1e-6)


def test_consequent_forward_shape():
    layer = ConsequentLayer(n_rules=3, n_inputs=2)
    x = np.array([[1.0, 2.0]])
    norm_w = np.array([[0.2, 0.3, 0.5]])

    y_hat = layer.forward(x, norm_w)

    assert y_hat.shape == (1, 1)


def test_consequent_backward_gradients():
    np.random.seed(0)
    n_rules = 3
    n_inputs = 2
    batch_size = 1

    layer = ConsequentLayer(n_rules=n_rules, n_inputs=n_inputs)
    x = np.random.randn(batch_size, n_inputs)
    norm_w = np.random.rand(batch_size, n_rules)
    norm_w /= norm_w.sum(axis=1, keepdims=True)  # garantir normalização

    y_hat = layer.forward(x, norm_w)
    dL_dy = np.ones_like(y_hat)  # gradiente da perda em relação à saída

    dL_dnorm_w, dL_dx = layer.backward(dL_dy)

    # Verificação numérica dos gradientes dos parâmetros
    epsilon = 1e-5
    numerical_grad = np.zeros_like(layer.parameters)

    for i in range(n_rules):
        for j in range(n_inputs + 1):
            original = layer.parameters[i, j]

            layer.parameters[i, j] = original + epsilon
            y_pos = layer.forward(x, norm_w)

            layer.parameters[i, j] = original - epsilon
            y_neg = layer.forward(x, norm_w)

            numerical_grad[i, j] = (y_pos - y_neg).squeeze() / (2 * epsilon)
            layer.parameters[i, j] = original  # restaurar

    np.testing.assert_allclose(layer.gradients, numerical_grad, rtol=1e-4, atol=1e-6)
