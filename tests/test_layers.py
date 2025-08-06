import numpy as np
import pytest

from anfis_toolbox.layers import RuleLayer
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
