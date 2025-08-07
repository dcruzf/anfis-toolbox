import numpy as np
import pytest

from anfis_toolbox.membership import GaussianMF


def test_gaussian_mf():
    """Test the Gaussian membership function."""
    mean = 0.0
    sigma = 1.0
    gaussian_mf = GaussianMF(mean, sigma)

    # Test forward pass
    x = np.array([-2, -1, 0, 1, 2])
    output = gaussian_mf(x)
    expected_output = np.exp(-((x - mean) ** 2) / (2 * sigma**2))

    assert np.allclose(output, expected_output), "Forward pass failed"

    # Test backward pass
    dL_dy = np.ones_like(output)
    gaussian_mf.backward(dL_dy)

    # Check if gradients are computed correctly
    assert "mean" in gaussian_mf.gradients and "sigma" in gaussian_mf.gradients, "Gradients not computed"

    gaussian_mf.reset()
    assert gaussian_mf.last_input is None and gaussian_mf.last_output is None, "Reset failed"
    assert all(v == 0.0 for v in gaussian_mf.gradients.values()), "Gradients not reset"


@pytest.mark.parametrize("param_name", ["mean", "sigma"])
def test_gaussianmf_gradient(param_name):
    eps = 1e-5
    x = np.array([1.0, 2.0, 3.0])

    # Inicializa a função de pertinência
    mf = GaussianMF(mean=0.5, sigma=1.0)

    # Calcula saída original e perda
    y = mf.forward(x)

    # Backward manual
    dL_dy = np.ones_like(y)
    mf.backward(dL_dy)
    grad_analytical = mf.gradients[param_name]

    # Gradiente numérico
    original_value = mf.parameters[param_name]

    # L(param + eps)
    mf.parameters[param_name] = original_value + eps
    y_plus = mf.forward(x)
    L_plus = np.sum(y_plus)

    # L(param - eps)
    mf.parameters[param_name] = original_value - eps
    y_minus = mf.forward(x)
    L_minus = np.sum(y_minus)

    grad_numerical = (L_plus - L_minus) / (2 * eps)

    # Restaurar o parâmetro original
    mf.parameters[param_name] = original_value

    # Verificação
    assert np.allclose(grad_analytical, grad_numerical, atol=1e-4), (
        f"Gradient check failed for {param_name}.\nAnalytical: {grad_analytical}, Numerical: {grad_numerical}"
    )
