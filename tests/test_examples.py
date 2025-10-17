import numpy as np

from anfis_toolbox.builders import ANFISBuilder
from anfis_toolbox.metrics import quick_evaluate


def _build_regression_model(X: np.ndarray, *, n_mfs: int = 3, mf_type: str = "gaussian"):
    builder = ANFISBuilder()
    for i in range(X.shape[1]):
        col = X[:, i]
        rmin = float(np.min(col))
        rmax = float(np.max(col))
        margin = (rmax - rmin) * 0.1
        builder.add_input(f"x{i + 1}", rmin - margin, rmax + margin, n_mfs, mf_type)
    return builder.build()


def test_example_1():
    X = np.random.uniform(-2, 2, (100, 2))  # 2 inputs
    y = X[:, 0] ** 2 + X[:, 1] ** 2  # Target: x1² + x2²

    model = _build_regression_model(X, n_mfs=3)
    _losses = model.fit(X, y, epochs=50)

    _metrics = quick_evaluate(model, X, y)
    _predictions = model.predict([[1.0, -0.5], [0.5, 1.2]])


def test_example_2():
    X = np.random.uniform(-1, 1, (100, 1))  # 1 input
    y = X**2  # Target: x1²

    model = _build_regression_model(X, n_mfs=3)
    _losses = model.fit(X, y, epochs=50)

    _metrics = quick_evaluate(model, X, y)
    _predictions = model.predict([[0.75]])


def test_example_3():
    X = np.random.uniform(-3, 3, (200, 2))
    y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(200)

    model = _build_regression_model(X, n_mfs=4, mf_type="gaussian")
    _losses = model.fit(X, y, epochs=10, learning_rate=0.01)

    _metrics = quick_evaluate(model, X, y)
    _predictions = model.predict(X[:5])
