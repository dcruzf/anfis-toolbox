def test_example_1():
    import numpy as np

    from anfis_toolbox import QuickANFIS, quick_evaluate

    # 1. Prepare your data
    X = np.random.uniform(-2, 2, (100, 2))  # 2 inputs
    y = X[:, 0] ** 2 + X[:, 1] ** 2  # Target: x1² + x2²

    # 2. Create and train model (one line!)
    model = QuickANFIS.for_regression(X, n_mfs=3)
    _losses = model.fit_hybrid(X, y, epochs=50)

    # 3. Evaluate and use
    _metrics = quick_evaluate(model, X, y)
    _predictions = model.predict([[1.0, -0.5], [0.5, 1.2]])


def test_example_2():
    import numpy as np

    from anfis_toolbox import QuickANFIS, quick_evaluate

    # 1. Prepare your data
    X = np.random.uniform(-1, 1, (100, 1))  # 1 input
    y = X**2  # Target: x1²

    # 2. Create and train model (one line!)
    model = QuickANFIS.for_regression(X, n_mfs=3)
    _losses = model.fit_hybrid(X, y, epochs=50)

    # 3. Evaluate and use
    _metrics = quick_evaluate(model, X, y)
    _predictions = model.predict([[0.75]])


def test_example_3():
    import numpy as np

    from anfis_toolbox import QuickANFIS, quick_evaluate

    # Generate data
    X = np.random.uniform(-3, 3, (200, 2))
    y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(200)

    # Create and train model
    model = QuickANFIS.for_regression(X, n_mfs=4, mf_type="gaussian")
    _losses = model.fit_hybrid(X, y, epochs=100, learning_rate=0.01)

    # Evaluate and visualize
    # Evaluate (placeholder)
    _metrics = quick_evaluate(model, X, y)
    _predictions = model.predict(X[:5])
