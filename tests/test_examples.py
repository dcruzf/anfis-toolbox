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
