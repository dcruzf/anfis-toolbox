import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anfis_toolbox import ANFISClassifier, ANFISRegressor


def _make_regression_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(60, 3))
    weights = np.array([1.5, -2.0, 0.5])
    y = X @ weights + rng.normal(0.0, 0.05, size=X.shape[0])
    return X, y


def _make_classification_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(80, 2))
    boundary = 0.8 * X[:, 0] - 0.6 * X[:, 1]
    y = (boundary > 0.0).astype(int)
    return X, y


def test_classifier_integration_with_sklearn_cross_val() -> None:
    X, y = _make_classification_data()
    estimator = ANFISClassifier(n_mfs=2, epochs=2, learning_rate=0.05, random_state=0)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_regressor_integration_with_sklearn_pipeline() -> None:
    X, y = _make_regression_data()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                ANFISRegressor(
                    n_mfs=2,
                    optimizer="sgd",
                    epochs=2,
                    learning_rate=0.05,
                    random_state=0,
                ),
            ),
        ]
    )

    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")

    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))
