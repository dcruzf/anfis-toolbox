import numpy as np
import pytest

from anfis_toolbox.estimator_utils import (
    BaseEstimatorLike,
    ClassifierMixinLike,
    FittedMixin,
    NotFittedError,
    RegressorMixinLike,
    _ensure_2d_array,
    _ensure_vector,
    check_is_fitted,
    infer_feature_names,
)


class CustomObject:
    def __init__(self, value: int):
        self.value = value


class DummyEstimator(BaseEstimatorLike, FittedMixin):
    def __init__(self, alpha: float = 1.0, options: dict | None = None):
        self.alpha = alpha
        self.options = options or {"beta": 2, "list": [1, 2], "obj": CustomObject(5)}

    def fit(self):
        self._mark_fitted()
        return self


def test_get_set_params_roundtrip_and_check_is_fitted():
    est = DummyEstimator(alpha=2.5, options={"gamma": 4, "obj": CustomObject(3), "list": [3, 4]})
    params = est.get_params()
    assert params["alpha"] == 2.5
    assert params["options"]["gamma"] == 4
    assert params["options"]["list"] == [3, 4]
    assert isinstance(params["options"]["obj"], CustomObject)

    est2 = DummyEstimator()
    est2.set_params(**params)
    assert est2.alpha == 2.5
    assert est2.options["gamma"] == 4
    assert est2.options["list"] == [3, 4]
    assert isinstance(est2.options["obj"], CustomObject)
    assert est2.options["obj"] is not est.options["obj"]

    with pytest.raises(ValueError):
        est2.set_params(unknown=1)

    with pytest.raises(NotFittedError):
        check_is_fitted(est2)

    est2.fit()
    check_is_fitted(est2)


def test_check_is_fitted_missing_attributes():
    class PartialEstimator(FittedMixin):
        pass

    est = PartialEstimator()
    est._mark_fitted()
    with pytest.raises(NotFittedError, match="missing fitted attribute"):
        check_is_fitted(est, attributes=["model_"])


def test_ensure_2d_array_and_infer_feature_names():
    X = [[1, 2, 3], [4, 5, 6]]
    array, names = _ensure_2d_array(X)
    assert array.shape == (2, 3)
    assert names == ["x1", "x2", "x3"]

    feature_names = infer_feature_names(np.asarray(X, dtype=float))
    assert feature_names == ["x1", "x2", "x3"]

    class DummyFrame:
        def __init__(self):
            self.columns = ["a", "b"]

        def to_numpy(self, dtype=float):
            return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)

    frame = DummyFrame()
    array2, names2 = _ensure_2d_array(frame)
    assert names2 == ["a", "b"]
    assert infer_feature_names(frame) == ["a", "b"]

    with pytest.raises(ValueError):
        _ensure_2d_array([1, 2, 3])

    with pytest.raises(ValueError):
        infer_feature_names(np.array([1.0, 2.0, 3.0]))


def test_ensure_vector_accepts_column_vector_and_rejects_matrix():
    vec = _ensure_vector([[1], [2], [3]])
    assert vec.shape == (3,)

    with pytest.raises(ValueError):
        _ensure_vector([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        _ensure_vector(np.zeros((2, 2, 1)))


class DummyRegressor(BaseEstimatorLike, FittedMixin, RegressorMixinLike):
    def __init__(self):
        self.predictions = None

    def fit(self, X, y):
        self._mark_fitted()
        self.predictions = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        n = np.asarray(X).shape[0]
        if self.predictions is None:
            raise RuntimeError("Model has no predictions configured")
        return np.asarray(self.predictions[:n], dtype=float)


def test_regressor_mixin_score_handles_perfect_and_imperfect_fit():
    reg = DummyRegressor()

    with pytest.raises(NotFittedError):
        check_is_fitted(reg)

    reg.fit([[0], [1]], [1, 2])
    reg.predictions = np.array([1, 2])
    assert reg.score([[0], [1]], [1, 2]) == 1.0

    reg.predictions = np.array([1, 0])
    assert reg.score([[0], [1]], [1, 2]) < 1.0

    with pytest.raises(ValueError, match="different shape"):
        reg.score([[0], [1]], [1])


def test_regressor_mixin_score_returns_zero_for_constant_target():
    reg = DummyRegressor()
    reg.fit([[0], [1]], [1, 1])
    reg.predictions = np.array([1, 1])
    assert reg.score([[0], [1]], [1, 1]) == 0.0


class DummyClassifier(FittedMixin, ClassifierMixinLike):
    def __init__(self, predictions):
        self._predictions = np.asarray(predictions)

    def predict(self, X):
        n = np.asarray(X).shape[0]
        return self._predictions[:n]


def test_classifier_mixin_score_validation_behaviour():
    clf = DummyClassifier([0, 1])
    clf._mark_fitted()
    with pytest.raises(ValueError, match="different shape"):
        clf.score([[0], [1]], [0])

    empty_clf = DummyClassifier([])
    empty_clf._mark_fitted()
    assert empty_clf.score(np.empty((0, 2)), np.array([])) == 0.0

    matched_clf = DummyClassifier([0, 1])
    matched_clf._mark_fitted()
    assert matched_clf.score([[0], [1]], np.array([0, 1])) == 1.0
