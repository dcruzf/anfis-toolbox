"""Tests for the clustering module (Fuzzy C-Means)."""

import numpy as np
import pytest

from anfis_toolbox import FuzzyCMeans


def make_two_blobs(n_per=30, d=2, locs=(-3.0, 3.0), scale=0.2, seed=0):
    rng = np.random.RandomState(seed)
    X1 = rng.normal(loc=locs[0], scale=scale, size=(n_per, d))
    X2 = rng.normal(loc=locs[1], scale=scale, size=(n_per, d))
    return np.vstack([X1, X2])


def test_fcm_fit_predict_shapes_and_membership():
    X = make_two_blobs()
    fcm = FuzzyCMeans(n_clusters=2, m=2.0, max_iter=200, tol=1e-5, random_state=42)
    fcm.fit(X)

    # Shapes
    assert fcm.cluster_centers_.shape == (2, 2)
    assert fcm.membership_.shape == (X.shape[0], 2)

    # Membership is probabilistic per row
    U = fcm.membership_
    assert np.all(U >= 0.0)
    assert np.allclose(U.sum(axis=1), 1.0)

    # Predictions
    labels = fcm.predict(X)
    assert labels.shape == (X.shape[0],)
    assert set(np.unique(labels)).issubset({0, 1})

    # Metrics
    pc = fcm.partition_coefficient()
    ce = fcm.classification_entropy()
    xb = fcm.xie_beni_index(X)
    assert np.isfinite(pc) and 1.0 / 2.0 <= pc <= 1.0
    assert np.isfinite(ce) and ce >= 0.0
    assert np.isfinite(xb) and xb > 0.0


def test_fcm_predict_proba_before_fit_raises():
    X = np.array([[0.0], [1.0]])
    fcm = FuzzyCMeans(n_clusters=2, random_state=0)
    with pytest.raises(RuntimeError):
        fcm.predict_proba(X)


def test_fcm_determinism_with_random_state():
    X = make_two_blobs(n_per=20, d=2, seed=123)
    f1 = FuzzyCMeans(n_clusters=2, m=2.0, max_iter=200, tol=1e-6, random_state=7)
    f2 = FuzzyCMeans(n_clusters=2, m=2.0, max_iter=200, tol=1e-6, random_state=7)
    f1.fit(X)
    f2.fit(X)

    # Same seed -> identical init and deterministic updates -> identical results (up to tolerance)
    assert np.allclose(f1.cluster_centers_, f2.cluster_centers_)
    assert np.allclose(f1.membership_, f2.membership_)


def test_fcm_1d_input_and_transform_alias():
    # 1D input should be accepted and reshaped internally
    X = np.array([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0])
    fcm = FuzzyCMeans(n_clusters=2, random_state=0)
    fcm.fit(X)
    U1 = fcm.predict_proba(X)
    U2 = fcm.transform(X)
    assert U1.shape == (X.shape[0], 2)
    assert np.allclose(U1, U2)
    assert np.allclose(U1.sum(axis=1), 1.0)
    labels = fcm.predict(X)
    assert labels.shape == (X.shape[0],)


def test_fcm_invalid_params_and_sample_count():
    with pytest.raises(ValueError):
        FuzzyCMeans(n_clusters=1)
    with pytest.raises(ValueError):
        FuzzyCMeans(n_clusters=2, m=1.0)

    # n_samples < n_clusters should raise on fit
    X = np.array([[0.0], [1.0]])  # n=2
    fcm = FuzzyCMeans(n_clusters=3)
    with pytest.raises(ValueError):
        fcm.fit(X)


def test_fcm_zero_distance_behavior():
    # Fit on simple 1D clusters; then evaluate exactly at learned centers
    X = np.concatenate([np.full(20, -2.0), np.full(20, 2.0)])
    fcm = FuzzyCMeans(n_clusters=2, random_state=0)
    fcm.fit(X)
    C = fcm.cluster_centers_

    # For each center, the membership for that point should be very close to 1 for some cluster
    for i in range(C.shape[0]):
        u = fcm.predict_proba(C[i : i + 1])  # shape (1, k)
        assert np.allclose(u.sum(axis=1), 1.0)
        assert np.max(u) > 0.99


def test_fcm_checkX_invalid_dims_fit_and_predict_proba():
    # 3D input should raise in _check_X
    fcm = FuzzyCMeans(n_clusters=2)
    X3 = np.zeros((5, 2, 2))
    with pytest.raises(ValueError, match="X must be 1D or 2D"):
        fcm.fit(X3)

    # After a valid fit, passing 3D to predict_proba also raises
    X = np.array([[-1.0], [1.0], [2.0]])
    fcm.fit(X)
    with pytest.raises(ValueError, match="X must be 1D or 2D"):
        fcm.predict_proba(X3)


def test_fcm_metrics_before_fit_raise():
    fcm = FuzzyCMeans(n_clusters=2)
    with pytest.raises(RuntimeError):
        fcm.partition_coefficient()
    with pytest.raises(RuntimeError):
        fcm.classification_entropy()
    with pytest.raises(RuntimeError):
        fcm.xie_beni_index(np.array([[0.0], [1.0]]))


def test_fcm_fit_no_break_path_via_max_iter_one():
    # Use data that requires updates but limit max_iter to 1 to avoid break
    X = np.array([[-2.0], [-1.9], [2.0], [2.1]])
    fcm = FuzzyCMeans(n_clusters=2, max_iter=1, tol=1e-12, random_state=0)
    fcm.fit(X)
    assert fcm.cluster_centers_.shape == (2, 1)
    assert fcm.membership_.shape == (X.shape[0], 2)


def test_fcm_fit_predict_covers_path_and_sets_attrs():
    # Ensure fit_predict executes and sets learned attributes
    X = make_two_blobs(n_per=10, d=2, seed=99)
    fcm = FuzzyCMeans(n_clusters=2, random_state=0)
    labels = fcm.fit_predict(X)
    assert labels.shape == (X.shape[0],)
    # Attributes populated
    assert fcm.cluster_centers_ is not None
    assert fcm.membership_ is not None
