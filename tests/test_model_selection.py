import numpy as np
import pytest

import anfis_toolbox.model_selection as ms
from anfis_toolbox.model_selection import KFold, train_test_split


def test_kfold_basic_no_shuffle_sizes_and_indices():
    X = np.arange(10)
    kf = KFold(n_splits=3, shuffle=False)
    splits = list(kf.split(X))
    # Fold sizes ~ 4,3,3
    sizes = [len(te) for _, te in splits]
    assert sizes == [4, 3, 3]
    # Ensure no overlap and full coverage
    all_test = np.concatenate([te for _, te in splits])
    assert np.array_equal(np.sort(all_test), np.arange(10))


def test_kfold_shuffle_with_random_state_int_and_instance():
    X = np.arange(12)
    # With int seed
    kf1 = KFold(n_splits=4, shuffle=True, random_state=123)
    splits1 = list(kf1.split(X))
    # With RandomState instance
    rs = np.random.RandomState(123)
    kf2 = KFold(n_splits=4, shuffle=True, random_state=rs)
    splits2 = list(kf2.split(X))
    # Deterministic
    for (tr1, te1), (tr2, te2) in zip(splits1, splits2, strict=False):
        assert np.array_equal(te1, te2)
        assert np.array_equal(tr1, tr2)


def test_kfold_shuffle_with_none_seed_covers_all():
    X = np.arange(9)
    kf = KFold(n_splits=3, shuffle=True)  # random_state=None path
    splits = list(kf.split(X))
    all_test = np.concatenate([te for _, te in splits])
    # Should be a permutation covering all indices
    assert np.array_equal(np.sort(all_test), np.arange(9))


def test_kfold_invalid_splits():
    with pytest.raises(ValueError):
        KFold(n_splits=1)


def test_train_test_split_default_fraction_and_alignment():
    X = np.arange(20)
    y = X * 2
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
    # Default test_size=0.25 -> 5 test, 15 train
    assert len(X_te) == 5 and len(X_tr) == 15
    # Alignment: y should be 2*x for corresponding indices
    assert np.array_equal(y_tr, X_tr * 2)
    assert np.array_equal(y_te, X_te * 2)


def test_train_test_split_shuffle_false_order():
    X = np.arange(8)
    X_tr, X_te = train_test_split(X, test_size=3, shuffle=False)
    # No shuffle: test is first 3, train is remaining
    assert np.array_equal(X_te, np.arange(3))
    assert np.array_equal(X_tr, np.arange(3, 8))


def test_train_test_split_float_and_int_sizes_and_only_one_provided():
    X = np.arange(10)
    # float test size
    X_tr, X_te = train_test_split(X, test_size=0.3, random_state=0)
    assert len(X_te) == int(np.ceil(10 * 0.3))
    # int train size only
    X_tr2, X_te2 = train_test_split(X, train_size=7, random_state=0)
    assert len(X_tr2) == 7 and len(X_te2) == 3


def test_train_test_split_multiple_arrays_and_types():
    X = np.arange(12)
    y = (X % 3).tolist()  # list type array
    z = np.vstack([X, X]).T  # 2D array
    X_tr, X_te, y_tr, y_te, z_tr, z_te = train_test_split(X, y, z, test_size=4, random_state=42)
    assert len(X_tr) == 8 and len(X_te) == 4
    assert np.array_equal(y_tr, (X_tr % 3))
    assert np.array_equal(y_te, (X_te % 3))
    assert z_tr.shape == (8, 2) and z_te.shape == (4, 2)


def test_train_test_split_with_randomstate_instance():
    X = np.arange(10)
    rs = np.random.RandomState(7)
    X_tr1, X_te1 = train_test_split(X, test_size=4, random_state=rs)
    # Repeat with same rs should give same split due to deterministic RNG
    rs2 = np.random.RandomState(7)
    X_tr2, X_te2 = train_test_split(X, test_size=4, random_state=rs2)
    assert np.array_equal(X_tr1, X_tr2)
    assert np.array_equal(X_te1, X_te2)


def test_train_test_split_errors_and_edge_cases():
    # No arrays
    with pytest.raises(ValueError):
        train_test_split()
    # Mismatched lengths
    with pytest.raises(ValueError):
        train_test_split(np.arange(3), np.arange(4))
    # Invalid test_size float bounds
    with pytest.raises(ValueError):
        train_test_split(np.arange(5), test_size=0.0)
    with pytest.raises(ValueError):
        train_test_split(np.arange(5), test_size=1.0)
    # Invalid types for sizes
    with pytest.raises(TypeError):
        train_test_split(np.arange(5), test_size="bad")
    with pytest.raises(TypeError):
        train_test_split(np.arange(5), train_size="bad")
    # Derived invalid sizes
    with pytest.raises(ValueError):
        train_test_split(np.arange(2), test_size=1, train_size=2)
    # _get_n_samples TypeError via non-sequence
    with pytest.raises(TypeError):
        train_test_split(42)  # not a sequence
    # _validate_split_sizes direct coverage: both None error
    with pytest.raises(ValueError):
        ms._validate_split_sizes(10, None, None)
    # _validate_split_sizes invalid train float bounds
    with pytest.raises(ValueError):
        ms._validate_split_sizes(10, 0.3, 1.0)
    # Invalid combination exceeding n_samples
    with pytest.raises(ValueError):
        ms._validate_split_sizes(5, 4, 2)
    # Invalid zero sizes
    with pytest.raises(ValueError):
        ms._validate_split_sizes(5, 0, 3)
    with pytest.raises(ValueError):
        ms._validate_split_sizes(5, 3, 0)

    # train_size as float path (floor rounding) -> covers line 172
    test_n, train_n = ms._validate_split_sizes(7, None, 0.6)
    assert (test_n, train_n) == (3, 4)  # floor(7*0.6)=4, rest=3

    # end-to-end: train_size float without shuffle keeps order; test slice comes first
    X = np.arange(10)
    X_train, X_test = ms.train_test_split(X, train_size=0.55, shuffle=False)
    # floor(10*0.55)=5 train, remainder=5 test
    assert X_test.tolist() == [0, 1, 2, 3, 4]
    assert X_train.tolist() == [5, 6, 7, 8, 9]
    # Direct branch coverage: test_n derived when None
    t, tr = ms._validate_split_sizes(10, None, 3)
    assert (t, tr) == (7, 3)
    # Direct branch coverage: train_n derived when None
    t2, tr2 = ms._validate_split_sizes(10, 2, None)
    assert (t2, tr2) == (2, 8)
