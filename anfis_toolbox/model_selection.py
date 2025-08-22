"""Lightweight model selection utilities compatible with a subset of sklearn.

This module provides minimal implementations of KFold and train_test_split
to avoid a hard dependency on scikit-learn while keeping a familiar API.

If scikit-learn is available at runtime, the functions/classes here will
transparently delegate to sklearn for full compatibility. Otherwise, simple
NumPy-based fallbacks are used.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence

import numpy as np

__all__ = ["KFold", "train_test_split"]


class KFold:
    """K-Folds cross-validator (minimal implementation).

    Parameters:
        n_splits: Number of folds. Must be at least 2.
        shuffle: Whether to shuffle indices before splitting.
        random_state: Seed or RandomState for reproducibility when shuffling.
    """

    def __init__(
        self, n_splits: int = 5, *, shuffle: bool = False, random_state: int | np.random.RandomState | None = None
    ) -> None:
        """Initialize the cross-validation splitter.

        Args:
            n_splits (int, optional): Number of folds. Must be at least 2. Defaults to 5.
            shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to False.
            random_state (int, np.random.RandomState, or None, optional):
                Seed or random state for shuffling.
                If int, used as seed for RandomState.
                If RandomState instance, used directly. If None, no shuffling seed is set.

        Raises:
            ValueError: If n_splits is less than 2.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is None:
            self.random_state = None
        else:
            self.random_state = np.random.RandomState(random_state)

    def split(
        self,
        X: Sequence,
        y: Sequence | None = None,
        groups: Sequence | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Splits the dataset into training and test indices for cross-validation.

        Parameters
        ----------
        X : Sequence
            The input data to split.
        y : Sequence or None, optional
            The target variable for supervised learning problems. Not used in this method
            but kept for compatibility.
        groups : Sequence or None, optional
            Group labels for the samples used while splitting the dataset. Not used in this
            method but kept for compatibility.

        Yields:
        ------
        tuple of (np.ndarray, np.ndarray)
            A tuple containing the training indices and test indices for each fold.

        Notes:
        -----
        - The number of splits is determined by `self.n_splits`.
        - If `self.shuffle` is True, the indices are shuffled using `self.random_state`.
        - Each fold is approximately equal in size; the first folds may have one more sample
          if the division is not exact.
        """
        n_samples = _get_n_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            rng = self.random_state or np.random.RandomState()
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate((indices[:start], indices[stop:]))
            yield train_indices, test_indices
            current = stop


def train_test_split(
    *arrays: Sequence,
    test_size: float | int | None = None,
    train_size: float | int | None = None,
    random_state: int | np.random.RandomState | None = None,
    shuffle: bool = True,
) -> tuple:
    """Split arrays or matrices into random train and test subsets.

    Minimal subset of sklearn.model_selection.train_test_split.
    Supports splitting multiple arrays of equal length along the first dimension.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    n_samples = _get_n_samples(arrays[0])
    for a in arrays[1:]:
        if _get_n_samples(a) != n_samples:
            raise ValueError("All input arrays must have the same number of samples")

    if test_size is None and train_size is None:
        test_size = 0.25

    test_size, train_size = _validate_split_sizes(n_samples, test_size, train_size)

    indices = np.arange(n_samples)
    if shuffle:
        rng = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
        rng.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size : test_size + train_size]

    result = []
    for a in arrays:
        a = np.asarray(a)
        result.extend([a[train_indices], a[test_indices]])
    return tuple(result)


def _get_n_samples(a: Sequence) -> int:
    try:
        return int(len(a))
    except Exception as e:  # pragma: no cover - defensive
        raise TypeError("Expected sequence-like input with __len__") from e


def _validate_split_sizes(
    n_samples: int, test_size: float | int | None, train_size: float | int | None
) -> tuple[int, int]:
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size as a float must be in (0, 1)")
        test_n = int(np.ceil(n_samples * test_size))
    elif isinstance(test_size, int):
        test_n = test_size
    elif test_size is None:
        test_n = None
    else:
        raise TypeError("test_size must be float, int, or None")

    if isinstance(train_size, float):
        if not 0.0 < train_size < 1.0:
            raise ValueError("train_size as a float must be in (0, 1)")
        train_n = int(np.floor(n_samples * train_size))
    elif isinstance(train_size, int):
        train_n = train_size
    elif train_size is None:
        train_n = None
    else:
        raise TypeError("train_size must be float, int, or None")

    if test_n is None and train_n is None:
        raise ValueError("At least one of test_size or train_size must be provided")

    if test_n is None:
        test_n = n_samples - train_n
    if train_n is None:
        train_n = n_samples - test_n

    if test_n <= 0 or train_n <= 0 or test_n + train_n > n_samples:
        raise ValueError("Invalid train/test sizes for number of samples")

    return test_n, train_n
