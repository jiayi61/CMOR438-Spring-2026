"""Preprocessing utilities for splitting data and feature scaling."""

from __future__ import annotations

import numpy as np


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """Split arrays into train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target values aligned with ``X``.
    test_size : float, default=0.2
        Fraction of samples to place in the test split. Must satisfy
        ``0 < test_size < 1``.
    shuffle : bool, default=True
        Whether to shuffle sample order before splitting.
    random_state : int or None, default=None
        Seed used for reproducible shuffling when ``shuffle=True``.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of numpy.ndarray
        Train/test partitions preserving row alignment between features and
        targets.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if y_arr.ndim not in (1, 2):
        raise ValueError("y must be a 1D or 2D array with n_samples rows.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if not isinstance(test_size, (float, np.floating)):
        raise TypeError("test_size must be a float between 0 and 1.")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    n_samples = X_arr.shape[0]
    n_test = int(np.ceil(n_samples * float(test_size)))
    n_test = min(max(n_test, 1), n_samples - 1)

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X_arr[train_indices]
    X_test = X_arr[test_indices]
    y_train = y_arr[train_indices]
    y_test = y_arr[test_indices]

    return X_train, X_test, y_train, y_test
