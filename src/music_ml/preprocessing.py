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


class StandardScaler:
    """Scale features to zero mean and unit variance.

    The scaler learns column-wise means and standard deviations from training
    data. Columns with zero variance are left unchanged during scaling.
    """

    def __init__(self):
        """Initialize an unfitted scaler."""
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Compute per-feature mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data used to estimate scaling statistics.

        Returns
        -------
        StandardScaler
            The fitted scaler instance.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X_arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        self.mean_ = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0)
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X):
        """Scale data using previously computed statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to scale.

        Returns
        -------
        numpy.ndarray
            Scaled data array.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before calling transform.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X_arr.shape[1] != self.mean_.shape[0]:
            raise ValueError("X must have the same number of features used in fit.")

        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit the scaler to ``X`` and return the transformed data."""
        return self.fit(X).transform(X)
