"""Unit tests for preprocessing utilities."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_ml.preprocessing import StandardScaler, train_test_split


def test_train_test_split_returns_correct_sizes():
    """Split outputs should match requested test proportion and input shapes."""
    X = np.arange(50).reshape(10, 5)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    assert X_train.shape == (7, 5)
    assert X_test.shape == (3, 5)
    assert y_train.shape == (7,)
    assert y_test.shape == (3,)


def test_train_test_split_reproducible_with_fixed_random_state():
    """Shuffled splits should be identical with the same random seed."""
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)

    split_1 = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=123)
    split_2 = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=123)

    for arr_1, arr_2 in zip(split_1, split_2):
        np.testing.assert_array_equal(arr_1, arr_2)


def test_standard_scaler_transform_has_zero_mean_columns():
    """Transformed feature columns should be approximately zero-centered."""
    X = np.array(
        [
            [1.0, 10.0, 3.0],
            [2.0, 20.0, 6.0],
            [3.0, 30.0, 9.0],
            [4.0, 40.0, 12.0],
        ]
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(X_scaled.mean(axis=0), np.zeros(3), atol=1e-12)


def test_standard_scaler_handles_zero_variance_columns():
    """Constant-valued columns should be transformed to zeros without errors."""
    X = np.array(
        [
            [5.0, 1.0],
            [5.0, 2.0],
            [5.0, 3.0],
        ]
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(X_scaled[:, 0], np.zeros(3), atol=1e-12)
    assert np.isfinite(X_scaled).all()
