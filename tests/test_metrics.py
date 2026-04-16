"""Unit tests for binary classification metrics."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_ml.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def test_accuracy_score_binary_example():
    """Accuracy should match simple hand-calculated result."""
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1])

    assert accuracy_score(y_true, y_pred) == 4 / 6


def test_precision_score_binary_example():
    """Precision should be TP / (TP + FP)."""
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1])

    assert precision_score(y_true, y_pred) == 2 / 3


def test_recall_score_binary_example():
    """Recall should be TP / (TP + FN)."""
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1])

    assert recall_score(y_true, y_pred) == 2 / 3


def test_f1_score_binary_example():
    """F1 should be harmonic mean of precision and recall."""
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1])

    assert f1_score(y_true, y_pred) == 2 / 3


def test_confusion_matrix_exact_output():
    """Confusion matrix should match exact TN, FP, FN, TP counts."""
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1])

    expected = np.array([[2, 1], [1, 2]])
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)
