"""Basic binary classification metrics implemented with NumPy."""

from __future__ import annotations

import numpy as np


def _validate_binary_inputs(y_true, y_pred):
    """Validate and coerce binary label arrays."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true and y_pred must contain at least one sample.")

    if not np.all(np.isin(np.unique(y_true_arr), [0, 1])):
        raise ValueError("y_true must contain only binary labels: 0 and 1.")
    if not np.all(np.isin(np.unique(y_pred_arr), [0, 1])):
        raise ValueError("y_pred must contain only binary labels: 0 and 1.")

    return y_true_arr.astype(int), y_pred_arr.astype(int)


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix for binary labels.

    Returns a 2x2 matrix in the order:
    [[TN, FP],
     [FN, TP]]
    """
    y_true_arr, y_pred_arr = _validate_binary_inputs(y_true, y_pred)

    tn = np.sum((y_true_arr == 0) & (y_pred_arr == 0))
    fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
    tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy_score(y_true, y_pred):
    """Compute classification accuracy."""
    y_true_arr, y_pred_arr = _validate_binary_inputs(y_true, y_pred)
    return float(np.mean(y_true_arr == y_pred_arr))


def precision_score(y_true, y_pred):
    """Compute precision for the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    denominator = tp + fp
    return 0.0 if denominator == 0 else float(tp / denominator)


def recall_score(y_true, y_pred):
    """Compute recall for the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    denominator = tp + fn
    return 0.0 if denominator == 0 else float(tp / denominator)


def f1_score(y_true, y_pred):
    """Compute F1 score for the positive class."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    denominator = precision + recall
    return 0.0 if denominator == 0 else float(2 * precision * recall / denominator)
