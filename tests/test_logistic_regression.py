"""Unit tests for the custom LogisticRegression classifier."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_ml.supervised import LogisticRegression


def test_fit_sets_weight_shape_correctly():
    """Model should learn one weight per input feature."""
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iters=200)
    model.fit(X, y)

    assert model.weights_.shape == (X.shape[1],)
    assert isinstance(model.bias_, float)


def test_predict_proba_outputs_values_between_zero_and_one():
    """Predicted probabilities should lie in [0, 1]."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iters=300)
    model.fit(X, y)
    probs = model.predict_proba(X)

    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_predict_outputs_binary_labels():
    """Predictions should only contain class labels 0 and 1."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iters=300)
    model.fit(X, y)
    preds = model.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})


def test_model_reaches_high_accuracy_on_linearly_separable_data():
    """On a simple separable dataset, training accuracy should be high."""
    X = np.array(
        [
            [-3.0, -2.0],
            [-2.0, -1.0],
            [-1.5, -1.0],
            [1.0, 1.0],
            [2.0, 1.5],
            [3.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iters=2000)
    model.fit(X, y)
    preds = model.predict(X)
    accuracy = np.mean(preds == y)

    assert accuracy >= 0.95


def test_invalid_input_shapes_raise_value_error():
    """Invalid training/inference input shapes should fail clearly."""
    model = LogisticRegression()

    X_valid = np.array([[0.0, 1.0], [1.0, 0.0]])
    y_valid = np.array([0, 1])

    with pytest.raises(ValueError):
        model.fit(np.array([1.0, 2.0]), y_valid)

    with pytest.raises(ValueError):
        model.fit(X_valid, np.array([[0], [1]]))

    with pytest.raises(ValueError):
        model.fit(X_valid, np.array([0, 1, 0]))

    model.fit(X_valid, y_valid)

    with pytest.raises(ValueError):
        model.predict_proba(np.array([1.0, 2.0]))
