"""Unit tests for the custom KMeans clustering implementation."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_ml.unsupervised import KMeans


def test_fit_sets_centroids_shape():
    """Centroids should have one row per cluster and one column per feature."""
    X = np.array([[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.2, 5.1]])

    model = KMeans(n_clusters=2, random_state=0)
    model.fit(X)

    assert model.centroids_.shape == (2, 2)


def test_labels_length_matches_sample_count():
    """Each training sample should receive one cluster label."""
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])

    model = KMeans(n_clusters=2, random_state=0)
    model.fit(X)

    assert model.labels_.shape == (X.shape[0],)


def test_inertia_is_nonnegative():
    """Within-cluster sum of squares cannot be negative."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [4.0, 4.0], [5.0, 5.0]])

    model = KMeans(n_clusters=2, random_state=2)
    model.fit(X)

    assert model.inertia_ >= 0.0


def test_reproducible_with_fixed_random_state():
    """Fixed random_state should produce repeatable clustering results."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [4.8, 5.1],
            [5.0, 4.9],
            [10.0, 10.0],
            [10.2, 9.9],
        ]
    )

    model_1 = KMeans(n_clusters=3, random_state=42)
    model_2 = KMeans(n_clusters=3, random_state=42)
    model_1.fit(X)
    model_2.fit(X)

    np.testing.assert_allclose(model_1.centroids_, model_2.centroids_)
    np.testing.assert_array_equal(model_1.labels_, model_2.labels_)
    assert model_1.inertia_ == model_2.inertia_


def test_identifies_structure_on_separated_blobs():
    """KMeans should separate clearly distinct clusters in simple synthetic data."""
    blob_a = np.array([[-3.0, -3.0], [-3.2, -2.9], [-2.8, -3.1]])
    blob_b = np.array([[3.0, 3.0], [3.1, 2.8], [2.9, 3.2]])
    X = np.vstack([blob_a, blob_b])

    model = KMeans(n_clusters=2, random_state=1, max_iters=200)
    labels = model.fit(X).labels_

    left_majority = np.mean(labels[:3] == labels[0])
    right_majority = np.mean(labels[3:] == labels[3])

    assert left_majority == 1.0
    assert right_majority == 1.0
    assert labels[0] != labels[3]
