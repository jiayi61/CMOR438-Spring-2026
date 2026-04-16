"""KMeans clustering implemented from scratch with NumPy."""

from __future__ import annotations

import numpy as np


class KMeans:
    """KMeans clustering with random centroid initialization.

    Parameters
    ----------
    n_clusters : int, default=4
        Number of clusters to form.
    max_iters : int, default=100
        Maximum number of update iterations.
    tol : float, default=1e-4
        Convergence tolerance on centroid movement (L2 norm).
    random_state : int or None, default=None
        Seed for reproducible random initialization.
    """

    def __init__(self, n_clusters=4, max_iters=100, tol=1e-4, random_state=None):
        """Initialize hyperparameters and learned attributes."""
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _validate_hyperparameters(self):
        """Validate clustering hyperparameters."""
        if not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if not isinstance(self.max_iters, (int, np.integer)) or self.max_iters <= 0:
            raise ValueError("max_iters must be a positive integer.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

    @staticmethod
    def _compute_distances(X, centroids):
        """Compute pairwise Euclidean distances to each centroid."""
        return np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

    def fit(self, X):
        """Fit KMeans clustering on data matrix ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.

        Returns
        -------
        KMeans
            Fitted clustering estimator.
        """
        self._validate_hyperparameters()

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        n_samples, _ = X_arr.shape
        if n_samples == 0:
            raise ValueError("X must contain at least one sample.")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples.")

        rng = np.random.default_rng(self.random_state)
        initial_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        self.centroids_ = X_arr[initial_indices].copy()

        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        for iteration in range(1, self.max_iters + 1):
            distances = self._compute_distances(X_arr, self.centroids_)
            labels = np.argmin(distances, axis=1)

            new_centroids = self.centroids_.copy()
            for cluster_idx in range(self.n_clusters):
                cluster_points = X_arr[labels == cluster_idx]
                if cluster_points.shape[0] > 0:
                    new_centroids[cluster_idx] = cluster_points.mean(axis=0)

            centroid_shift = np.linalg.norm(new_centroids - self.centroids_)
            self.centroids_ = new_centroids
            self.labels_ = labels
            self.n_iter_ = iteration

            if centroid_shift <= self.tol:
                break

        final_distances = self._compute_distances(X_arr, self.centroids_)
        closest_distances = final_distances[np.arange(n_samples), self.labels_]
        self.inertia_ = float(np.sum(closest_distances**2))

        return self

    def predict(self, X):
        """Assign each sample in ``X`` to the nearest learned centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data points to cluster.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Cluster assignments as integer labels.
        """
        if self.centroids_ is None:
            raise ValueError("KMeans must be fitted before calling predict.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X_arr.shape[1] != self.centroids_.shape[1]:
            raise ValueError("X must have the same number of features used during fit.")

        distances = self._compute_distances(X_arr, self.centroids_)
        return np.argmin(distances, axis=1)
