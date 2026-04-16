"""Binary logistic regression trained with batch gradient descent."""

from __future__ import annotations

import numpy as np


class LogisticRegression:
    """Binary logistic regression classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size used by gradient descent.
    n_iters : int, default=1000
        Number of optimization iterations.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        """Initialize hyperparameters and learned attributes."""
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        """Compute the sigmoid function element-wise."""
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X, y):
        """Fit model parameters with batch gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary targets encoded as 0 and 1.

        Returns
        -------
        LogisticRegression
            Fitted estimator.
        """
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.n_iters <= 0:
            raise ValueError("n_iters must be a positive integer.")

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array of binary labels.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample.")

        unique_values = np.unique(y_arr)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError("y must contain only binary values: 0 and 1.")

        n_samples, n_features = X_arr.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iters):
            logits = X_arr @ self.weights_ + self.bias_
            probs = self._sigmoid(logits)

            error = probs - y_arr
            dw = (X_arr.T @ error) / n_samples
            db = np.mean(error)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        """Predict probabilities for the positive class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Predicted probabilities for class 1.
        """
        if self.weights_ is None or self.bias_ is None:
            raise ValueError("Model must be fitted before calling predict_proba.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X_arr.shape[1] != self.weights_.shape[0]:
            raise ValueError("X must have the same number of features used during fit.")

        linear_output = X_arr @ self.weights_ + self.bias_
        return self._sigmoid(linear_output)

    def predict(self, X):
        """Predict binary labels using a 0.5 probability threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
