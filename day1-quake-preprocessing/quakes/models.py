"""Part 2, Task B: an sklearn-style estimator and model factories."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from .transform import build_preprocessor


class GDLinearRegression(RegressorMixin, BaseEstimator):
    """Linear regression trained with vectorized batch gradient descent."""

    def __init__(self, lr: float = 0.1, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        if len(X) == 0:
            raise ValueError("Cannot fit on an empty dataset")
        if self.lr <= 0 or self.n_iter < 1:
            raise ValueError("lr must be positive and n_iter must be at least 1")

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0
        self.loss_history_ = []
        for _ in range(self.n_iter):
            errors = X @ self.coef_ + self.intercept_ - y
            self.loss_history_.append(float(np.mean(errors ** 2)))
            grad_w = (2.0 / n_samples) * (X.T @ errors)
            grad_b = 2.0 * float(np.mean(errors))
            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * grad_b
        final_errors = X @ self.coef_ + self.intercept_ - y
        self.loss_history_.append(float(np.mean(final_errors ** 2)))
        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("This GDLinearRegression instance is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be 2D and have the same feature count used in fit")
        return X @ self.coef_ + self.intercept_


def make_regressor(kind: str = "linear", degree: int = 2, alpha: float = 1.0):
    """Build a preprocessing/model pipeline for regression."""
    if degree < 1:
        raise ValueError("degree must be at least 1")
    if kind == "dummy":
        estimator = DummyRegressor(strategy="mean")
        prep_degree = 1
    elif kind == "linear":
        estimator = LinearRegression()
        prep_degree = 1
    elif kind == "ridge":
        estimator = Ridge(alpha=alpha)
        prep_degree = 1
    elif kind == "poly":
        estimator = Ridge(alpha=alpha)
        prep_degree = degree
    else:
        raise ValueError(f"Unknown regressor kind {kind!r}; choose dummy, linear, ridge, or poly")
    return Pipeline([
        ("prep", build_preprocessor(degree=prep_degree)),
        ("model", estimator),
    ])


def make_classifier(kind: str = "logistic", C: float = 1.0, class_weight=None):
    """Build a preprocessing/model pipeline for binary classification."""
    if kind == "dummy":
        estimator = DummyClassifier(strategy="prior")
    elif kind == "logistic":
        estimator = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)
    else:
        raise ValueError(f"Unknown classifier kind {kind!r}; choose dummy or logistic")
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])
