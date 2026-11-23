"""Small reusable estimators (importable so joblib-serialised pipelines can be loaded by the API)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class WinsorisedRegressor(BaseEstimator, RegressorMixin):
    """Fits ``base`` on a target winsorised at the TRAIN-fold quantile; predictions stay on the raw scale.

    Keeps a handful of extreme resolution times (up to ~900h) from dominating squared-error fits while the
    model is still *evaluated* against the raw target.
    """

    def __init__(self, base=None, upper_quantile: float = 0.999):
        self.base = base
        self.upper_quantile = upper_quantile

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.cap_ = float(np.quantile(y, self.upper_quantile))
        self.model_ = clone(self.base).fit(X, np.minimum(y, self.cap_))
        return self

    def predict(self, X):
        return self.model_.predict(X)


class MedianRegressor(BaseEstimator, RegressorMixin):
    """Predict the training median (optimal constant for MAE)."""

    def __init__(self, stat: str = "median"):
        self.stat = stat

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.value_ = float(np.median(y) if self.stat == "median" else np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)
