from __future__ import annotations
import numpy as np
from src.base import BaseClassifier


class KNN(BaseClassifier):
    def __init__(
        self,
        k=15,
        metric="euclidean",
        weighted=False,
        batch_size=512,
        p=3.0,
    ):
        if int(k) <= 0:
            raise ValueError("k must be positive")
        if metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError("metric must be euclidean, manhattan, or minkowski")
        if metric == "minkowski" and float(p) < 1:
            raise ValueError("Minkowski p must be at least 1")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        self.k = int(k)
        self.metric = metric
        self.weighted = bool(weighted)
        self.batch_size = int(batch_size)
        self.p = float(p)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(X) == 0:
            raise ValueError("Cannot fit KNN on an empty dataset")
        if not np.isfinite(X).all():
            raise ValueError("KNN requires finite features; use matrix preprocessing first")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("KNN labels must be binary 0/1")
        if self.k > len(X):
            raise ValueError(f"k={self.k} exceeds n_train={len(X)}")
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (len(y),):
                raise ValueError("sample_weight must have shape (n_samples,)")
            if not np.isfinite(sample_weight).all() or (sample_weight < 0).any():
                raise ValueError("sample_weight must be finite and non-negative")
            if sample_weight.sum() == 0:
                raise ValueError("sample_weight must contain positive total weight")
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.sample_weight_ = sample_weight.copy()
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if not hasattr(self, "X_"):
            raise RuntimeError("Call fit before predict_proba")
        query = np.asarray(X, dtype=float)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        if not np.isfinite(query).all():
            raise ValueError("KNN requires finite features; use matrix preprocessing first")

        probabilities = np.empty(len(query), dtype=float)
        for start in range(0, len(query), self.batch_size):
            stop = min(start + self.batch_size, len(query))
            distances = self._distances(query[start:stop])
            indices = np.argpartition(distances, self.k - 1, axis=1)[:, : self.k]
            nearest_distances = np.take_along_axis(distances, indices, axis=1)
            
            order = np.argsort(nearest_distances, axis=1, kind="mergesort")
            indices = np.take_along_axis(indices, order, axis=1)
            nearest_distances = np.take_along_axis(nearest_distances, order, axis=1)
            labels = self.y_[indices]
            weights = self.sample_weight_[indices]

            if self.weighted:
                weights = weights / (nearest_distances + 1e-12)
            positive_weight = (weights * labels).sum(axis=1)
            total_weight = weights.sum(axis=1)
            probabilities[start:stop] = np.divide(
                positive_weight,
                total_weight,
                out=np.zeros(stop - start, dtype=float),
                where=total_weight > 0,
            )
        return np.column_stack([1.0 - probabilities, probabilities])

    def _distances(self, query):
        if self.metric == "euclidean":
            train_norm = (self.X_**2).sum(axis=1)[None, :]
            query_norm = (query**2).sum(axis=1)[:, None]
            squared = query_norm + train_norm - 2.0 * query @ self.X_.T
            return np.sqrt(np.maximum(squared, 0.0))

        power = 1.0 if self.metric == "manhattan" else self.p
        distances = np.zeros((len(query), len(self.X_)), dtype=float)
        for feature in range(self.n_features_in_):
            distances += np.abs(query[:, feature, None] - self.X_[None, :, feature]) ** power
        return distances if power == 1.0 else distances ** (1.0 / power)

