import numpy as np
from src.base import BaseClassifier
from src.validation import binary_data, prediction_data, sample_weights, positive_int, finite_scalar


class KNN(BaseClassifier):







    def __init__(
        self,
        k=15,
        metric="euclidean",
        weighted=False,
        batch_size=512,
        p=3.0,
        max_distance_mb=128.0,
    ):
        k = positive_int(k, "k")
        batch_size = positive_int(batch_size, "batch_size")
        p = finite_scalar(p, "p", positive=True)
        if p < 1:
            raise ValueError("Minkowski p must be at least 1")
        if metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError("metric must be euclidean, manhattan, or minkowski")
        self.k = int(k)
        self.metric = metric
        self.weighted = bool(weighted)
        self.batch_size = int(batch_size)
        self.p = float(p)
        self.max_distance_mb = finite_scalar(max_distance_mb, "max_distance_mb", positive=True)

    def fit(self, X, y, sample_weight=None):
        X, y = binary_data(X, y)
        sample_weight = sample_weights(sample_weight, len(y))
        keep = sample_weight > 0
        X, y, sample_weight = X[keep], y[keep], sample_weight[keep]
        if self.k > len(X):
            raise ValueError(f"k={self.k} exceeds positive-weight n_train={len(X)}")
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.sample_weight_ = sample_weight.copy()
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.train_norm_ = (X**2).sum(axis=1)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "X_"):
            raise RuntimeError("Call fit before predict_proba")
        query = prediction_data(X, self.n_features_in_)

        probabilities = np.empty(len(query), dtype=float)
        memory_batch = max(1, int(self.max_distance_mb * 1024**2 / (32 * len(self.X_))))
        batch_size = min(self.batch_size, memory_batch)
        for start in range(0, len(query), batch_size):
            stop = min(start + batch_size, len(query))
            distances = self._distances(query[start:stop])
            indices = np.argpartition(distances, self.k - 1, axis=1)[:, : self.k]
            nearest_distances = np.take_along_axis(distances, indices, axis=1)
            boundary = nearest_distances.max(axis=1)
            for row in np.flatnonzero((distances == boundary[:, None]).sum(axis=1) > 1):
                nearer = np.flatnonzero(distances[row] < boundary[row])
                tied = np.flatnonzero(distances[row] == boundary[row])
                indices[row] = np.r_[nearer, tied[:self.k - len(nearer)]]
            nearest_distances = np.take_along_axis(distances, indices, axis=1)
            
            order = np.argsort(nearest_distances, axis=1, kind="mergesort")
            indices = np.take_along_axis(indices, order, axis=1)
            nearest_distances = np.take_along_axis(nearest_distances, order, axis=1)
            labels = self.y_[indices]
            weights = self.sample_weight_[indices]

            if self.weighted:
                exact = nearest_distances <= 1e-12
                inverse = 1.0 / np.maximum(nearest_distances, 1e-12)
                weights = weights * np.where(exact.any(axis=1, keepdims=True), exact, inverse)
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
            train_norm = self.train_norm_[None, :]
            query_norm = (query**2).sum(axis=1)[:, None]
            squared = query_norm + train_norm - 2.0 * query @ self.X_.T
            return np.sqrt(np.maximum(squared, 0.0))

        power = 1.0 if self.metric == "manhattan" else self.p
        distances = np.zeros((len(query), len(self.X_)), dtype=float)
        for feature in range(self.n_features_in_):
            distances += np.abs(query[:, feature, None] - self.X_[None, :, feature]) ** power
        return distances if power == 1.0 else distances ** (1.0 / power)

