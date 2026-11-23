import pickle
from pathlib import Path

import numpy as np


# Pemeriksaan masukan yang dipakai bersama oleh kelima model from scratch.
def finite_scalar(value, name, positive=False):
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def random_seed(value):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer)) or value < 0
    ):
        raise ValueError("random_state must be a nonnegative integer or None")
    return None if value is None else int(value)


def sample_weights(sample_weight, n_samples):
    weights = np.ones(n_samples) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite, nonnegative, shape (n_samples,)")
    if not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("sample_weight must have a finite positive sum")
    return weights


def binary_data(X, y, allow_nan=False):
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if X.ndim != 2 or not X.size:
        raise ValueError("X must be a nonempty 2-D matrix with at least one feature")
    if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
        raise ValueError("y must be a binary 0/1 vector with one label per row")
    if np.isinf(X).any() or (not allow_nan and np.isnan(X).any()):
        raise ValueError("X must contain finite features" + (" or NaN" if allow_nan else ""))
    return X, y.astype(int, copy=False)


def prediction_data(X, n_features, allow_nan=False):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ValueError(f"X must have shape (n_samples, {n_features})")
    if np.isinf(X).any() or (not allow_nan and np.isnan(X).any()):
        raise ValueError("X must contain finite features" + (" or NaN" if allow_nan else ""))
    return X


def sigmoid(z):
    # Dihitung terpisah untuk z positif dan negatif agar exp tidak overflow.
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


class BaseClassifier:
    """Induk kelima model: prediksi label, akurasi, parameter, serta simpan dan muat model."""

    def predict(self, X, threshold=0.5):
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., np.inf):
            raise ValueError('threshold must be in [0, 1] or nextafter(1, +inf)')
        proba = np.asarray(self.predict_proba(X))
        return (proba[:, 1] >= threshold).astype(int)

    def score(self, X, y):
        prediction, y = self.predict(X), np.asarray(y)
        if y.shape != prediction.shape or not len(y) or not np.isin(y, [0, 1]).all():
            raise ValueError('y must contain one binary label per prediction')
        return float(np.mean(prediction == y))

    def get_params(self, deep=True):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.endswith("_") and not key.startswith("_")
        }

    def set_params(self, **params):
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}")
        self.__init__(**{**self.get_params(), **params})
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model
