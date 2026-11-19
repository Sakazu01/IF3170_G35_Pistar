from __future__ import annotations

import numpy as np


def finite_scalar(value, name, positive=False):
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def unit_interval(value, name, exclude_one=False):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value >= 1 if exclude_one else value > 1):
        raise ValueError(f"{name} must be in [0, {'1)' if exclude_one else '1]'}")
    return value


def random_seed(value):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError("random_state must be a nonnegative integer or None")
    return None if value is None else int(value)


def decision_threshold(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= np.nextafter(1.0, np.inf):
        raise ValueError("threshold must be in [0,1] or nextafter(1,+inf) for all-negative")
    return value


def sample_weights(sample_weight, n_samples):
    weights = np.ones(n_samples) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite, nonnegative, shape (n_samples,)")
    if not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("sample_weight must have a finite positive sum")
    return weights


def binary_data(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty 2-D matrix with at least one feature")
    if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
        raise ValueError("y must be a binary 0/1 vector with one label per row")
    if not np.isfinite(X).all():
        raise ValueError("X must contain finite features")
    return X, y.astype(float, copy=False)


def prediction_data(X, n_features):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[1] != n_features or not np.isfinite(X).all():
        raise ValueError(f"X must be finite with shape (n_samples, {n_features})")
    return X
