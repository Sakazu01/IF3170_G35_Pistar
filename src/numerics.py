from __future__ import annotations

import numpy as np

from src.validation import sample_weights


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1.0 + exp_z)
    return result


def binary_cross_entropy(y, logits, sample_weight=None):
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)
    if logits.ndim != 1 or y.shape != logits.shape or not len(y):
        raise ValueError("y and logits must be nonempty 1-D arrays of equal shape")
    if not np.isfinite(logits).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("BCE requires finite logits and binary 0/1 labels")
    losses = np.logaddexp(0.0, (1.0 - 2.0 * y) * logits)
    weights = sample_weights(sample_weight, len(y))
    return float(np.dot(weights / weights.sum(), losses))
