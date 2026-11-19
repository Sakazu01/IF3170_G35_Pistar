from __future__ import annotations

import numpy as np

from src.validation import finite_scalar, unit_interval


def soft_threshold(value, amount):
    return np.sign(value) * np.maximum(np.abs(value) - amount, 0.0)


class Regularizer:
    def __init__(self, penalty=None, reg_strength=0.0, l1_ratio=0.5):
        if penalty == "none":
            penalty = None
        if penalty not in (None, "l1", "l2", "elasticnet"):
            raise ValueError("penalty must be None, 'l1', 'l2', or 'elasticnet'")
        self.penalty = penalty
        self.reg_strength = finite_scalar(reg_strength, "reg_strength")
        self.l1_ratio = unit_interval(l1_ratio, "l1_ratio")
        rho = {None: 0.0, "l1": 1.0, "l2": 0.0, "elasticnet": self.l1_ratio}[penalty]
        strength = 0.0 if penalty is None else self.reg_strength
        self.l1 = strength * rho
        self.l2 = strength * (1.0 - rho)

    def value(self, weights):
        weights = np.asarray(weights, dtype=float)
        flat = weights.ravel()
        return float(self.l1 * np.abs(weights).sum() + 0.5 * self.l2 * np.dot(flat, flat))

    def smooth_gradient(self, weights):
        return self.l2 * np.asarray(weights, dtype=float)

    def subgradient(self, weights):
        weights = np.asarray(weights, dtype=float)
        return self.smooth_gradient(weights) + self.l1 * np.sign(weights)

    def proximal(self, weights, step_size):
        return soft_threshold(weights, self.l1 * step_size)

    def gradient_mapping(self, weights, smooth_gradient):
        return weights - self.proximal(weights - smooth_gradient, 1.0)
