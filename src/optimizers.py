from __future__ import annotations

import numpy as np

from src.validation import finite_scalar, unit_interval


class GradientDescent:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = finite_scalar(learning_rate, "learning_rate", positive=True)
        self.reset()

    def reset(self):
        self.step_size = self.learning_rate
        self.n_updates_ = 0
        return self

    @staticmethod
    def _inputs(parameters, gradient):
        parameters = np.asarray(parameters, dtype=float)
        gradient = np.asarray(gradient, dtype=float)
        if not parameters.size or parameters.shape != gradient.shape:
            raise ValueError("parameters and gradient must have matching, nonempty shapes")
        if not np.isfinite(parameters).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("parameters and gradient must be finite")
        return parameters, gradient

    def step(self, parameters, gradient):
        parameters, gradient = self._inputs(parameters, gradient)
        self.n_updates_ += 1
        self.step_size = self.learning_rate
        return parameters - self.step_size * gradient


class Adam(GradientDescent):
    def __init__(self, learning_rate=0.03, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = unit_interval(beta1, "beta1", exclude_one=True)
        self.beta2 = unit_interval(beta2, "beta2", exclude_one=True)
        self.epsilon = finite_scalar(epsilon, "epsilon", positive=True)
        super().__init__(learning_rate)

    def reset(self):
        super().reset()
        self.moment_ = None
        self.variance_ = None
        return self

    def step(self, parameters, gradient):
        parameters, gradient = self._inputs(parameters, gradient)
        if self.moment_ is None:
            self.moment_ = np.zeros_like(gradient)
            self.variance_ = np.zeros_like(gradient)
        elif self.moment_.shape != gradient.shape:
            raise ValueError("Call reset before optimizing parameters with a different shape")
        self.n_updates_ += 1
        self.moment_ = self.beta1 * self.moment_ + (1.0 - self.beta1) * gradient
        self.variance_ = self.beta2 * self.variance_ + (1.0 - self.beta2) * gradient**2
        corrected_moment = self.moment_ / (1.0 - self.beta1**self.n_updates_)
        corrected_variance = self.variance_ / (1.0 - self.beta2**self.n_updates_)
        self.step_size = self.learning_rate / (np.sqrt(corrected_variance) + self.epsilon)
        return parameters - self.step_size * corrected_moment


def make_optimizer(name, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    if name in ("gd", "sgd"):
        return GradientDescent(learning_rate)
    if name == "adam":
        return Adam(learning_rate, beta1, beta2, epsilon)
    raise ValueError("optimizer must be 'gd', 'sgd', or 'adam'")
