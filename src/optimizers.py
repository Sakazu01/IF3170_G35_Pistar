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
        with np.errstate(over="raise", invalid="raise"):
            updated = parameters - self.learning_rate * gradient
        if not np.isfinite(updated).all():
            raise FloatingPointError("Gradient descent produced non-finite parameters")
        self.n_updates_ += 1
        self.step_size = self.learning_rate
        return updated


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
            previous_moment = np.zeros_like(gradient)
            previous_variance = np.zeros_like(gradient)
        elif self.moment_.shape != gradient.shape:
            raise ValueError("Call reset before optimizing parameters with a different shape")
        else:
            previous_moment, previous_variance = self.moment_, self.variance_
        update = self.n_updates_ + 1
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            moment = self.beta1 * previous_moment + (1.0 - self.beta1) * gradient
            variance = self.beta2 * previous_variance + (1.0 - self.beta2) * gradient**2
            corrected_moment = moment / (1.0 - self.beta1**update)
            corrected_variance = variance / (1.0 - self.beta2**update)
            step_size = self.learning_rate / (np.sqrt(corrected_variance) + self.epsilon)
            updated = parameters - step_size * corrected_moment
        if not all(np.isfinite(value).all() for value in (moment, variance, step_size, updated)):
            raise FloatingPointError("Adam produced non-finite state; scale features or gradients")
        self.n_updates_ = update
        self.moment_, self.variance_, self.step_size = moment, variance, step_size
        return updated


def make_optimizer(name, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    if name in ("gd", "sgd"):
        return GradientDescent(learning_rate)
    if name == "adam":
        return Adam(learning_rate, beta1, beta2, epsilon)
    raise ValueError("optimizer must be 'gd', 'sgd', or 'adam'")
