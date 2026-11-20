from __future__ import annotations

from time import perf_counter

import numpy as np

from src.base import BaseClassifier
from src.numerics import binary_cross_entropy, sigmoid
from src.optimizers import make_optimizer
from src.regularization import Regularizer, soft_threshold
from src.validation import (
    binary_data,
    decision_threshold,
    finite_scalar,
    positive_int,
    prediction_data,
    random_seed,
    sample_weights,
    unit_interval,
)


class LogisticRegressionScratch(BaseClassifier):
    sigmoid = staticmethod(sigmoid)
    binary_cross_entropy = staticmethod(binary_cross_entropy)
    _finite_scalar = staticmethod(finite_scalar)
    _positive_int = staticmethod(positive_int)
    _unit_interval = staticmethod(unit_interval)
    _weights = staticmethod(sample_weights)
    _xy = staticmethod(binary_data)
    _soft_threshold = staticmethod(soft_threshold)
    _decision_threshold = staticmethod(decision_threshold)

    def __init__(
        self,
        learning_rate=0.1,
        max_iter=2000,
        penalty=None,
        reg_strength=0.0,
        l1_ratio=0.5,
        optimizer="gd",
        batch_size=None,
        shuffle=True,
        random_state=42,
        tol=1e-5,
        patience=3,
        fit_intercept=True,
        init="zeros",
        init_scale=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        threshold=0.5,
        store_parameters=False,
    ):
        self.learning_rate = self._finite_scalar(learning_rate, "learning_rate", positive=True)
        self.max_iter = self._positive_int(max_iter, "max_iter")
        regularizer = Regularizer(penalty, reg_strength, l1_ratio)
        if optimizer not in ("gd", "sgd", "adam"):
            raise ValueError("optimizer must be 'gd', 'sgd', or 'adam'")
        if init not in ("zeros", "normal"):
            raise ValueError("init must be 'zeros' or 'normal'")
        self.penalty = regularizer.penalty
        self.reg_strength = regularizer.reg_strength
        self.l1_ratio = regularizer.l1_ratio
        self.optimizer = optimizer
        self.batch_size = None if batch_size is None else self._positive_int(batch_size, "batch_size")
        if optimizer == "gd" and batch_size is not None:
            raise ValueError("GD uses full batches; use optimizer='sgd' for batch_size")
        self.shuffle = bool(shuffle)
        self.random_state = random_seed(random_state)
        self.tol = self._finite_scalar(tol, "tol")
        self.patience = self._positive_int(patience, "patience")
        self.fit_intercept = bool(fit_intercept)
        self.init = init
        self.init_scale = self._finite_scalar(init_scale, "init_scale")
        self.beta1 = self._unit_interval(beta1, "beta1", exclude_one=True)
        self.beta2 = self._unit_interval(beta2, "beta2", exclude_one=True)
        self.epsilon = self._finite_scalar(epsilon, "epsilon", positive=True)
        self.threshold = self._decision_threshold(threshold)
        self.store_parameters = bool(store_parameters)

    def _make_regularizer(self):
        return Regularizer(self.penalty, self.reg_strength, self.l1_ratio)

    def _penalty_weights(self):
        regularizer = self._make_regularizer()
        return regularizer.l1, regularizer.l2

    def _penalty(self, w):
        return self._make_regularizer().value(w)

    def loss_and_gradient(self, X, y, coef=None, intercept=None, sample_weight=None):
        X, y = self._xy(X, y)
        if coef is None:
            self._check_fitted()
        w = np.asarray(self.coef_ if coef is None else coef, dtype=float)
        b = float(getattr(self, "intercept_", 0.0) if intercept is None else intercept)
        if w.shape != (X.shape[1],) or not np.isfinite(w).all() or not np.isfinite(b):
            raise ValueError("coef/intercept must be finite and match X's features")
        if not self.fit_intercept and b != 0:
            raise ValueError("intercept must be zero when fit_intercept=False")
        weights = self._weights(sample_weight, len(y))
        z = X @ w + b
        residual = (self.sigmoid(z) - y) * (weights / weights.sum())
        regularizer = self._make_regularizer()
        grad_w = X.T @ residual + regularizer.smooth_gradient(w) + regularizer.l1 * np.sign(w)
        grad_b = float(residual.sum()) if self.fit_intercept else 0.0
        return self.binary_cross_entropy(y, z, weights) + regularizer.value(w), grad_w, grad_b

    def _initialize_fit(self, n_features, rng):
        self.n_features_in_ = n_features
        self.classes_ = np.array([0, 1], dtype=int)
        self.coef_ = np.zeros(n_features)
        if self.init == "normal":
            self.coef_ = rng.normal(0.0, self.init_scale, n_features)
        self.intercept_ = 0.0
        self.n_iter_ = 0
        self.n_updates_ = 0
        self.converged_ = False
        self.convergence_reason_ = "max_iter"
        self.history_ = {
            key: [] for key in (
                "epoch", "loss", "train_log_loss", "validation_log_loss",
                "validation_objective", "optimality",
            )
        }
        self.parameter_history_ = []

    def _record_history(self, X, y, weights, normalized_weights, validation, regularizer, epoch):
        z = X @ self.coef_ + self.intercept_
        data_loss = self.binary_cross_entropy(y, z, weights)
        residual = (self.sigmoid(z) - y) * normalized_weights
        smooth_gradient = X.T @ residual + regularizer.smooth_gradient(self.coef_)
        mapping = regularizer.gradient_mapping(self.coef_, smooth_gradient)
        bias_gradient = float(residual.sum()) if self.fit_intercept else 0.0
        optimality = float(max(np.max(np.abs(mapping)), abs(bias_gradient)))
        penalty = regularizer.value(self.coef_)
        objective = data_loss + penalty
        if not np.isfinite(objective) or not np.isfinite(optimality):
            raise FloatingPointError("Training diverged; scale features or lower learning_rate")
        val_loss = np.nan
        if validation is not None:
            val_loss = self.binary_cross_entropy(
                validation[1], validation[0] @ self.coef_ + self.intercept_
            )
        values = (epoch, objective, data_loss, val_loss, val_loss + penalty, optimality)
        for key, value in zip(self.history_, values):
            self.history_[key].append(value)
        if self.store_parameters:
            self.parameter_history_.append(np.r_[self.intercept_, self.coef_].copy())
        return optimality

    def _update_parameters(self, X, y, normalized_weights, scale, regularizer, optimizer):
        residual = (
            (self.sigmoid(X @ self.coef_ + self.intercept_) - y)
            * normalized_weights * scale
        )
        gradient = np.r_[
            X.T @ residual + regularizer.smooth_gradient(self.coef_),
            residual.sum() if self.fit_intercept else 0.0,
        ]
        parameters = optimizer.step(np.r_[self.coef_, self.intercept_], gradient)
        self.n_updates_ = optimizer.n_updates_
        step_size = optimizer.step_size
        weight_step = step_size[:-1] if np.ndim(step_size) else step_size
        self.coef_ = regularizer.proximal(parameters[:-1], weight_step)
        self.intercept_ = float(parameters[-1]) if self.fit_intercept else 0.0
        if not np.isfinite(self.coef_).all() or not np.isfinite(self.intercept_):
            raise FloatingPointError("Non-finite parameters; lower learning_rate")

    def fit(self, X, y, sample_weight=None, validation_data=None):
        X, y = self._xy(X, y)
        if np.unique(y).size != 2:
            raise ValueError("Training requires both classes 0 and 1")
        weights = self._weights(sample_weight, len(y))
        val = None
        if validation_data is not None:
            if len(validation_data) != 2:
                raise ValueError("validation_data must be (X_val, y_val)")
            val = self._xy(*validation_data)
            if val[0].shape[1] != X.shape[1]:
                raise ValueError("Validation features must match training features")
        started = perf_counter()
        rng = np.random.default_rng(self.random_state)
        self._initialize_fit(X.shape[1], rng)
        regularizer = self._make_regularizer()
        optimizer = make_optimizer(
            self.optimizer, self.learning_rate, self.beta1, self.beta2, self.epsilon
        )
        size = len(X) if self.optimizer == "gd" else (self.batch_size or (32 if self.optimizer == "sgd" else len(X)))
        normalized_weights = weights / weights.sum()
        self._record_history(X, y, weights, normalized_weights, val, regularizer, 0)
        small_residual_epochs = 0
        for epoch in range(1, self.max_iter + 1):
            order = rng.permutation(len(X)) if self.shuffle and size < len(X) else np.arange(len(X))
            for start in range(0, len(X), size):
                indices = order[start:start + size]
                xb, yb = (X, y) if size >= len(X) else (X[indices], y[indices])
                self._update_parameters(
                    xb, yb, normalized_weights[indices], len(X) / len(indices),
                    regularizer, optimizer,
                )
            self.n_iter_ = epoch
            optimality = self._record_history(
                X, y, weights, normalized_weights, val, regularizer, epoch
            )
            small_residual_epochs = small_residual_epochs + 1 if self.tol > 0 and optimality <= self.tol else 0
            if small_residual_epochs >= self.patience:
                self.converged_ = True
                self.convergence_reason_ = "proximal_gradient_tolerance"
                break
        self.training_time_ = perf_counter() - started
        self.loss_history_ = np.asarray(self.history_["loss"])
        self.parameter_history_ = np.asarray(self.parameter_history_).reshape((-1, X.shape[1] + 1))
        self.optimality_ = self.history_["optimality"][-1]
        return self

    def _check_fitted(self):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before prediction")

    def decision_function(self, X):
        self._check_fitted()
        X = prediction_data(X, self.n_features_in_)
        with np.errstate(over="raise", invalid="raise"):
            z = X @ self.coef_ + self.intercept_
        if not np.isfinite(z).all():
            raise FloatingPointError("Non-finite linear combination; rescale input")
        return z

    def predict_proba(self, X):
        return self.sigmoid(self.decision_function(X))

    def predict(self, X, threshold=None):
        threshold = self._decision_threshold(self.threshold if threshold is None else threshold)
        return (self.predict_proba(X) >= threshold).astype(int)

LogisticRegression = LogisticRegressionScratch
