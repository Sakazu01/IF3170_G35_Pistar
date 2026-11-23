import numpy as np
from src.base import BaseClassifier
from src.base import binary_data, prediction_data, sample_weights, finite_scalar


class NaiveBayes(BaseClassifier):
    def __init__(self, alpha=1.0, var_smoothing=1e-9, categorical_idx=None):
        self.alpha = finite_scalar(alpha, "alpha", positive=True)
        self.var_smoothing = finite_scalar(var_smoothing, "var_smoothing")
        self.categorical_idx = [] if categorical_idx is None else list(categorical_idx)
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or i < 0
               for i in self.categorical_idx):
            raise ValueError("categorical_idx must contain nonnegative integer indices")

    def fit(self, X, y, sample_weight=None):
        X, y = binary_data(X, y, allow_nan=True)
        if any(i >= X.shape[1] for i in self.categorical_idx):
            raise ValueError("categorical_idx contains an index outside X")
        weights = sample_weights(sample_weight, len(y))
        keep = weights > 0
        X, y, weights = X[keep], y[keep], weights[keep]
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        counts = np.bincount(y, weights=weights, minlength=2)
        if (counts <= 0).any():
            raise ValueError("Both classes need positive total weight")
        self.class_prior_ = counts / counts.sum()
        self.cat_log_prob_, self.category_values_ = {}, {}
        self.theta_ = np.zeros((2, X.shape[1]))
        self.var_ = np.ones((2, X.shape[1]))
        self.constant_columns_ = set()
        numeric = [j for j in range(X.shape[1]) if j not in self.categorical_idx]
        observed_variances = [np.var(X[~np.isnan(X[:, j]), j]) for j in numeric
                              if (~np.isnan(X[:, j])).any()]
        self.epsilon_ = self.var_smoothing * max(observed_variances, default=0.0)
        for j in range(X.shape[1]):
            column = X[:, j]
            observed = ~np.isnan(column)
            if j in self.categorical_idx:
                values = column[observed]
                if np.any((values < 0) | (values != np.floor(values))):
                    raise ValueError("Training categorical codes must be nonnegative integers or NaN")
                categories = np.unique(values)
                self.category_values_[j] = categories
                table = np.zeros((2, len(categories)))
                for label in self.classes_:
                    rows = observed & (y == label)
                    codes = np.searchsorted(categories, column[rows])
                    counts = np.bincount(codes, weights=weights[rows], minlength=len(categories))
                    if len(categories):
                        table[label] = np.log((counts + self.alpha) /
                                              (counts.sum() + self.alpha * len(categories)))
                self.cat_log_prob_[j] = table
            elif not observed.any():
                self.constant_columns_.add(j)
            else:
                global_mean = np.average(column[observed], weights=weights[observed])
                global_var = np.average((column[observed] - global_mean)**2, weights=weights[observed])
                if global_var == 0:
                    self.constant_columns_.add(j)
                for label in self.classes_:
                    rows = observed & (y == label)
                    if rows.any():
                        mean = np.average(column[rows], weights=weights[rows])
                        variance = np.average((column[rows] - mean)**2, weights=weights[rows])
                    else:
                        mean, variance = global_mean, global_var
                    self.theta_[label, j] = mean
                    self.var_[label, j] = max(variance + self.epsilon_, np.finfo(float).eps)
        return self

    def predict_proba(self, X):
        log_scores = self._joint_log_likelihood(X)
        probabilities = np.exp(log_scores - log_scores.max(axis=1, keepdims=True))
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def _joint_log_likelihood(self, X):
        if not hasattr(self, "class_prior_"):
            raise RuntimeError("Call fit before prediction")
        X = prediction_data(X, self.n_features_in_, allow_nan=True)
        scores = np.tile(np.log(self.class_prior_), (len(X), 1))
        for j in range(X.shape[1]):
            if j in self.constant_columns_:
                continue
            column = X[:, j]
            observed = ~np.isnan(column)
            if j in self.cat_log_prob_:
                categories = self.category_values_[j]
                if not len(categories):
                    continue
                position = np.minimum(np.searchsorted(categories, column), len(categories) - 1)
                observed &= categories[position] == column
                for label in self.classes_:
                    scores[observed, label] += self.cat_log_prob_[j][label, position[observed]]
            else:
                for label in self.classes_:
                    variance = self.var_[label, j]
                    scores[observed, label] -= 0.5 * (
                        np.log(2 * np.pi * variance)
                        + (column[observed] - self.theta_[label, j])**2 / variance
                    )
        return scores
