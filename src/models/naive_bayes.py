import numpy as np
from src.base import BaseClassifier


class NaiveBayes(BaseClassifier):
    def __init__(
        self,
        alpha=1.0,
        var_smoothing=1e-9,
        categorical_idx=None,
    ):
        self.alpha = alpha
        self.var_smoothing = var_smoothing
        self.categorical_idx = [] if categorical_idx is None else categorical_idx

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n_samples, n_features = X.shape
        weights = np.ones(n_samples)
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)

        self.classes_ = np.array([0, 1])

        # Hitung peluang awal tiap kelas dari bobot data train.
        class_weights = np.zeros(2)
        for label in self.classes_:
            class_rows = y == label
            class_weights[label] = weights[class_rows].sum()

        if (class_weights <= 0).any():
            raise ValueError("kelas 0 dan 1 harus punya total bobot positif")
        self.class_prior_ = class_weights / class_weights.sum()

        categorical_columns = list(self.categorical_idx)
        numeric_columns = []
        for column_idx in range(n_features):
            if column_idx not in categorical_columns:
                numeric_columns.append(column_idx)

        self._fit_categorical(X, y, weights, categorical_columns)
        self._fit_gaussian(X, y, weights, numeric_columns)
        return self

    def _fit_categorical(self, X, y, weights, categorical_columns):
        self.cat_log_prob_ = {}
        for column_idx in categorical_columns:
            column = X[:, column_idx]
            observed = ~np.isnan(column)
            n_categories = int(column[observed].max()) + 1
            table = np.zeros((2, n_categories))
            for label in self.classes_:
                rows = observed & (y == label)
                category_counts = np.bincount(
                    column[rows].astype(int),
                    weights=weights[rows],
                    minlength=n_categories,
                )
                # Tambahkan smoothing supaya kategori dengan frekuensi nol tetap punya peluang.
                smoothed_counts = category_counts + self.alpha
                total_count = category_counts.sum() + self.alpha * n_categories
                category_probabilities = smoothed_counts / total_count
                table[label] = np.log(category_probabilities)
            self.cat_log_prob_[column_idx] = table

    def _fit_gaussian(self, X, y, weights, numeric_columns):
        self.epsilon_ = 0.0
        self.constant_columns_ = set()
        if numeric_columns:
            variances = np.nanvar(X[:, numeric_columns], axis=0)
            self.epsilon_ = self.var_smoothing * variances.max()
            for column_idx, variance in zip(numeric_columns, variances):
                if variance == 0:
                    self.constant_columns_.add(column_idx)

        n_features = X.shape[1]
        self.theta_ = np.zeros((2, n_features))
        self.var_ = np.ones((2, n_features))
        for column_idx in numeric_columns:
            column = X[:, column_idx]
            observed = ~np.isnan(column)
            for label in self.classes_:
                rows = (y == label) & observed
                if not rows.any():
                    raise ValueError(f"kolom {column_idx} kosong semua untuk kelas {label}")
                values = column[rows]
                mean = np.average(values, weights=weights[rows])
                variance = np.average((values - mean) ** 2, weights=weights[rows])
                self.theta_[label, column_idx] = mean
                smoothed_variance = variance + self.epsilon_
                if smoothed_variance <= 0:
                    smoothed_variance = np.finfo(float).eps
                self.var_[label, column_idx] = smoothed_variance

    def predict_proba(self, X):
        log_scores = self._joint_log_likelihood(X)
        shifted_scores = log_scores - log_scores.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted_scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, 1]

    def _joint_log_likelihood(self, X) -> np.ndarray:
        # Mulai dari log prior, lalu tambahkan kontribusi fitur yang tersedia.
        X = np.asarray(X, dtype=float)
        log_scores = np.zeros((len(X), 2))
        for label in self.classes_:
            log_scores[:, label] = np.log(self.class_prior_[label])
        constant_columns = getattr(self, "constant_columns_", ())

        for column_idx in range(X.shape[1]):
            # Nilai konstan memberi bukti yang sama untuk kedua kelas.
            if column_idx in constant_columns:
                continue
            column = X[:, column_idx]
            observed = ~np.isnan(column)

            if column_idx in self.cat_log_prob_:
                table = self.cat_log_prob_[column_idx]
                code = np.zeros(len(column), dtype=int)
                in_range = (column >= 0) & (column < table.shape[1])
                observed = observed & in_range
                code[observed] = column[observed].astype(int)
                observed = observed & (column == code)
                for label in self.classes_:
                    log_scores[observed, label] += table[label, code[observed]]
            else:
                for label in self.classes_:
                    mean = self.theta_[label, column_idx]
                    variance = self.var_[label, column_idx]
                    squared_distance = (column[observed] - mean) ** 2
                    log_density = -0.5 * (
                        np.log(2 * np.pi * variance) + squared_distance / variance
                    )
                    log_scores[observed, label] += log_density
        return log_scores
