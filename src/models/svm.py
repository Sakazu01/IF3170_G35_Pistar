from __future__ import annotations
from time import perf_counter
import numpy as np

from src.base import BaseClassifier
from src.kernels import check_kernel_params, kernel_matrix, resolve_gamma
from src.numerics import sigmoid
from src.validation import (
    binary_data,
    finite_scalar,
    positive_int,
    prediction_data,
    random_seed,
    sample_weights,
)


def platt_scaling(scores, y, max_iter=100, tol=1e-9):
    n_pos, n_neg = float(y.sum()), float(len(y) - y.sum())
    target = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    A, B = 0.0, float(np.log((n_neg + 1.0) / (n_pos + 1.0)))

    def loss(a, b):
        z = a * scores + b
        return float(np.sum(target * np.logaddexp(0.0, z) + (1 - target) * np.logaddexp(0.0, -z)))

    current = loss(A, B)
    for _ in range(max_iter):
        p = sigmoid(-(A * scores + B))
        g = target - p
        h = np.maximum(p * (1 - p), 1e-12)
        grad = np.array([np.dot(g, scores), g.sum()])
        hess = np.array([[np.dot(h, scores**2), np.dot(h, scores)],
                         [np.dot(h, scores), h.sum()]]) + 1e-12 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        if np.abs(grad).max() < tol:
            break
        size = 1.0
        while size > 1e-10:
            candidate = loss(A - size * step[0], B - size * step[1])
            if candidate < current + 1e-4 * size * float(grad @ -step):
                A, B, current = A - size * step[0], B - size * step[1], candidate
                break
            size /= 2
        else:
            break
    return A, B


class SVMScratch(BaseClassifier):
    _finite_scalar = staticmethod(finite_scalar)
    _positive_int = staticmethod(positive_int)
    _weights = staticmethod(sample_weights)
    _xy = staticmethod(binary_data)

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=1.0,
        tol=1e-3,
        max_iter=100,
        random_state=42,
    ):
        self.C = self._finite_scalar(C, "C", positive=True)
        self.degree, self.coef0 = check_kernel_params(kernel, degree, coef0)
        self.kernel = kernel
        if isinstance(gamma, str):
            resolve_gamma(gamma, np.zeros((1, 1)))
        else:
            gamma = self._finite_scalar(gamma, "gamma", positive=True)
        self.gamma = gamma
        self.tol = self._finite_scalar(tol, "tol", positive=True)
        self.max_iter = self._positive_int(max_iter, "max_iter")
        self.random_state = random_seed(random_state)

    def _smo(self, K, s, c):
        n, tol = len(s), self.tol
        rng = np.random.default_rng(self.random_state)
        alpha, b = np.zeros(n), 0.0
        err = -s.copy() 

        def take_step(i, j):
            nonlocal b
            if i == j:
                return 0
            ai, aj, yi, yj = alpha[i], alpha[j], s[i], s[j]
            if yi != yj:
                low, high = max(0.0, aj - ai), min(c[j], c[i] + aj - ai)
            else:
                low, high = max(0.0, ai + aj - c[i]), min(c[j], ai + aj)
            if low >= high:
                return 0
            eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
            if eta <= 1e-12:
                return 0
            aj_new = float(np.clip(aj + yj * (err[i] - err[j]) / eta, low, high))
            if abs(aj_new - aj) < 1e-5 * (aj_new + aj + 1e-5):
                return 0
            ai_new = ai + yi * yj * (aj - aj_new)
            dai, daj = ai_new - ai, aj_new - aj
            b1 = b - err[i] - yi * dai * K[i, i] - yj * daj * K[i, j]
            b2 = b - err[j] - yi * dai * K[i, j] - yj * daj * K[j, j]
            if 0 < ai_new < c[i]:
                b_new = b1
            elif 0 < aj_new < c[j]:
                b_new = b2
            else:
                b_new = (b1 + b2) / 2.0
            err[:] += yi * dai * K[i] + yj * daj * K[j] + (b_new - b)
            alpha[i], alpha[j], b = ai_new, aj_new, b_new
            return 1

        def examine(j):
            r = err[j] * s[j]
            if (r < -tol and alpha[j] < c[j]) or (r > tol and alpha[j] > 0):
                free = np.flatnonzero((alpha > 0) & (alpha < c))
                if len(free) > 1 and take_step(free[np.argmax(np.abs(err[free] - err[j]))], j):
                    return 1
                for i in rng.permutation(free):
                    if take_step(i, j):
                        return 1
                for i in rng.permutation(n):
                    if take_step(i, j):
                        return 1
            return 0

        changed, examine_all, passes = 0, True, 0
        while (changed > 0 or examine_all) and passes < self.max_iter:
            changed = 0
            candidates = range(n) if examine_all else np.flatnonzero((alpha > 0) & (alpha < c))
            for j in candidates:
                changed += examine(j)
            examine_all = (not examine_all) if changed == 0 else False
            passes += 1
        return alpha, b, passes, changed == 0

    def fit(self, X, y, sample_weight=None):
        X, y = self._xy(X, y)
        if np.unique(y).size != 2:
            raise ValueError("Training requires both classes 0 and 1")
        weights = self._weights(sample_weight, len(y))
        weights = weights / weights.mean()
        started = perf_counter()
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1], dtype=int)
        self.gamma_ = resolve_gamma(self.gamma, X)
        s = np.where(y == 1, 1.0, -1.0)
        K = kernel_matrix(self.kernel, X, X, self.degree, self.gamma_, self.coef0)
        alpha, b, passes, converged = self._smo(K, s, self.C * weights)

        support = alpha > 1e-8
        self.alpha_ = alpha[support]
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[support]
        self.support_labels_ = s[support]
        self.dual_coef_ = self.alpha_ * self.support_labels_
        self.intercept_ = float(b)
        self.n_support_ = int(support.sum())
        self.n_iter_ = passes
        self.converged_ = bool(converged)
        self.dual_objective_ = float(alpha.sum() - 0.5 * (alpha * s) @ K @ (alpha * s))
        if self.kernel == "linear":
            self.coef_ = self.dual_coef_ @ self.support_vectors_

        scores = K[:, support] @ self.dual_coef_ + self.intercept_ 
        self.platt_a_, self.platt_b_ = platt_scaling(scores, y)
        self.training_time_ = perf_counter() - started
        return self

    def _check_fitted(self):
        if not hasattr(self, "dual_coef_"):
            raise RuntimeError("Call fit before prediction")

    def decision_function(self, X):
        self._check_fitted()
        X = prediction_data(X, self.n_features_in_)
        K = kernel_matrix(self.kernel, X, self.support_vectors_, self.degree, self.gamma_, self.coef0)
        return K @ self.dual_coef_ + self.intercept_

    def predict_proba(self, X):
        positive = sigmoid(-(self.platt_a_ * self.decision_function(X) + self.platt_b_))
        return np.column_stack([1.0 - positive, positive])


class MulticlassSVM(BaseClassifier):

    def __init__(self, strategy="ovr", **svm_params):
        if strategy not in ("ovr", "ovo"):
            raise ValueError("strategy must be 'ovr' or 'ovo'")
        self.strategy = strategy
        self.svm_params = dict(svm_params)
        SVMScratch(**self.svm_params)  # validasi parameter lebih awal

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Training requires at least two classes")
        self.estimators_ = []
        if self.strategy == "ovr":
            pairs = [(c, None) for c in self.classes_]
        else:
            pairs = [(a, b) for i, a in enumerate(self.classes_) for b in self.classes_[i + 1:]]
        for a, b in pairs:
            mask = np.ones(len(y), bool) if b is None else (y == a) | (y == b)
            target = (y[mask] == a).astype(int)
            self.estimators_.append((a, b, SVMScratch(**self.svm_params).fit(X[mask], target)))
        return self

    def decision_function(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Call fit before prediction")
        X = np.asarray(X, dtype=float)
        return np.column_stack([m.decision_function(X) for _, _, m in self.estimators_])

    def predict(self, X):
        scores = self.decision_function(X)
        if self.strategy == "ovr":
            return self.classes_[np.argmax(scores, axis=1)]
        votes = np.zeros((len(scores), len(self.classes_)))
        index = {c: i for i, c in enumerate(self.classes_)}
        rows = np.arange(len(scores))
        for k, (a, b, _) in enumerate(self.estimators_):
            votes[rows, np.where(scores[:, k] >= 0, index[a], index[b])] += 1
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X):
        raise NotImplementedError("MulticlassSVM exposes predict/decision_function only")


SVM = SVMScratch