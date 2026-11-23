import numpy as np

from src.base import BaseClassifier


from src.validation import (finite_scalar, positive_int, random_seed, sample_weights,
                            binary_data, prediction_data, sigmoid)


KERNELS = ("linear", "poly", "rbf")


def resolve_gamma(gamma, X):
    if isinstance(gamma, str):
        if gamma == "scale":
            variance = float(np.var(X))
            return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise ValueError("gamma must be 'scale', 'auto', or a positive float")
    return finite_scalar(gamma, "gamma", positive=True)


def linear_kernel(A, B):
    return A @ B.T


def polynomial_kernel(A, B, degree=3, gamma=1.0, coef0=1.0):
    return (gamma * (A @ B.T) + coef0) ** degree


def gaussian_kernel(A, B, gamma=1.0):
    squared = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(squared, 0.0))


def kernel_matrix(name, A, B, degree=3, gamma=1.0, coef0=1.0):
    if name == "linear":
        return linear_kernel(A, B)
    if name == "poly":
        return polynomial_kernel(A, B, degree, gamma, coef0)
    if name == "rbf":
        return gaussian_kernel(A, B, gamma)
    raise ValueError(f"kernel must be one of {KERNELS}")


def check_kernel_params(kernel, degree, coef0):
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    return positive_int(degree, "degree"), finite_scalar(coef0, "coef0")


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
        max_kernel_mb=1024.0,
        batch_size=256,
        calibration_fraction=0.0,
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
        self.max_kernel_mb = finite_scalar(max_kernel_mb, "max_kernel_mb", positive=True)
        self.batch_size = positive_int(batch_size, "batch_size")
        self.calibration_fraction = float(calibration_fraction)
        if not np.isfinite(self.calibration_fraction) or not 0 <= self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in [0, 0.5)")

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
            gradient = yj * (err[i] - err[j])
            if eta <= 1e-12:
                
                
                endpoints = np.array([low, high]) - aj
                gains = gradient * endpoints - 0.5 * eta * endpoints**2
                aj_new = float(aj + endpoints[np.argmax(gains)])
                if gains.max() <= 1e-12:
                    return 0
            else:
                aj_new = float(np.clip(aj + gradient / eta, low, high))
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
        keep = weights > 0
        X, y, weights = X[keep], y[keep], weights[keep]
        if np.unique(y).size != 2:
            raise ValueError("Both classes need positive total weight")
        calibration = None
        if self.calibration_fraction:
            rng = np.random.default_rng(self.random_state)
            held = []
            for label in (0, 1):
                rows = rng.permutation(np.flatnonzero(y == label))
                if len(rows) < 2:
                    raise ValueError("Calibration requires at least two rows of each class")
                held.extend(rows[:max(1, int(len(rows) * self.calibration_fraction))])
            held = np.sort(held)
            training = np.ones(len(y), dtype=bool)
            training[held] = False
            calibration = X[held], y[held]
            X, y, weights = X[training], y[training], weights[training]
        self.kernel_peak_mb_ = 3 * len(X)**2 * 8 / 1024**2
        if self.kernel_peak_mb_ > self.max_kernel_mb:
            raise MemoryError(f"Estimated kernel peak {self.kernel_peak_mb_:.1f} MiB exceeds "
                              f"max_kernel_mb={self.max_kernel_mb:g}; use a stratified training subset")
        weights = weights / weights.mean()
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1], dtype=int)
        self.gamma_ = resolve_gamma(self.gamma, X)
        s = np.where(y == 1, 1.0, -1.0)
        K = kernel_matrix(self.kernel, X, X, self.degree, self.gamma_, self.coef0)
        if not np.isfinite(K).all():
            raise FloatingPointError("Kernel overflow; scale features or lower degree/gamma")
        alpha, b, passes, _ = self._smo(K, s, self.C * weights)

        support = alpha > 1e-8
        self.alpha_ = alpha[support]
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[support]
        self.support_labels_ = s[support]
        self.dual_coef_ = self.alpha_ * self.support_labels_
        self.intercept_ = float(b)
        self.n_support_ = int(support.sum())
        self.n_iter_ = passes
        if self.kernel == "linear":
            self.coef_ = self.dual_coef_ @ self.support_vectors_

        scores = K[:, support] @ self.dual_coef_ + self.intercept_ 
        margin = s * scores
        violation = np.maximum(np.where(alpha < self.C * weights - 1e-8, 1 - margin, 0),
                               np.where(alpha > 1e-8, margin - 1, 0))
        self.max_kkt_violation_ = float(max(0, violation.max()))
        self.converged_ = self.max_kkt_violation_ <= self.tol * 2
        self.dual_equality_residual_ = float(abs(alpha @ s))
        self.n_training_samples_ = len(X)
        if calibration is None:
            self.calibration_method_ = "in_sample_platt"
            self.platt_a_, self.platt_b_ = platt_scaling(scores, y)
        else:
            self.calibration_method_ = "held_out_platt"
            self.platt_a_, self.platt_b_ = platt_scaling(self.decision_function(calibration[0]), calibration[1])
        return self

    def _check_fitted(self):
        if not hasattr(self, "dual_coef_"):
            raise RuntimeError("Call fit before prediction")

    def decision_function(self, X):
        self._check_fitted()
        X = prediction_data(X, self.n_features_in_)
        scores = np.empty(len(X))
        batch = min(self.batch_size, max(1, int(self.max_kernel_mb * 1024**2 /
                                               (24 * max(1, self.n_support_)))))
        for start in range(0, len(X), batch):
            K = kernel_matrix(self.kernel, X[start:start + batch], self.support_vectors_,
                              self.degree, self.gamma_, self.coef0)
            scores[start:start + batch] = K @ self.dual_coef_ + self.intercept_
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite SVM decision scores")
        return scores

    def predict_proba(self, X):
        positive = sigmoid(-(self.platt_a_ * self.decision_function(X) + self.platt_b_))
        return np.column_stack([1.0 - positive, positive])


SVM = SVMScratch
