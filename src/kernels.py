"""
Kernel SVM: linear, polynomial, gaussian
"""
from __future__ import annotations

import numpy as np

from src.validation import finite_scalar, positive_int

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