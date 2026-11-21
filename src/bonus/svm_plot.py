import matplotlib.pyplot as plt
import numpy as np

from src.models.svm import SVMScratch

KERNEL_TITLES = {"linear": "Linear", "poly": "Polynomial", "rbf": "Gaussian (RBF)"}


def fit_kernels(X2, y, kernels=("linear", "poly", "rbf"), **params):
    X2 = np.asarray(X2, dtype=float)
    if X2.ndim != 2 or X2.shape[1] != 2:
        raise ValueError("X2 harus punya tepat 2 kolom (dua fitur atau hasil reduksi 2D)")
    return {name: SVMScratch(kernel=name, **params).fit(X2, y) for name in kernels}


def plot_svm_kernels(X2, y, models=None, feature_names=("x1", "x2"), path=None, **params):
    """Gambar bidang pemisah (f=0), margin (f=+-1), dan support vector tiap kernel."""
    X2 = np.asarray(X2, dtype=float)
    y = np.asarray(y).astype(int)
    models = models or fit_kernels(X2, y, **params)
    pad = 0.1 * (X2.max(axis=0) - X2.min(axis=0))
    lo, hi = X2.min(axis=0) - pad, X2.max(axis=0) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.6), squeeze=False)
    for ax, (name, model) in zip(axes[0], models.items()):
        score = model.decision_function(grid).reshape(gx.shape)
        ax.contourf(gx, gy, score, levels=[score.min(), 0, score.max()],
                    colors=["#f4d9c6", "#cfe0f1"], alpha=0.6)
        ax.contour(gx, gy, score, levels=[-1, 1], colors="#555555", linestyles="--", linewidths=1)
        ax.contour(gx, gy, score, levels=[0], colors="#1f1f1f", linewidths=1.8)
        for label, color, text in ((0, "#c8672d", "tidak batal"), (1, "#3b6ea5", "batal")):
            rows = y == label
            ax.scatter(X2[rows, 0], X2[rows, 1], s=14, color=color, alpha=0.55, label=text)
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=70,
                   facecolors="none", edgecolors="#1f1f1f", linewidths=1.2,
                   label=f"support vector ({model.n_support_})")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title(KERNEL_TITLES.get(name, name))
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("SVM: bidang pemisah (garis tebal), margin (putus-putus), support vector (lingkaran)")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
