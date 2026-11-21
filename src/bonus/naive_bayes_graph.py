import matplotlib.pyplot as plt
import numpy as np


def feature_strengths(model, feature_names=None):
    """Jarak Bhattacharyya (nat) antara distribusi fitur di kelas 0 dan kelas 1."""
    n_features = len(model.theta_[0]) if hasattr(model, "theta_") else 0
    names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(n_features)]
    strengths = {}
    for j in range(n_features):
        if j in model.constant_columns_:
            strengths[names[j]] = 0.0
        elif j in model.cat_log_prob_:
            table = np.exp(model.cat_log_prob_[j])
            strengths[names[j]] = float(-np.log(np.sqrt(table[0] * table[1]).sum()))
        else:
            v0, v1 = model.var_[0, j], model.var_[1, j]
            gap = model.theta_[1, j] - model.theta_[0, j]
            strengths[names[j]] = float(
                0.25 * gap**2 / (v0 + v1) + 0.5 * np.log((v0 + v1) / (2.0 * np.sqrt(v0 * v1)))
            )
    return strengths


def plot_naive_bayes_graph(model, feature_names=None, top_n=15, class_name="is_canceled", path=None):
    """Graf Naive Bayes: simpul kelas di tengah, panah ke tiap fitur (tebal = makin informatif)."""
    strengths = feature_strengths(model, feature_names)
    ranked = sorted(strengths.items(), key=lambda item: item[1], reverse=True)[: int(top_n)]
    top = max(value for _, value in ranked) or 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, len(ranked), endpoint=False)

    fig, ax = plt.subplots(figsize=(9, 9))
    for angle, (name, value) in zip(angles, ranked):
        x, y = 3.2 * np.cos(angle), 3.2 * np.sin(angle)
        ax.annotate("", xy=(0.92 * x, 0.92 * y), xytext=(0.0, 0.0),
                    arrowprops={"arrowstyle": "-|>", "color": "#3b6ea5",
                                "lw": 0.8 + 5.0 * value / top, "alpha": 0.75})
        ax.text(x, y, f"{name}\n{value:.3f}", ha="center", va="center", fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "fc": "#f4f1ea", "ec": "#8a8577"})
    ax.text(0.0, 0.0, f"{class_name}\nP(0)={model.class_prior_[0]:.2f}\nP(1)={model.class_prior_[1]:.2f}",
            ha="center", va="center", fontsize=10, fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.5", "fc": "#cfe0f1", "ec": "#3b6ea5"})
    ax.set_xlim(-4.3, 4.3)
    ax.set_ylim(-4.3, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Naive Bayes: kelas -> fitur (fitur saling independen jika kelas diketahui)\n"
                 "angka = jarak Bhattacharyya antar-kelas, makin besar makin informatif")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
