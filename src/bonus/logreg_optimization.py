from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.evaluation import evaluate
from src.models.logistic_regression import LogisticRegressionScratch, binary_data

REFERENCES = (
    {
        "authors": "D. P. Kingma and J. Ba",
        "title": "Adam: A Method for Stochastic Optimization",
        "year": 2014,
        "url": "https://arxiv.org/abs/1412.6980",
    },
    {
        "authors": "N. Parikh and S. Boyd",
        "title": "Proximal Algorithms",
        "url": "https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf",
    },
)


def compare_optimizers(
    X_train, y_train, X_validation, y_validation, learning_rates=None,
    max_iter=1000, tol=1e-5, random_state=42, penalty="l2",
    reg_strength=0.01, l1_ratio=0.5, batch_size=None,
):
    X_train, y_train = binary_data(X_train, y_train)
    X_validation, y_validation = binary_data(X_validation, y_validation)
    if learning_rates is None:
        learning_rates = {"gd": (0.01, 0.1, 0.5), "adam": (0.005, 0.03, 0.1)}
    if not isinstance(learning_rates, Mapping) or not learning_rates:
        raise ValueError("learning_rates must map optimizer names to nonempty learning-rate sequences")
    models, rows, histories = {}, [], []
    for optimizer, rates in learning_rates.items():
        for learning_rate in rates:
            key = f"{optimizer}_lr{float(learning_rate):g}"
            if key in models:
                raise ValueError(f"Duplicate optimizer configuration: {key}")
            model = LogisticRegressionScratch(
                optimizer=optimizer, learning_rate=learning_rate, max_iter=max_iter,
                tol=tol, random_state=random_state, penalty=penalty,
                reg_strength=reg_strength, l1_ratio=l1_ratio,
                batch_size=None if optimizer == "gd" else batch_size,
            ).fit(X_train, y_train, validation_data=(X_validation, y_validation))
            models[key] = model
            rows.append({
                "run": key, "optimizer": optimizer, "learning_rate": model.learning_rate,
                **evaluate(y_validation, model.predict_proba(X_validation)[:, 1]),
                "train_objective": model.loss_history_[-1],
                "train_log_loss": model.history_["train_log_loss"][-1],
                "validation_log_loss": model.history_["validation_log_loss"][-1],
                "epochs": model.n_iter_, "updates": model.n_updates_,
                "converged": model.converged_,
                "epochs_to_convergence": model.n_iter_ if model.converged_ else None,
                "optimality": model.optimality_, "fit_time": model.training_time_,
                "coefficient_norm": float(np.linalg.norm(model.coef_)),
            })
            histories.append(pd.DataFrame(model.history_).assign(run=key))
    if not rows:
        raise ValueError("At least one optimizer and learning rate are required")
    return {
        "results": pd.DataFrame(rows),
        "history": pd.concat(histories, ignore_index=True),
        "models": models,
        "references": REFERENCES,
    }


def plot_optimizer_comparison(comparison):
    import matplotlib.pyplot as plt

    results, history = comparison["results"], comparison["history"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for run, rows in history.groupby("run", sort=False):
        axes[0].plot(rows.epoch, rows.loss, label=run)
    axes[0].set(xlabel="Epoch", ylabel="Training objective", title="Actual loss convergence")
    axes[0].legend(fontsize=7)
    for optimizer, rows in results.groupby("optimizer", sort=False):
        rows = rows.sort_values("learning_rate")
        axes[1].plot(rows.learning_rate, rows.validation_log_loss, "o-", label=optimizer)
    axes[1].set(xscale="log", xlabel="Learning rate", ylabel="Validation log-loss", title="Learning-rate sensitivity")
    axes[1].legend()
    positions = np.arange(len(results))
    axes[2].barh(positions, results.fit_time, color="#3b6ea5")
    axes[2].set_yticks(positions, results.run, fontsize=7)
    axes[2].set(xlabel="Fit time (seconds)", title="Measured time (one run)")
    axes[2].invert_yaxis()
    for ax in axes[:2]:
        ax.grid(alpha=0.2)
    return fig
