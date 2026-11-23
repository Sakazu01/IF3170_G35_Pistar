import numpy as np
import pandas as pd

from src.evaluation import COST_FN, COST_FP, best_threshold, confusion, evaluate, threshold_curve


def cost_sensitive_analysis(model, X_validation, y_validation, X_test, y_test):






    validation_proba = model.predict_proba(X_validation)[:, 1]
    threshold = best_threshold(y_validation, validation_proba, metric="cost")
    curve = threshold_curve(y_validation, validation_proba)
    test_proba = model.predict_proba(X_test)[:, 1]
    bayes_threshold = COST_FP / (COST_FP + COST_FN)
    policies = (("baseline", 0.5), ("bayes", bayes_threshold), ("validation_tuned", threshold))
    rows = []
    for split, labels, proba in (
        ("validation", y_validation, validation_proba), ("test", y_test, test_proba)
    ):
        proba = np.asarray(proba, dtype=float)
        for policy, cutoff in policies:
            metrics = evaluate(labels, proba, cutoff)
            tn, fp, fn, tp = confusion(labels, proba >= cutoff)
            rows.append({
                "split": split, "policy": policy, "threshold": cutoff,
                **metrics, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "total_cost": float(COST_FP * fp + COST_FN * fn),
                "mean_cost": metrics["expected_cost"],
            })
    return {
        "threshold": threshold,
        "bayes_threshold": bayes_threshold,
        "positive_class": 1,
        "cost_matrix": np.array([[0.0, COST_FP], [COST_FN, 0.0]]),
        "results": pd.DataFrame(rows),
        "curve": curve,
    }


def plot_cost_sensitive(analysis):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    curve = analysis["curve"]
    axes[0].step(curve.threshold, curve.cost, where="pre", color="#3b6ea5")
    for cutoff, label, color in (
        (0.5, "Baseline", "gray"),
        (analysis["bayes_threshold"], "Bayes threshold", "#29838c"),
        (analysis["threshold"], "Validation optimum", "#df8b25"),
    ):
        axes[0].axvline(cutoff, label=label, color=color, linestyle="--")
    axes[0].set(xlabel="Decision threshold", ylabel="Validation total cost", title="Cost = 2 FP + FN")
    axes[0].legend(fontsize=7)
    rows = analysis["results"].query("split == 'test'").set_index("policy")
    maximum = float(rows[["tn", "fp", "fn", "tp"]].max().max())
    for ax, policy in zip(axes[1:], ("baseline", "validation_tuned")):
        row = rows.loc[policy]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap="Purples", vmin=0, vmax=maximum)
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, str(count), ha="center", va="center", fontsize=13,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
               title=f"Held-out {policy}\ncost={row.total_cost:g}, t={row.threshold:.3f}")
    return fig
