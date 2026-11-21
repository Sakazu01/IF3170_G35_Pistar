from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.evaluation import evaluate
from src.models.decision_tree import DecisionTree


def _preorder(root):
    nodes, stack = [], [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(reversed(node.children))
    return nodes


def _pruning_sequence(root):
    nodes = _preorder(root)
    node_index = {id(node): node_idx for node_idx, node in enumerate(nodes)}
    parent = np.full(len(nodes), -1)
    for node_idx, node in enumerate(nodes):
        for child in node.children:
            parent[node_index[id(child)]] = node_idx
    subtree_size = np.ones(len(nodes), dtype=int)
    for node_idx in range(len(nodes) - 1, 0, -1):
        subtree_size[parent[node_idx]] += subtree_size[node_idx]

    risk = np.array([node.n / root.n * DecisionTree._entropy(node.proba, 1.0) for node in nodes])
    internal = np.array([node.feature is not None for node in nodes])
    subtree_risk = np.where(internal, 0.0, risk)
    subtree_leaves = (~internal).astype(float)
    for node_idx in range(len(nodes) - 1, 0, -1):
        subtree_risk[parent[node_idx]] += subtree_risk[node_idx]
        subtree_leaves[parent[node_idx]] += subtree_leaves[node_idx]

    steps, path, alpha = [], [(0.0, int(subtree_leaves[0]))], 0.0
    while internal[0]:
        with np.errstate(divide="ignore", invalid="ignore"):
            link_strength = np.where(
                internal, (risk - subtree_risk) / (subtree_leaves - 1), np.inf
            )
        alpha = max(alpha, float(link_strength.min()))
        for node_idx in np.flatnonzero(internal & (link_strength <= alpha + 1e-12)):
            if not internal[node_idx]:
                continue
            risk_change = risk[node_idx] - subtree_risk[node_idx]
            leaves_change = 1.0 - subtree_leaves[node_idx]
            internal[node_idx:node_idx + subtree_size[node_idx]] = False
            ancestor = node_idx
            while ancestor >= 0:
                subtree_risk[ancestor] += risk_change
                subtree_leaves[ancestor] += leaves_change
                ancestor = parent[ancestor]
            steps.append((alpha, int(node_idx)))
        path.append((alpha, int(subtree_leaves[0])))
    return steps, path


def _apply(model, steps, ccp_alpha):
    pruned = copy.deepcopy(model)
    nodes = _preorder(pruned.tree_)
    for alpha, node_idx in steps:
        if alpha > ccp_alpha:
            break
        node = nodes[node_idx]
        node.feature = node.threshold = node.categories = None
        node.children, node.fractions, node.gain = [], np.empty(0), 0.0
    return pruned


def cost_complexity_path(model):
    _, path = _pruning_sequence(model.tree_)
    return pd.DataFrame(path, columns=["ccp_alpha", "n_leaves"])


def prune(model, ccp_alpha):
    steps, _ = _pruning_sequence(model.tree_)
    return _apply(model, steps, ccp_alpha)


def tune_ccp_alpha(model, X_val, y_val, metric="roc_auc", max_points=40):
    steps, path = _pruning_sequence(model.tree_)
    picked = np.linspace(0, len(path) - 1, min(max_points, len(path))).round().astype(int)
    rows = []
    for alpha, n_leaves in (path[path_idx] for path_idx in np.unique(picked)):
        probabilities = _apply(model, steps, alpha).predict_proba(X_val)
        rows.append({"ccp_alpha": alpha, "n_leaves": n_leaves, **evaluate(y_val, probabilities)})
    table = pd.DataFrame(rows)
    score = (-table[metric] if metric == "expected_cost" else table[metric]).round(12)
    best_alpha = float(table.ccp_alpha[score == score.max()].max())
    return _apply(model, steps, best_alpha), table


def plot_pruning(table, metric="roc_auc", path=None):
    import matplotlib.pyplot as plt

    table = table[table.ccp_alpha > 0]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    top.plot(table.ccp_alpha, table[metric], marker="o", ms=3)
    top.set_ylabel(f"{metric} (validasi)")
    best = table.loc[table[metric].idxmax()]
    top.axvline(best.ccp_alpha, ls="--", c="gray", label=f"alpha={best.ccp_alpha:.2e}")
    top.legend()
    bottom.step(table.ccp_alpha, table.n_leaves, where="post")
    bottom.set_ylabel("jumlah daun")
    bottom.set_yscale("log")
    bottom.set_xlabel("ccp_alpha")
    bottom.set_xscale("log")
    fig.suptitle("Cost-complexity pruning DecisionTree (C4.5)")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
