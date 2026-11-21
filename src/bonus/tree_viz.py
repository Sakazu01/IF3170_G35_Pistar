from __future__ import annotations

import matplotlib.pyplot as plt


def _count(node):
    return sum(1 + _count(child) for child in node.children)


def _branch_label(node, i, names, levels):
    if node.categories is None:
        return f"<= {node.threshold:.4g}" if i == 0 else f"> {node.threshold:.4g}"
    code = int(node.categories[i])
    level_names = levels.get(names[node.feature], [])
    return str(level_names[code]) if 0 <= code < len(level_names) else str(code)


def plot_tree(model, top_n=3, feature_names=None, category_levels=None, path=None):
    if int(top_n) < 0:
        raise ValueError("top_n must be nonnegative")
    names = list(feature_names) if feature_names is not None else [
        f"x{j}" for j in range(model.n_features_in_)
    ]
    levels = category_levels or {}
    boxes, edges, next_leaf = [], [], [0]

    def visit(node, depth):
        expand = node.feature is not None and depth < top_n
        child_x = [visit(child, depth + 1) for child in node.children] if expand else []
        if child_x:
            x = (child_x[0] + child_x[-1]) / 2
        else:
            x, next_leaf[0] = next_leaf[0], next_leaf[0] + 1
        for i, cx in enumerate(child_x):
            edges.append((x, depth, cx, depth + 1, _branch_label(node, i, names, levels)))
        stats = f"n={node.n:.0f}  p={node.proba:.2f}"
        if node.feature is None:
            text = f"daun\n{stats}"
        else:
            text = f"{names[node.feature]}\ngain={node.gain:.3f}\n{stats}"
            if not expand:
                text += f"\n... (+{_count(node)} simpul)"
        boxes.append((x, depth, text, node.proba))
        return x

    visit(model.tree_, 0)
    width, depth = next_leaf[0], max(d for _, d, _, _ in boxes)
    fig, ax = plt.subplots(figsize=(max(8, 2.0 * width), 1.9 * (depth + 1)))
    cmap = plt.get_cmap("coolwarm")
    for x0, d0, x1, d1, label in edges:
        ax.plot([x0, x1], [-d0, -d1], color="gray", lw=1, zorder=1)
        ax.text((x0 + x1) / 2, -(d0 + d1) / 2, label, ha="center", va="center", fontsize=7,
                bbox=dict(fc="white", ec="none", pad=0.5), zorder=2)
    for x, d, text, proba in boxes:
        ax.text(x, -d, text, ha="center", va="center", fontsize=8, zorder=3,
                bbox=dict(boxstyle="round", fc=cmap(proba), alpha=0.85, ec="black", lw=0.5))
    ax.set_xlim(-0.6, width - 0.4)
    ax.set_ylim(-depth - 0.6, 0.6)
    ax.axis("off")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.02, label="P(is_canceled=1)")
    ax.set_title(f"DecisionTree C4.5, {top_n} tingkat teratas ({_count(model.tree_) + 1} simpul total)")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
