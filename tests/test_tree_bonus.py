import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.bonus.tree_prune import cost_complexity_path, prune, tune_ccp_alpha
from src.bonus.tree_viz import plot_tree
from src.models.decision_tree import DecisionTree


def leaves(node):
    return 1 if node.feature is None else sum(leaves(child) for child in node.children)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.8, size=300) > 0).astype(int)
    return X, y


def test_path_matches_sklearn(data):
    X, y = data[0][:, :1], data[1]
    ours = DecisionTree(criterion="entropy").fit(X, y)
    ref = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
    assert leaves(ours.tree_) == ref.get_n_leaves()
    np.testing.assert_allclose(
        cost_complexity_path(ours).ccp_alpha.unique(),
        np.unique(ref.cost_complexity_pruning_path(X, y).ccp_alphas),
        atol=1e-10,
    )


def test_prune_extremes_and_original_untouched(data):
    X, y = data
    model = DecisionTree().fit(X, y)
    before = model.predict_proba(X)
    path = cost_complexity_path(model)
    assert path.ccp_alpha.is_monotonic_increasing and path.n_leaves.is_monotonic_decreasing
    np.testing.assert_allclose(prune(model, 0.0).predict_proba(X), before)
    stump = prune(model, path.ccp_alpha.max())
    assert stump.tree_.feature is None
    np.testing.assert_allclose(stump.predict_proba(X)[:, 1], y.mean())
    np.testing.assert_allclose(model.predict_proba(X), before)


def test_tune_picks_best_validation_score(data):
    X, y = data
    model = DecisionTree().fit(X[:200], y[:200])
    best, table = tune_ccp_alpha(model, X[200:], y[200:])
    assert leaves(best.tree_) <= leaves(model.tree_)
    assert best.predict_proba(X[200:]).shape == (100, 2)
    assert table.roc_auc.max() >= table.roc_auc.iloc[0]


def test_plot_tree_saves_figure(data, tmp_path):
    X, y = data
    X = np.column_stack([X, (X[:, 2] > 0).astype(float)])
    model = DecisionTree(categorical_idx=[3]).fit(X, y)
    out = tmp_path / "tree.png"
    plot_tree(model, top_n=2, feature_names=["a", "b", "c", "flag"],
              category_levels={"flag": ["no", "yes"]}, path=out)
    assert out.stat().st_size > 0
