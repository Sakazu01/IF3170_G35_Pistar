import numpy as np
import pytest

from src.models.decision_tree import DecisionTree

PLAY_TENNIS = np.array([
    [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 1], [2, 1, 0, 0, 1],
    [2, 2, 1, 0, 1], [2, 2, 1, 1, 0], [1, 2, 1, 1, 1], [0, 1, 0, 0, 0],
    [0, 2, 1, 0, 1], [2, 1, 1, 0, 1], [0, 1, 1, 1, 1], [1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1], [2, 1, 0, 1, 0],
], dtype=float)
X_TENNIS, Y_TENNIS = PLAY_TENNIS[:, :4], PLAY_TENNIS[:, 4].astype(int)
OUTLOOK, TEMPERATURE, HUMIDITY, WIND = range(4)


def tennis_tree(criterion):
    return DecisionTree(criterion=criterion, categorical_idx=[0, 1, 2, 3]).fit(X_TENNIS, Y_TENNIS)


def test_entropy_matches_slide():
    entropy = DecisionTree._entropy
    assert entropy(4, 4) == 0.0
    assert entropy(2, 4) == pytest.approx(1.0)
    assert entropy(1, 4) == pytest.approx(0.811, abs=1e-3)
    assert entropy(2, 6) == pytest.approx(0.918, abs=1e-3)
    assert entropy(9, 14) == pytest.approx(0.940, abs=1e-3)


def test_information_gain_matches_slide():
    model, w = DecisionTree(), np.ones(14)
    gains = [model._split_categorical(X_TENNIS[:, j], Y_TENNIS, w)[0] for j in range(4)]
    assert gains == pytest.approx([0.246, 0.029, 0.151, 0.048], abs=1e-3)

    sunny = X_TENNIS[:, OUTLOOK] == 0
    gains = [model._split_categorical(X_TENNIS[sunny, j], Y_TENNIS[sunny], w[sunny])[0]
             for j in (TEMPERATURE, HUMIDITY, WIND)]
    assert gains == pytest.approx([0.571, 0.971, 0.020], abs=1e-3)


@pytest.mark.parametrize("criterion", ["entropy", "gain_ratio"])
def test_tree_matches_slide(criterion):
    root = tennis_tree(criterion).tree_
    assert root.feature == OUTLOOK
    sunny, overcast, rain = root.children
    assert sunny.feature == HUMIDITY
    assert [leaf.proba for leaf in sunny.children] == [0.0, 1.0]
    assert overcast.feature is None and overcast.proba == 1.0
    assert rain.feature == WIND
    assert [leaf.proba for leaf in rain.children] == [1.0, 0.0]
    np.testing.assert_array_equal(tennis_tree(criterion).predict(X_TENNIS), Y_TENNIS)


def test_numeric_threshold_is_midpoint():
    temperature = np.array([40, 48, 60, 72, 80, 90], dtype=float)
    play = np.array([0, 0, 1, 1, 1, 0])
    _, _, threshold = DecisionTree()._split_numeric(temperature, play, np.ones(6))
    assert threshold == 54.0
    model = DecisionTree(max_depth=1).fit(temperature[:, None], play)
    assert model.tree_.threshold == 54.0


def test_missing_values_and_unseen_categories():
    X = np.array([[0], [0], [0], [1], [1], [1], [np.nan]], dtype=float)
    y = np.array([1, 1, 1, 0, 0, 0, 1])
    model = DecisionTree(categorical_idx=[0]).fit(X, y)
    root = model.tree_
    assert root.feature == 0 and root.fractions == pytest.approx([0.5, 0.5])
    assert [c.proba for c in root.children] == pytest.approx([1.0, 0.5 / 3.5])
    assert root.gain == pytest.approx(6 / 7)

    expected = 0.5 * 1.0 + 0.5 * (0.5 / 3.5)
    np.testing.assert_allclose(model.predict_proba([[np.nan], [7.0]]), [expected, expected])

    X = X_TENNIS.copy()
    X[[0, 5, 9], OUTLOOK] = np.nan
    X[3, HUMIDITY] = np.nan
    proba = DecisionTree(categorical_idx=[0, 1, 2, 3]).fit(X, Y_TENNIS).predict_proba(X)
    assert proba.shape == (14,) and np.all((proba >= 0) & (proba <= 1))


def test_sample_weight_equals_duplication():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    X[rng.random((60, 3)) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, 0]) + rng.normal(scale=0.5, size=60) > 0).astype(int)
    weights = rng.integers(1, 4, size=60)
    weighted = DecisionTree(max_depth=4).fit(X, y, sample_weight=weights)
    repeated = DecisionTree(max_depth=4).fit(np.repeat(X, weights, axis=0), np.repeat(y, weights))
    np.testing.assert_allclose(weighted.predict_proba(X), repeated.predict_proba(X))


def test_stopping_rules():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(200, 4)), rng.integers(0, 2, size=200)

    def depth(node):
        return 0 if node.feature is None else 1 + max(depth(c) for c in node.children)

    def leaves(node):
        return [node] if node.feature is None else [l for c in node.children for l in leaves(c)]

    assert depth(DecisionTree(max_depth=3).fit(X, y).tree_) <= 3
    assert min(l.n for l in leaves(DecisionTree(min_samples_leaf=20).fit(X, y).tree_)) >= 20
    assert DecisionTree(min_samples_split=1000).fit(X, y).tree_.feature is None


def test_save_load_roundtrip(tmp_path):
    model = tennis_tree("gain_ratio")
    loaded = DecisionTree.load(model.save(tmp_path / "dtl.pkl"))
    np.testing.assert_array_equal(loaded.predict_proba(X_TENNIS), model.predict_proba(X_TENNIS))


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        DecisionTree(criterion="gini")
    with pytest.raises(ValueError):
        DecisionTree().fit(X_TENNIS, Y_TENNIS + 1)
    with pytest.raises(ValueError):
        DecisionTree().fit(X_TENNIS, Y_TENNIS).predict_proba(X_TENNIS[:, :2])
