from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from src.base import BaseClassifier
from src.base import sample_weights, positive_int, prediction_data, finite_scalar


@dataclass
class Node:
    proba: float
    n: float
    feature: Optional[int] = None
    threshold: Optional[float] = None
    categories: Optional[np.ndarray] = None
    children: list = field(default_factory=list)
    fractions: np.ndarray = field(default_factory=lambda: np.empty(0))
    gain: float = 0.0
    category_left: Optional[np.ndarray] = None


class DecisionTree(BaseClassifier):
    def __init__(
        self,
        criterion="gain_ratio",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        categorical_idx=None,
        min_impurity_decrease=0.0,
        leaf_smoothing=0.0,
    ):
        if criterion not in ("gain_ratio", "entropy", "gini"):
            raise ValueError("criterion must be 'gain_ratio', 'entropy', or 'gini'")
        if max_depth is not None:
            if isinstance(max_depth, (bool, np.bool_)) or not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
                raise ValueError("max_depth must be None or a nonnegative integer")
        positive_int(min_samples_split, "min_samples_split")
        positive_int(min_samples_leaf, "min_samples_leaf")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if categorical_idx is not None and any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in categorical_idx):
            raise ValueError("categorical_idx must contain integer indices")
        self.criterion = criterion
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_idx = [] if categorical_idx is None else [int(i) for i in categorical_idx]
        self.min_impurity_decrease = finite_scalar(min_impurity_decrease, "min_impurity_decrease")
        self.leaf_smoothing = finite_scalar(leaf_smoothing, "leaf_smoothing")

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or not X.size:
            raise ValueError("X must be a nonempty 2-D array")
        if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
            raise ValueError("y must be a binary 0/1 vector with one label per row")
        if np.isinf(X).any():
            raise ValueError("X may contain NaN (missing) but not infinite values")
        if any(not 0 <= i < X.shape[1] for i in self.categorical_idx):
            raise ValueError("categorical_idx contains an index outside X's columns")
        weights = sample_weights(sample_weight, len(X))
        keep = weights > 0
        X, y, weights = X[keep], y[keep], weights[keep]
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.is_categorical_ = np.isin(np.arange(X.shape[1]), self.categorical_idx)
        self.tree_weight_ = float(weights.sum())
        self.tree_ = self._grow(X, y.astype(int), weights, depth=0)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "tree_"):
            raise RuntimeError("Call fit before predict_proba")
        X = prediction_data(X, self.n_features_in_, allow_nan=True)
        probabilities = np.zeros(len(X))
        self._accumulate(self.tree_, X, np.arange(len(X)), np.ones(len(X)), probabilities)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return np.column_stack([1.0 - probabilities, probabilities])

    def _accumulate(self, node, X, rows, row_weights, probabilities):
        if not len(rows):
            return
        if node.feature is None:
            smoothing = getattr(self, 'leaf_smoothing', 0.0)
            probability = (node.proba * node.n + smoothing) / (node.n + 2 * smoothing)
            probabilities[rows] += row_weights * probability
            return
        column = X[rows, node.feature]
        if node.categories is None:
            branch = np.where(column <= node.threshold, 0, 1)
        else:
            position = np.minimum(np.searchsorted(node.categories, column), len(node.categories) - 1)
            branch = np.where(node.categories[position] == column, position, -1)
            if getattr(node, 'category_left', None) is not None:
                branch = np.where(branch >= 0, np.where(np.isin(column, node.category_left), 0, 1), -1)
        branch[np.isnan(column)] = -1
        unknown = branch < 0
        for branch_idx, (child, fraction) in enumerate(zip(node.children, node.fractions)):
            selected = (branch == branch_idx) | unknown
            child_weights = np.where(
                unknown[selected], row_weights[selected] * fraction, row_weights[selected]
            )
            self._accumulate(child, X, rows[selected], child_weights, probabilities)

    def _grow(self, X, y, weights, depth) -> Node:
        total_weight = float(weights.sum())
        positive_weight = float(weights @ y)
        node = Node(proba=positive_weight / total_weight, n=total_weight)
        if (
            min(positive_weight, total_weight - positive_weight) <= 1e-12 * total_weight
            or (self.max_depth is not None and depth >= self.max_depth)
            or total_weight < self.min_samples_split
        ):
            return node
        split = self._best_split(X, y, weights)
        if split is None:
            return node

        feature, gain, rule = split
        if gain * total_weight / self.tree_weight_ < self.min_impurity_decrease:
            return node
        column = X[:, feature]
        observed = ~np.isnan(column)
        if self.is_categorical_[feature]:
            if self.criterion == 'gini':
                node.categories, node.category_left = rule
                left = np.isin(column, node.category_left)
                branch_masks = [observed & left, observed & ~left]
            else:
                node.categories = rule
                branch_masks = [observed & (column == code) for code in rule]
        else:
            node.threshold = rule
            branch_masks = [observed & (column <= rule), observed & (column > rule)]
        branch_weights = np.array([weights[mask].sum() for mask in branch_masks])
        node.feature, node.gain = feature, gain
        node.fractions = branch_weights / branch_weights.sum()
        for mask, fraction in zip(branch_masks, node.fractions):
            rows = mask | ~observed
            child_weights = np.where(observed[rows], weights[rows], weights[rows] * fraction)
            node.children.append(self._grow(X[rows], y[rows], child_weights, depth + 1))
        return node

    def _best_split(self, X, y, weights):
        candidates = []
        for column_idx in range(X.shape[1]):
            if self.is_categorical_[column_idx]:
                result = self._split_categorical(X[:, column_idx], y, weights)
            else:
                result = self._split_numeric(X[:, column_idx], y, weights)
            if result is not None and result[0] > 1e-12:
                candidates.append((column_idx, *result))
        if not candidates:
            return None
        gains = np.array([gain for _, gain, _, _ in candidates])
        if self.criterion in ("entropy", "gini"):
            best = int(np.argmax(gains))
        else:
            gain_ratios = np.array([gain / split_info for _, gain, split_info, _ in candidates])
            gain_ratios[gains < gains.mean() - 1e-12] = -np.inf
            best = int(np.argmax(gain_ratios))
        feature, gain, _, rule = candidates[best]
        return feature, gain, rule

    def _split_numeric(self, column, y, weights):
        observed = ~np.isnan(column)
        order = np.argsort(column[observed], kind="stable")
        sorted_values = column[observed][order]
        sorted_weights = weights[observed][order]
        if not len(sorted_values):
            return None
        cumulative_weight = np.cumsum(sorted_weights)
        cumulative_positive = np.cumsum(sorted_weights * y[observed][order])
        observed_weight, observed_positive = cumulative_weight[-1], cumulative_positive[-1]

        cut = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
        left_weight, left_positive = cumulative_weight[cut], cumulative_positive[cut]
        right_weight = observed_weight - left_weight
        right_positive = observed_positive - left_positive
        valid = (left_weight >= self.min_samples_leaf) & (right_weight >= self.min_samples_leaf)
        if not valid.any():
            return None
        cut, left_weight, left_positive = cut[valid], left_weight[valid], left_positive[valid]
        right_weight, right_positive = right_weight[valid], right_positive[valid]

        parent_entropy = self._impurity(observed_positive, observed_weight)
        children_entropy = (
            left_weight * self._impurity(left_positive, left_weight)
            + right_weight * self._impurity(right_positive, right_weight)
        ) / observed_weight
        best = int(np.argmax(parent_entropy - children_entropy))
        total_weight = weights.sum()
        gain = observed_weight / total_weight * (parent_entropy - children_entropy[best])
        split_info = self._split_info(
            [left_weight[best], right_weight[best], total_weight - observed_weight], total_weight
        )
        threshold = sorted_values[cut[best]] / 2 + sorted_values[cut[best] + 1] / 2
        threshold = max(sorted_values[cut[best]], min(threshold, np.nextafter(sorted_values[cut[best] + 1], -np.inf)))
        return float(gain), split_info, float(threshold)

    def _split_categorical(self, column, y, weights):
        observed = ~np.isnan(column)
        categories, branch = np.unique(column[observed], return_inverse=True)
        if len(categories) < 2:
            return None
        branch_weights = np.bincount(branch, weights=weights[observed])
        branch_positive = np.bincount(branch, weights=weights[observed] * y[observed])
        if self.criterion == 'gini':
            
            order = np.argsort(branch_positive / branch_weights, kind='stable')
            ranks = np.empty(len(categories), dtype=float)
            ranks[order] = np.arange(len(categories))
            ranked = np.full(len(column), np.nan)
            ranked[observed] = ranks[branch]
            result = self._split_numeric(ranked, y, weights)
            if result is None:
                return None
            gain, split_info, cut = result
            return gain, split_info, (categories, categories[order[:int(np.floor(cut)) + 1]])
        if np.any(branch_weights < self.min_samples_leaf):
            return None
        observed_weight, total_weight = branch_weights.sum(), weights.sum()
        children_entropy = (
            branch_weights * self._impurity(branch_positive, branch_weights)
        ).sum() / observed_weight
        parent_entropy = self._impurity(branch_positive.sum(), observed_weight)
        gain = observed_weight / total_weight * (parent_entropy - children_entropy)
        split_info = self._split_info(
            [*branch_weights, total_weight - observed_weight], total_weight
        )
        return float(gain), split_info, categories

    @staticmethod
    def _entropy(positive, total):
        positive = np.asarray(positive, dtype=float)
        total = np.asarray(total, dtype=float)
        shape = np.broadcast(positive, total).shape
        p = np.divide(positive, total, out=np.zeros(shape), where=total > 0).clip(0.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -(
                np.where(p > 0, p * np.log2(p), 0.0)
                + np.where(p < 1, (1 - p) * np.log2(1 - p), 0.0)
            )
        return float(entropy) if entropy.ndim == 0 else entropy

    def _impurity(self, positive, total):
        if self.criterion != "gini":
            return self._entropy(positive, total)
        positive, total = np.asarray(positive, dtype=float), np.asarray(total, dtype=float)
        p = np.divide(positive, total, out=np.zeros(np.broadcast(positive, total).shape), where=total > 0).clip(0, 1)
        return 2 * p * (1 - p)

    @staticmethod
    def _split_info(branch_weights, total_weight):
        proportions = np.asarray(branch_weights, dtype=float) / total_weight
        proportions = proportions[proportions > 0]
        return float(-(proportions * np.log2(proportions)).sum())
