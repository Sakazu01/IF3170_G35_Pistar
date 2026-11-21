from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from src.base import BaseClassifier


def sample_weights(sample_weight, n_samples):
    weights = np.ones(n_samples) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be finite, nonnegative, shape (n_samples,)")
    if not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("sample_weight must have a finite positive sum")
    return weights


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


class DecisionTree(BaseClassifier):
    def __init__(
        self,
        criterion="gain_ratio",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        categorical_idx=None,
    ):
        if criterion not in ("gain_ratio", "entropy"):
            raise ValueError("criterion must be 'gain_ratio' (C4.5) or 'entropy' (ID3)")
        if max_depth is not None and int(max_depth) < 0:
            raise ValueError("max_depth must be None or a nonnegative integer")
        if not min_samples_split > 0 or not min_samples_leaf > 0:
            raise ValueError("min_samples_split and min_samples_leaf must be positive")
        self.criterion = criterion
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_idx = [] if categorical_idx is None else [int(i) for i in categorical_idx]

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
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.is_categorical_ = np.isin(np.arange(X.shape[1]), self.categorical_idx)
        self.tree_ = self._grow(X, y.astype(int), weights, depth=0)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "tree_"):
            raise RuntimeError("Call fit before predict_proba")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n_samples, {self.n_features_in_})")
        probabilities = np.zeros(len(X))
        self._accumulate(self.tree_, X, np.arange(len(X)), np.ones(len(X)), probabilities)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return np.column_stack([1.0 - probabilities, probabilities])

    def _accumulate(self, node, X, rows, row_weights, probabilities):
        if not len(rows):
            return
        if node.feature is None:
            probabilities[rows] += row_weights * node.proba
            return
        column = X[rows, node.feature]
        if node.categories is None:
            branch = np.where(column <= node.threshold, 0, 1)
        else:
            position = np.minimum(np.searchsorted(node.categories, column), len(node.categories) - 1)
            branch = np.where(node.categories[position] == column, position, -1)
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
        column = X[:, feature]
        observed = ~np.isnan(column)
        if self.is_categorical_[feature]:
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
        if self.criterion == "entropy":
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

        parent_entropy = self._entropy(observed_positive, observed_weight)
        children_entropy = (
            left_weight * self._entropy(left_positive, left_weight)
            + right_weight * self._entropy(right_positive, right_weight)
        ) / observed_weight
        best = int(np.argmax(parent_entropy - children_entropy))
        total_weight = weights.sum()
        gain = observed_weight / total_weight * (parent_entropy - children_entropy[best])
        split_info = self._split_info(
            [left_weight[best], right_weight[best], total_weight - observed_weight], total_weight
        )
        threshold = (sorted_values[cut[best]] + sorted_values[cut[best] + 1]) / 2
        return float(gain), split_info, float(threshold)

    def _split_categorical(self, column, y, weights):
        observed = ~np.isnan(column)
        categories, branch = np.unique(column[observed], return_inverse=True)
        if len(categories) < 2:
            return None
        branch_weights = np.bincount(branch, weights=weights[observed])
        branch_positive = np.bincount(branch, weights=weights[observed] * y[observed])
        if np.sum(branch_weights >= self.min_samples_leaf) < 2:
            return None
        observed_weight, total_weight = branch_weights.sum(), weights.sum()
        children_entropy = (
            branch_weights * self._entropy(branch_positive, branch_weights)
        ).sum() / observed_weight
        parent_entropy = self._entropy(branch_positive.sum(), observed_weight)
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

    @staticmethod
    def _split_info(branch_weights, total_weight):
        proportions = np.asarray(branch_weights, dtype=float) / total_weight
        proportions = proportions[proportions > 0]
        return float(-(proportions * np.log2(proportions)).sum())
