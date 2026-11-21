from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import COST_FN, COST_FP, SEED, TARGET
from src.preprocessing import Preprocessor, balanced_weights, split_xy
from src.validation import decision_threshold, finite_scalar, positive_int, random_seed

__all__ = [
    "confusion", "roc_auc", "pr_auc", "expected_cost", "evaluate",
    "threshold_curve", "best_threshold", "cross_validate", "benchmark",
]


def _labels(values, name="y"):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{name} must be a nonempty 1-D vector of binary 0/1 labels")
    return values.astype(int, copy=False)


def _scores(y, score, probabilities=False):
    y = _labels(y)
    score = np.asarray(score, dtype=float)
    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("Scores must be finite with one value per label")
    if probabilities and np.any((score < 0) | (score > 1)):
        raise ValueError("Probabilities must be in [0,1]")
    return y, score


def confusion(y, pred):
    y, pred = _labels(y), _labels(pred, "pred")
    if y.shape != pred.shape:
        raise ValueError("y and pred must have the same shape")
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    return tn, fp, fn, tp


def roc_auc(y, score):
    y, score = _scores(y, score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pr_auc(y, score):
    y, score = _scores(y, score)
    n_positive = int(y.sum())
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    sorted_score, sorted_y = score[order], y[order]
    endpoints = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(y) - 1]
    true_positives = np.cumsum(sorted_y)[endpoints]
    recall = np.r_[0.0, true_positives / n_positive]
    precision = np.r_[1.0, true_positives / (endpoints + 1)]
    return float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) / 2))


def expected_cost(y, pred, fp_cost=COST_FP, fn_cost=COST_FN):
    fp_cost = finite_scalar(fp_cost, "fp_cost")
    fn_cost = finite_scalar(fn_cost, "fn_cost")
    tn, fp, fn, tp = confusion(y, pred)
    return float((fp_cost * fp + fn_cost * fn) / (tn + fp + fn + tp))


def evaluate(y, proba, threshold=0.5):
    y, proba = _scores(y, proba, probabilities=True)
    threshold = decision_threshold(threshold)
    tn, fp, fn, tp = confusion(y, proba >= threshold)
    return {
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "roc_auc": roc_auc(y, proba),
        "pr_auc": pr_auc(y, proba),
        "expected_cost": float((COST_FP * fp + COST_FN * fn) / len(y)),
    }


def threshold_curve(y, proba):
    y, proba = _scores(y, proba, probabilities=True)
    thresholds = np.unique(np.r_[0.0, proba, 0.5, np.nextafter(proba.max(), np.inf)])
    order = np.argsort(proba, kind="stable")
    positions = np.searchsorted(proba[order], thresholds, side="left")
    cumulative_positive = np.r_[0, np.cumsum(y[order])]
    cumulative_negative = np.r_[0, np.cumsum(1 - y[order])]
    fn = cumulative_positive[positions]
    tn = cumulative_negative[positions]
    tp = cumulative_positive[-1] - fn
    fp = cumulative_negative[-1] - tn
    return pd.DataFrame({
        "threshold": thresholds,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / np.maximum(tp + fp, 1),
        "recall": tp / np.maximum(tp + fn, 1),
        "f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1),
        "cost": COST_FP * fp + COST_FN * fn,
        "expected_cost": (COST_FP * fp + COST_FN * fn) / len(y),
    })


def best_threshold(y, proba, metric="f1"):
    if metric in ("roc_auc", "pr_auc"):
        _scores(y, proba, probabilities=True)
        return 0.5
    if metric not in ("accuracy", "precision", "recall", "f1", "cost", "expected_cost"):
        raise ValueError("metric must be accuracy, precision, recall, f1, roc_auc, pr_auc, or cost/expected_cost")
    curve = threshold_curve(y, proba)
    if metric in ("cost", "expected_cost"):
        index = int(np.argmin(curve[metric].to_numpy()))
    else:
        index = int(np.argmax(curve[metric].to_numpy()))
    return float(curve.threshold.iloc[index])


def _stratified_kfold(y, k, seed):
    k = positive_int(k, "k")
    seed = random_seed(seed)
    if k < 2 or k > len(y):
        raise ValueError("k must be between 2 and the number of samples")
    if np.bincount(y, minlength=2).min() < k:
        raise ValueError("Each binary class must have at least k samples for stratified cross-validation")
    rng = np.random.default_rng(seed)
    class_folds = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        class_folds.append(np.array_split(indices, k))
    all_indices = np.arange(len(y))
    for fold in range(k):
        validation = np.concatenate([parts[fold] for parts in class_folds])
        training = np.setdiff1d(all_indices, validation, assume_unique=True)
        yield training, np.sort(validation)


def _positive_probabilities(model, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.ndim != 2 or not hasattr(model, "classes_"):
        raise ValueError("predict_proba must return shape (n,) or class-labeled probability columns")
    classes = _labels(model.classes_, "classes_")
    if probabilities.shape[1] != len(classes) or len(np.unique(classes)) != len(classes):
        raise ValueError("Probability columns must match unique binary classes_")
    if (
        not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("Probability rows must contain finite values in [0,1] summing to one")
    positive = np.flatnonzero(classes == 1)
    return probabilities[:, positive[0]] if len(positive) else np.zeros(len(probabilities))


def _fit_and_evaluate(model, X_tr, y_tr, X_va, y_va, sample_weight=None):
    y_tr, y_va = _labels(y_tr), _labels(y_va)
    if len(X_tr) != len(y_tr) or len(X_va) != len(y_va):
        raise ValueError("Training and validation features must match their label counts")
    started = perf_counter()
    if sample_weight is None:
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
    fitted = perf_counter()
    probabilities = model.predict_proba(X_va)
    predicted = perf_counter()
    proba = _positive_probabilities(model, probabilities)
    return {
        **evaluate(y_va, proba),
        "fit_time": fitted - started,
        "predict_time": predicted - fitted,
    }


def cross_validate(make_model, df, mode, k=5, seed=SEED, pre_kw=None, balance=False):
    if not callable(make_model):
        raise TypeError("make_model must accept a fitted preprocessor and return a new model")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if TARGET not in df:
        raise ValueError(f"df must contain the target column {TARGET!r}")
    y = _labels(df[TARGET].to_numpy())
    X_raw, _ = split_xy(df)
    if pre_kw is not None and not isinstance(pre_kw, Mapping):
        raise TypeError("pre_kw must be a mapping of Preprocessor arguments or None")
    options = dict(pre_kw) if pre_kw is not None else {}
    if "mode" in options:
        raise ValueError("Pass mode directly, rather than through pre_kw")
    rows = []
    for fold, (training, validation) in enumerate(_stratified_kfold(y, k, seed)):
        preprocessor = Preprocessor(mode, **options).fit(X_raw.iloc[training], y[training])
        X_tr = preprocessor.transform(X_raw.iloc[training])
        X_va = preprocessor.transform(X_raw.iloc[validation])
        model = make_model(preprocessor)
        weights = balanced_weights(y[training]) if balance else None
        rows.append({
            "fold": fold,
            **_fit_and_evaluate(model, X_tr, y[training], X_va, y[validation], weights),
        })
    return pd.DataFrame(rows)


def benchmark(name, impl, model, X_tr, y_tr, X_va, y_va):
    return {
        "model": name,
        "impl": impl,
        **_fit_and_evaluate(model, X_tr, y_tr, X_va, y_va),
    }
