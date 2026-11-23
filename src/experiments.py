from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation import evaluate, threshold_curve
from src.models import KNN, DecisionTree, LogisticRegressionScratch, SVMScratch, NaiveBayes
from src.preprocessing import BookingPreprocessor, TARGET, feature_groups
from src.bonus.tree_prune import prune

SEED = 42
METRIC = 'macro_f1'


@dataclass
class Candidate:
    name: str
    family: str
    params: dict
    view: str = 'matrix'
    groups: tuple = ('basic',)
    balanced: bool = False
    max_train: int | None = None
    ccp_alpha: float = 0.0
    select_k: int | None = None

    @property
    def key(self):
        fields = asdict(self)
        fields.pop('name')
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:12]


def split_development_holdout(train, seed=SEED):

    groups = feature_groups(train)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    development, holdout = next(splitter.split(train, train[TARGET], groups))
    assert not set(groups[development]).intersection(groups[holdout])
    return train.iloc[development].reset_index(drop=True), train.iloc[holdout].reset_index(drop=True)


def make_model(candidate, prep):
    classes = {'KNN': KNN, 'Tree': DecisionTree, 'Logistic': LogisticRegressionScratch,
               'SVM': SVMScratch, 'Bayes': NaiveBayes}
    params = dict(candidate.params)
    if candidate.family in ('Tree', 'Bayes'):
        params['categorical_idx'] = prep.categorical_idx_
    return classes[candidate.family](**params)


def training_subset(candidate, frame):
    y = frame[TARGET].to_numpy(dtype=int)
    if candidate.max_train is not None and len(frame) > candidate.max_train:
        rows, _ = train_test_split(np.arange(len(frame)), train_size=candidate.max_train,
                                   stratify=y, random_state=SEED)
        frame = frame.iloc[rows]
    return frame


def fit_candidate(candidate, frame):

    frame = training_subset(candidate, frame)
    y = frame[TARGET].to_numpy(dtype=int)
    started = perf_counter()
    prep = BookingPreprocessor(candidate.view, candidate.groups, candidate.select_k)
    X = prep.fit_transform(frame, y)
    prepared = perf_counter()
    model = make_model(candidate, prep)
    weights = compute_sample_weight('balanced', y) if candidate.balanced else None
    model.fit(X, y, sample_weight=weights)
    if candidate.ccp_alpha:
        model = prune(model, candidate.ccp_alpha)
    finished = perf_counter()
    timing = {'preprocess_time': prepared - started, 'fit_time': finished - prepared,
              'n_train': len(y), 'n_core_train': getattr(model, 'n_training_samples_', len(y)),
              'converged': getattr(model, 'converged_', None),
              'training_matrix_mb': X.nbytes / 1024**2,
              'estimated_kernel_peak_mb': getattr(model, 'kernel_peak_mb_', None)}
    return prep, model, timing


def choose_threshold(y, probabilities, low=0.2, high=0.8, metric=METRIC):

    if metric not in ('accuracy', 'macro_f1', 'f1_positive', 'roc_auc'):
        raise ValueError('Unsupported selection metric')
    if not 0 <= low < high <= 1:
        raise ValueError('Require 0 <= low < high <= 1')
    exact = threshold_curve(y, probabilities)

    def grid_scores(thresholds, stage):
        positions = np.searchsorted(exact.threshold.to_numpy(), thresholds, side='left')
        table = exact.iloc[np.minimum(positions, len(exact) - 1)].copy()
        table['threshold'], table['stage'] = thresholds, stage
        if metric == 'roc_auc':
            table['roc_auc'] = evaluate(y, probabilities)['roc_auc']
        return table

    coarse = np.unique(np.r_[np.arange(low, high + 1e-10, .05), .5, high])
    curve = grid_scores(coarse, 'coarse')
    best = curve[curve[metric] >= curve[metric].max() - 1e-12].copy()
    best['distance'] = abs(best.threshold - .5)
    center = float(best.sort_values(['distance', 'threshold']).iloc[0].threshold)
    fine = np.round(np.arange(max(low, center - .05), min(high, center + .05) + 1e-10, .01), 10)
    curve = pd.concat([curve, grid_scores(fine, 'fine')], ignore_index=True)
    best = curve[curve[metric] >= curve[metric].max() - 1e-12].copy()
    best['distance'] = abs(best.threshold - 0.5)
    cutoff = float(best.sort_values(['distance', 'threshold']).iloc[0].threshold)
    return cutoff, curve.sort_values('threshold').drop_duplicates('threshold')


class Search:
    def __init__(self, development, output, folds=3, metric=METRIC):
        self.metric = metric
        self.frame = development.reset_index(drop=True)
        self.y = self.frame[TARGET].to_numpy(dtype=int)
        groups = feature_groups(self.frame)
        self.folds = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
                          .split(self.frame, self.y, groups))
        for training, validation in self.folds:
            assert not set(groups[training]).intersection(groups[validation])
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows, self.candidates, self.oof = [], {}, {}

    def run(self, candidates, stage):
        for candidate in candidates:
            if candidate.key in self.candidates:
                continue
            print(f'{stage}: {candidate.name}', flush=True)
            oof = np.full(len(self.y), np.nan)
            for fold, (training, validation) in enumerate(self.folds):
                prep, model, timing = fit_candidate(candidate, self.frame.iloc[training])
                started = perf_counter()
                p = model.predict_proba(prep.transform(self.frame.iloc[validation]))[:, 1]
                predict_time = perf_counter() - started
                fitted_frame = training_subset(candidate, self.frame.iloc[training])
                train_scores = evaluate(fitted_frame[TARGET].to_numpy(), model.predict_proba(prep.transform(fitted_frame))[:, 1])
                oof[validation] = p
                self.rows.append({'key': candidate.key, 'candidate': candidate.name,
                                  'family': candidate.family, 'stage': stage, 'fold': fold,
                                  **evaluate(self.y[validation], p), **timing,
                                  'predict_time': predict_time})
                self.rows[-1].update({'train_' + k: v for k, v in train_scores.items()})
            assert np.isfinite(oof).all()
            self.candidates[candidate.key], self.oof[candidate.key] = candidate, oof
            pd.DataFrame(self.rows).to_csv(self.output / 'cv_folds.csv', index=False)
        return self.summary()

    def summary(self):
        rows = []
        for key, candidate in self.candidates.items():
            folds = pd.DataFrame(self.rows).query('key == @key')
            cutoff, _ = choose_threshold(self.y, self.oof[key], metric=self.metric)
            rows.append({'key': key, 'candidate': candidate.name, 'family': candidate.family,
                         'macro_f1_mean': folds.macro_f1.mean(),
                         'macro_f1_std': folds.macro_f1.std(ddof=1),
                         'accuracy_mean': folds.accuracy.mean(), 'accuracy_std': folds.accuracy.std(ddof=1),
                         'roc_auc_mean': folds.roc_auc.mean(), 'fit_time_mean': folds.fit_time.mean(),
                         'roc_auc_std': folds.roc_auc.std(ddof=1),
                         'f1_positive_mean': folds.f1_positive.mean(), 'f1_positive_std': folds.f1_positive.std(ddof=1),
                         'threshold_oof': cutoff,
                         'oof_macro_f1_tuned': evaluate(self.y, self.oof[key], cutoff)['macro_f1'],
                         'oof_accuracy_tuned': evaluate(self.y, self.oof[key], cutoff)['accuracy'],
                         'train_accuracy_mean': folds.train_accuracy.mean(),
                         'train_macro_f1_mean': folds.train_macro_f1.mean(),
                         'configuration': json.dumps(asdict(candidate), sort_keys=True)})
        return pd.DataFrame(rows).sort_values([self.metric + '_mean', self.metric + '_std', 'key'], ascending=[False, True, True])

    def best(self, family):
        row = self.summary().query('family == @family').iloc[0]
        return self.candidates[row.key]


def baseline_candidates():
    return [
        Candidate('Tree baseline C4.5 d10 leaf5', 'Tree', {'max_depth': 10, 'min_samples_leaf': 5}, view='tree'),
        Candidate('KNN baseline k15', 'KNN', {'k': 15}, max_train=5000),
        Candidate('Logistic baseline GD L2', 'Logistic', {'optimizer': 'gd', 'penalty': 'l2', 'reg_strength': .001}, max_train=5000),
        Candidate('SVM baseline RBF', 'SVM', {'kernel': 'rbf'}, max_train=2000),
        Candidate('Bayes baseline raw scale', 'Bayes', {}, view='tree'),
    ]


def run_search(development, output, search=None, metric=METRIC):
    from src.tuning import run_search as run_staged_search
    return run_staged_search(development, output, search, metric)


def evaluate_holdout(development, holdout, search, selected):

    rows, fitted = [], {}
    for kind, candidates in (('baseline', {c.family: c for c in baseline_candidates()}), ('optimized', selected)):
        for family, candidate in candidates.items():
            print('Holdout:', kind, candidate.name, flush=True)
            prep, model, timing = fit_candidate(candidate, development)
            started = perf_counter()
            p = model.predict_proba(prep.transform(holdout))[:, 1]
            elapsed = perf_counter() - started
            cutoff, _ = choose_threshold(search.y, search.oof[candidate.key], metric=search.metric)
            training_frame = training_subset(candidate, development)
            train_p = model.predict_proba(prep.transform(training_frame))[:, 1]
            for threshold_kind, threshold in (('default', .5), ('oof', cutoff)):
                rows.append({'family': family, 'phase': kind, 'threshold_kind': threshold_kind,
                             'threshold': threshold, 'candidate': candidate.name,
                             **evaluate(holdout[TARGET].to_numpy(), p, threshold), **timing,
                             'predict_time': elapsed})
                rows[-1].update({'train_' + k: v for k, v in evaluate(training_frame[TARGET].to_numpy(), train_p, threshold).items()})
            fitted[kind, family] = prep, model, p
    table = pd.DataFrame(rows)
    table.to_csv(search.output / 'holdout_results.csv', index=False)
    return table, fitted
