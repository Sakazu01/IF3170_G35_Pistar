from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import Candidate, Search, baseline_candidates, choose_threshold, METRIC, SEED
from src.evaluation import evaluate


def _best_trees(search, count):
    summary = search.summary().query("family == 'Tree'")
    return [search.candidates[key] for key in summary.key.head(count)]


def run_search(development, output, search=None, metric=METRIC):

    output = Path(output)
    screen = Search(development, output, folds=3, metric=metric) if search is None else search
    screen.metric = metric
    baselines = baseline_candidates()
    screen.run(baselines, 'baseline')
    prior = Candidate('Previous selected C4.5', 'Tree',
                      dict(criterion='gain_ratio', max_depth=13, min_samples_leaf=10, min_samples_split=10),
                      view='tree', groups=('basic', 'history'), balanced=True)
    coarse = [prior]
    for view in ('tree', 'tree_imputed', 'matrix'):
        for criterion in ('gain_ratio', 'entropy', 'gini'):
            for depth, leaf in ((10, 10), (15, 5), (21, 20)):
                coarse.append(Candidate(f'{view} {criterion} d{depth} leaf{leaf}', 'Tree',
                                        dict(criterion=criterion, max_depth=depth, min_samples_leaf=leaf), view=view))
    screen.run(coarse, 'tree_coarse')
    winners, seen = [], set()
    for c in _best_trees(screen, len(coarse) + 1):
        signature = (c.view, c.params.get('criterion', 'gain_ratio'))
        if signature not in seen:
            winners.append(c)
            seen.add(signature)
        if len(winners) == 2:
            break
    refine = []
    for c in winners:
        for leaf in (1, 3, 5, 8, 10, 15, 20, 30):
            refine.append(replace(c, name=f'{c.view} {c.params["criterion"]} leaf{leaf}', params={**c.params, 'min_samples_leaf': leaf}))
        for depth in (8, 10, 12, 13, 15, 18, 21, 24):
            refine.append(replace(c, name=f'{c.view} {c.params["criterion"]} depth{depth}', params={**c.params, 'max_depth': depth}))
        refine.append(replace(c, name=c.name + ' balanced', balanced=True))
    screen.run(refine, 'tree_refine')
    winner = _best_trees(screen, 1)[0]
    regularized = [replace(winner, name=f'Tree minsplit{v}', params={**winner.params, 'min_samples_split': v}) for v in (2, 5, 10, 20, 40)]
    regularized += [replace(winner, name=f'Tree smoothing{v}', params={**winner.params, 'leaf_smoothing': v}) for v in (.5, 1., 2., 5.)]
    regularized += [replace(winner, name=f'Tree impurity{v}', params={**winner.params, 'min_impurity_decrease': v}) for v in (.00005, .0001, .0005)]
    regularized += [replace(winner, name=f'Tree pruning{v}', ccp_alpha=v) for v in (.00005, .0001, .0002, .0005)]
    screen.run(regularized, 'tree_regularization')
    winner = _best_trees(screen, 1)[0]
    groups = [(), ('basic',), ('basic', 'stay'), ('basic', 'history'), ('basic', 'price'),
              ('basic', 'season', 'calendar'), ('basic', 'interactions'),
              ('basic', 'stay', 'history', 'price'),
              ('basic', 'stay', 'history', 'price', 'season', 'calendar', 'interactions')]
    ablations = [replace(winner, name='Ablation ' + ('+'.join(g) or 'raw'), groups=g) for g in groups]
    ablations += [replace(winner, name='Encoding ablation ' + view, view=view, groups=()) for view in ('tree', 'tree_imputed', 'matrix')]
    screen.run(ablations, 'feature_ablation')
    table = screen.summary()
    tested = table[table.key.isin([c.key for c in ablations])].copy()
    tested.to_csv(output / 'feature_ablation.csv', index=False)
    improved = [c for c in ablations if len(c.groups) > 1 and
                float(table.set_index('key').loc[c.key, metric + '_mean']) >
                float(table.set_index('key').loc[ablations[1].key, metric + '_mean'])]
    if improved:
        combined = tuple(dict.fromkeys(g for c in improved for g in c.groups))
        screen.run([replace(winner, name='Ablation beneficial combination', groups=combined)], 'feature_combination')
    others = [
        Candidate('KNN full k15 distance', 'KNN', dict(k=15, weighted=True)),
        Candidate('KNN full k31 distance', 'KNN', dict(k=31, weighted=True)),
        Candidate('KNN Manhattan k15', 'KNN', dict(k=15, metric='manhattan', weighted=True), max_train=5000),
        Candidate('LR full GD', 'Logistic', dict(optimizer='gd', learning_rate=.3, max_iter=3000, penalty='l2', reg_strength=.001)),
        Candidate('LR full Adam', 'Logistic', dict(optimizer='adam', learning_rate=.03, max_iter=1000, penalty='l2', reg_strength=.001)),
        Candidate('LR full interactions', 'Logistic', dict(optimizer='adam', learning_rate=.03, max_iter=1000, penalty='elasticnet', reg_strength=.001), groups=('basic', 'stay', 'history', 'price', 'interactions')),
        Candidate('LR MI top25', 'Logistic', dict(optimizer='adam', learning_rate=.03, max_iter=1000, penalty='l1', reg_strength=.001), select_k=25),
        Candidate('SVM RBF 3000 calibrated', 'SVM', dict(kernel='rbf', C=3., max_iter=300, calibration_fraction=.15), max_train=3000),
        Candidate('SVM poly calibrated', 'SVM', dict(kernel='poly', degree=2, C=1., max_iter=200, calibration_fraction=.15), max_train=2000),
        Candidate('SVM linear calibrated', 'SVM', dict(kernel='linear', C=1., max_iter=200, calibration_fraction=.15), max_train=2000),
    ]
    others += [Candidate(f'Bayes scaled {v}', 'Bayes', dict(var_smoothing=v), view='bayes') for v in (1e-9, .001, .01)]
    screen.run(others, 'other_models')
    screened = screen.summary()
    screened.to_csv(output / 'screening_summary.csv', index=False)
    pd.DataFrame(screen.rows).to_csv(output / 'screening_folds.csv', index=False)
    finalists = _best_trees(screen, 8) + [prior] + baselines
    accuracy_keys = screened.query("family == 'Tree'").sort_values(['accuracy_mean', 'accuracy_std', 'key'], ascending=[False, True, True]).key.head(2)
    finalists += [screen.candidates[key] for key in accuracy_keys]
    finalists += [screen.best(family) for family in ('KNN', 'Logistic', 'SVM', 'Bayes')]
    confirm = Search(development, output, folds=5, metric=metric)
    confirm.run(finalists, 'confirmation_5fold')
    summary = confirm.summary()
    summary.to_csv(output / 'cv_summary.csv', index=False)
    selected = {}
    indexed = summary.set_index('key')
    for baseline in baselines:
        best = confirm.best(baseline.family)
        selected[baseline.family] = best if indexed.loc[best.key, metric + '_mean'] > indexed.loc[baseline.key, metric + '_mean'] + .0005 else baseline
    final = max(selected.values(), key=lambda c: float(indexed.loc[c.key, metric + '_mean']))
    cutoff, curve = choose_threshold(confirm.y, confirm.oof[final.key], metric=metric)
    curve.to_csv(output / 'threshold_oof.csv', index=False)
    diagnostics = []
    for fold, (_, validation) in enumerate(confirm.folds):
        for t in curve.threshold.unique():
            diagnostics.append(dict(fold=fold, threshold=t, **evaluate(confirm.y[validation], confirm.oof[final.key][validation], t)))
    pd.DataFrame(diagnostics).to_csv(output / 'threshold_stability.csv', index=False)
    np.savez_compressed(output / 'oof_predictions.npz', y=confirm.y, **confirm.oof)
    frozen = dict(metric=metric, kaggle_metric_verified=False, seed=SEED, folds=5,
                  selection_rule='3-fold screening; 5-fold mean score at 0.5; OOF coarse/fine threshold',
                  selected={family: asdict(c) for family, c in selected.items()}, final=asdict(final), threshold=cutoff,
                  n_development=len(development), screening_candidates=len(screened), confirmation_candidates=len(summary),
                  holdout_previously_observed=True)
    (output / 'selection.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    return confirm, selected, final, cutoff
