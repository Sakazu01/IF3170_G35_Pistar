import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from src.experiments import Candidate, run_search, split_development_holdout, evaluate_holdout
from src.submission import save_final


def generate(root, retune=False, metric='macro_f1'):
    root = Path(root)
    output = root / 'output'
    experiment_dir = output / 'laporan/evaluasi'
    train, test, sample = [pd.read_csv(root / 'data' / name, dtype={'id': str})
                           for name in ('train.csv', 'test.csv', 'sample_submission.csv')]
    selection_path = experiment_dir / 'selection.json'
    if retune or not selection_path.is_file():
        development, holdout = split_development_holdout(train)
        search, selected, final, threshold = run_search(development, experiment_dir, metric=metric)
        evaluate_holdout(development, holdout, search, selected)
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
    else:
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        final, threshold = Candidate(**selection['final']), selection['threshold']
    hashes = {name: hashlib.sha256((root / 'data' / name).read_bytes()).hexdigest()
              for name in ('train.csv', 'test.csv', 'sample_submission.csv')}
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((root / 'src').rglob('*.py'))}
    summary = pd.read_csv(experiment_dir / 'cv_summary.csv')
    row = summary.set_index('key').loc[final.key]
    holdout_scores = pd.read_csv(experiment_dir / 'holdout_results.csv')
    selected_holdout = holdout_scores.query("family == @final.family and phase == 'optimized' and threshold_kind == 'oof'")
    metadata = dict(input_sha256=hashes, source_sha256=sources, selection=selection,
                    cv=row.to_dict(), holdout=selected_holdout.to_dict('records'),
                    official_kaggle_metric_verified=selection['kaggle_metric_verified'])
    submission, bundle, meta = save_final(final, threshold, train, test, sample, output, metadata)
    result = dict(path=str(output / 'submission.csv'), rows=len(submission), columns=list(submission.columns),
                  predictions={str(k): int(v) for k, v in submission.is_canceled.value_counts().items()},
                  sha256=meta['sha256'], serialization_identical=meta['serialization_identical'],
                  configuration=asdict(final), threshold=threshold)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Reproduce the frozen scratch submission, or rerun development-only tuning.')
    parser.add_argument('--retune', action='store_true')
    parser.add_argument('--metric', choices=['macro_f1', 'accuracy', 'f1_positive', 'roc_auc'], default='macro_f1')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        generate(Path(__file__).resolve().parents[1], args.retune, args.metric)


if __name__ == '__main__':
    main()
