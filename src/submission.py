import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import fit_candidate
from src.preprocessing import TARGET
from src.evaluation import decision_threshold


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_submission(submission, sample, test):
    if list(sample.columns) != ['id', TARGET]:
        raise ValueError('Sample submission must have columns id,is_canceled')
    if list(submission.columns) != ['id', TARGET]:
        raise ValueError('Submission columns must be id,is_canceled')
    if len(submission) != len(test) or len(sample) != len(test):
        raise ValueError('Submission must contain every Kaggle test row')
    if submission.isna().any().any() or not submission.id.is_unique or not test.id.is_unique:
        raise ValueError('Missing values or duplicate IDs in submission/test')
    if not submission.id.equals(sample.id) or set(sample.id) != set(test.id):
        raise ValueError('IDs must match sample submission in the exact same order')
    if not submission[TARGET].isin([0, 1]).all():
        raise ValueError('Labels must be binary 0/1')


def predict_bundle(bundle, frame):
    p = bundle['model'].predict_proba(bundle['preprocessor'].transform(frame))[:, 1]
    return (p >= decision_threshold(bundle['threshold'])).astype(int)


def save_final(candidate, threshold, train, test, sample, output, metadata=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    from dataclasses import replace, asdict
    candidate = replace(candidate, max_train=None)
    prep, model, timing = fit_candidate(candidate, train)
    bundle = {'preprocessor': prep, 'model': model, 'threshold': decision_threshold(threshold),
              'configuration': asdict(candidate), 'metadata': _json_safe(metadata or {}), 'timing': timing}
    labels = predict_bundle(bundle, test)
    by_id = pd.Series(labels, index=test.id)
    submission = pd.DataFrame({'id': sample.id, TARGET: by_id.loc[sample.id].to_numpy(dtype=int)})
    validate_submission(submission, sample, test)
    path = output / 'submission_pipeline.pkl'
    with path.open('wb') as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with path.open('rb') as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(labels, predict_bundle(restored, test))
    submission.to_csv(output / 'submission.csv', index=False)
    digest = hashlib.sha256((output / 'submission.csv').read_bytes()).hexdigest()
    meta = {k: v for k, v in bundle.items() if k not in ('model', 'preprocessor')}
    meta.update({'sha256': digest, 'n_train': len(train), 'n_test': len(test),
                 'model_class': type(model).__name__, 'serialization_identical': True,
                 'kaggle_uploaded': False})
    meta = _json_safe(meta)
    (output / 'submission_metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False), encoding='utf-8')
    return submission, bundle, meta
