import pickle
from pathlib import Path

import numpy as np


class BaseClassifier:


    def predict(self, X, threshold=0.5):

        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., np.inf):
            raise ValueError('threshold must be in [0, 1] or nextafter(1, +inf)')
        proba = np.asarray(self.predict_proba(X))
        return (proba[:, 1] >= threshold).astype(int)

    def score(self, X, y):
        prediction, y = self.predict(X), np.asarray(y)
        if y.shape != prediction.shape or not len(y) or not np.isin(y, [0, 1]).all():
            raise ValueError('y must contain one binary label per prediction')
        return float(np.mean(prediction == y))

    def get_params(self, deep=True):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.endswith("_") and not key.startswith("_")
        }

    def set_params(self, **params):
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}")
        self.__init__(**{**self.get_params(), **params})
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model

