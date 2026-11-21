import pickle
from pathlib import Path

import numpy as np


class BaseClassifier:
    def predict(self, X):
        proba = np.asarray(self.predict_proba(X))
        return (proba[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_params(self):
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

