from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class BaseClassifier:
    threshold = 0.5

    def predict(self, X):
        return (np.asarray(self.predict_proba(X)) >= self.threshold).astype(int)

    def get_params(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.endswith("_") and not key.startswith("_")
        }

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

