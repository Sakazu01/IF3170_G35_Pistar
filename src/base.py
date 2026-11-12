"""Kelas dasar model dan utilitas bersama (seed, k-fold).

Semua model punya antarmuka yang sama, jadi kode evaluasi cukup ditulis
sekali dan tidak perlu tahu modelnya apa.
"""
import pickle
import random
from abc import ABC, abstractmethod

import numpy as np

from src.config import SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


class BaseClassifier(ABC):
    """Klasifikasi biner, y = 0 atau 1, kelas positif = 1.

    Aturan penamaan (sama seperti sklearn):
      - parameter konstruktor: tanpa underscore di belakang
      - hasil belajar: pakai underscore di belakang (coef_, tree_, ...)
      - fit() mengembalikan self
    """

    threshold: float = 0.5  # bisa digeser untuk cost-sensitive

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> "BaseClassifier":
        """Latih model, kembalikan self."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(y=1) tiap baris, shape (n,). SVM mengembalikan skor terkalibrasi."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def get_params(self) -> dict:
        """Parameter konstruktor saja, hasil belajar dan atribut privat dilewati."""
        params = {}
        for name, value in vars(self).items():
            is_learned = name.endswith("_")
            is_private = name.startswith("_")
            if not is_learned and not is_private:
                params[name] = value
        return params

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BaseClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)


def stratified_kfold(y: np.ndarray, k: int = 5, seed: int = SEED):
    """Hasilkan k pasang (train_idx, val_idx) dengan proporsi kelas terjaga.

    Penting karena kelas positif cuma ~33%. Caranya:
      1. Tiap kelas diacak lalu dipotong jadi k bagian.
      2. Fold ke-i: bagian ke-i tiap kelas jadi validasi, sisanya train.
    """
    rng = np.random.default_rng(seed)

    # potong tiap kelas jadi k bagian
    chunks_per_class = []
    for kelas in np.unique(y):
        idx = np.where(y == kelas)[0]
        rng.shuffle(idx)
        chunks_per_class.append(np.array_split(idx, k))

    # susun train dan val untuk tiap fold
    for i in range(k):
        val_parts = []
        train_parts = []
        for chunks in chunks_per_class:
            for j, chunk in enumerate(chunks):
                if j == i:
                    val_parts.append(chunk)
                else:
                    train_parts.append(chunk)

        train_idx = np.sort(np.concatenate(train_parts))
        val_idx = np.sort(np.concatenate(val_parts))
        yield train_idx, val_idx
