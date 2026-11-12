"""Dari data mentah jadi matriks angka.

Kata spesifikasi, train dan test diproses terpisah. Semua statistik (median,
mean, std, daftar kategori) dipelajari dari train di fit(), lalu dipakai apa
adanya buat test di transform(). Ini yang mencegah leakage.

Ada dua bentuk data ("view"):
  matrix : buat KNN, LogReg, SVM. Tanpa NaN, skalanya seragam.
  tree   : buat Decision Tree dan Naive Bayes. NaN dibiarkan, kategori jadi kode integer.

Tiga langkah dari spesifikasi, semuanya dipelajari dari train dan default-nya mati:
  imbalance : balanced_weights(y) dipakai sebagai sample_weight
  seleksi   : select_k = ambil k kolom dengan mutual information tertinggi
  PCA       : pca_var = ambil komponen sampai variansi yang diminta tercapai

Catatan: 39% baris punya nilai kosong, jadi barisnya jangan dibuang.
"""
import numpy as np
import pandas as pd

from src.config import (CATEGORICAL, ENGINEERED, ID_COL, NUMERIC, SKEWED,
                        TARGET)


def load_data(data_dir: str = "data"):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Buang baris duplikat (id tidak dihitung). Cuma buat train.

    Baris test jangan dibuang, soalnya tiap id harus punya prediksi.
    """
    subset = [c for c in df.columns if c != ID_COL]
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def split_xy(df: pd.DataFrame):
    """Pisahkan fitur dan target. Kolom id dibuang karena nggak punya sinyal."""
    y = None
    if TARGET in df.columns:
        y = df[TARGET].to_numpy(dtype=int)

    drop_cols = [c for c in (ID_COL, TARGET) if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


def balanced_weights(y) -> np.ndarray:
    """Bobot tiap sampel biar total bobot kedua kelas sama: n / (2 * n_kelas).

    Dipakai sebagai sample_weight di fit(). Barisnya nggak dibuang atau
    digandakan, jadi aman dari leakage.
    """
    y = np.asarray(y)
    weight_of = {}
    for kelas in (0, 1):
        weight_of[kelas] = len(y) / (2.0 * (y == kelas).sum())
    return np.array([weight_of[v] for v in y], dtype=float)


def mutual_info(X: np.ndarray, y, bins: int = 10) -> np.ndarray:
    """Mutual information tiap kolom X terhadap y (biner).

    Tiap kolom dibagi ke `bins` kuantil dulu. Kolom 0/1 otomatis jadi 2 bin.
    """
    y = np.asarray(y)
    n = len(y)
    p_y = np.array([(y == 0).mean(), (y == 1).mean()])
    out = np.zeros(X.shape[1])

    for j in range(X.shape[1]):
        col = X[:, j]
        edges = np.unique(np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1]))
        bin_of = np.searchsorted(edges, col, side="right")

        mi = 0.0
        for b in np.unique(bin_of):
            in_bin = bin_of == b
            p_x = in_bin.sum() / n
            for kelas in (0, 1):
                p_xy = (in_bin & (y == kelas)).sum() / n
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x * p_y[kelas]))
        out[j] = mi
    return out


class Preprocessor:
    def __init__(self, mode: str = "matrix", missing_flags: bool = True,
                 select_k=None, pca_var=None):
        assert mode in ("matrix", "tree")
        self.mode = mode
        self.missing_flags = missing_flags  # matrix: tandai sel yang diisi imputasi
        self.select_k = select_k  # matrix: jumlah kolom terbaik (fit() butuh y)
        self.pca_var = pca_var    # matrix: target variansi, mis. 0.95

    def fit(self, X: pd.DataFrame, y=None) -> "Preprocessor":
        X = self._engineer(X)
        self.num_cols_ = NUMERIC + ENGINEERED
        self.cat_levels_ = {}
        for c in CATEGORICAL:
            self.cat_levels_[c] = sorted(X[c].dropna().unique())

        if self.mode == "tree":
            self.feature_names_ = self.num_cols_ + CATEGORICAL
            # DecisionTree dan NaiveBayes perlu tahu kolom mana yang kategorikal
            first_cat = len(self.num_cols_)
            self.categorical_idx_ = list(range(first_cat, len(self.feature_names_)))
            return self

        # matrix view: semua statistik dari train
        self.medians_ = X[self.num_cols_].median()

        self.flag_cols_ = []
        if self.missing_flags:
            for c in self.num_cols_ + CATEGORICAL:
                if X[c].isna().any():
                    self.flag_cols_.append(c)

        num = self._numeric_matrix(X)
        self.mean_ = num.mean(axis=0)
        self.std_ = num.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # kolom konstan, biar nggak bagi nol

        onehot_names = []
        for c in CATEGORICAL:
            for level in self.cat_levels_[c]:
                onehot_names.append(f"{c}={level}")
        flag_names = [f"{c}_missing" for c in self.flag_cols_]
        self.feature_names_ = self.num_cols_ + onehot_names + flag_names
        self.categorical_idx_ = []

        # seleksi fitur lalu PCA (opsional, dari train aja)
        Z = self._matrix(X)
        self.keep_ = np.arange(Z.shape[1])

        if self.select_k is not None:
            assert y is not None, "select_k butuh y di fit()"
            mi = mutual_info(Z, y)
            best = np.argsort(-mi)[: self.select_k]
            self.keep_ = np.sort(best)
            self.feature_names_ = [self.feature_names_[i] for i in self.keep_]
            Z = Z[:, self.keep_]

        if self.pca_var is not None:
            self.pca_mean_ = Z.mean(axis=0)
            _, S, Vt = np.linalg.svd(Z - self.pca_mean_, full_matrices=False)
            ratio = np.cumsum(S ** 2) / np.sum(S ** 2)
            m = int(np.searchsorted(ratio, self.pca_var) + 1)
            self.components_ = Vt[:m]
            self.explained_ratio_ = (S[:m] ** 2) / np.sum(S ** 2)
            self.feature_names_ = [f"pc{i + 1}" for i in range(m)]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert hasattr(self, "num_cols_"), "panggil fit() dulu"
        X = self._engineer(X)

        if self.mode == "tree":
            num = X[self.num_cols_].to_numpy(dtype=float)  # NaN dipertahankan
            cat_columns = []
            for c in CATEGORICAL:
                code_of = {level: i for i, level in enumerate(self.cat_levels_[c])}
                # kategori kosong atau nggak dikenal jadi NaN
                cat_columns.append(X[c].map(code_of).to_numpy(dtype=float))
            return np.hstack([num, np.column_stack(cat_columns)])

        Z = self._matrix(X)[:, self.keep_]
        if self.pca_var is not None:
            Z = (Z - self.pca_mean_) @ self.components_.T
        return Z

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @staticmethod
    def _engineer(X: pd.DataFrame) -> pd.DataFrame:
        """Fitur turunan yang simpel. Nggak butuh statistik dari data, jadi aman dari leakage."""
        X = X.copy()
        X["total_nights"] = X["weekend_nights"] + X["weekday_nights"]
        X["total_guests"] = X["num_adults"] + X["num_children"]  # NaN kalau salah satunya NaN
        # tarif 0 hampir nggak pernah batal (0,7% di train), jadi dijadikan fitur sendiri
        X["is_free"] = (X["nightly_rate_idr"] == 0).astype(float)
        return X

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Matrix view penuh sebelum seleksi dan PCA. Dipakai fit dan transform."""
        if "total_nights" not in X.columns:
            X = self._engineer(X)

        num = (self._numeric_matrix(X) - self.mean_) / self.std_

        onehot_blocks = []
        for c in CATEGORICAL:
            cols = []
            for level in self.cat_levels_[c]:
                cols.append((X[c] == level).to_numpy(dtype=float))
            onehot_blocks.append(np.column_stack(cols))  # kosong atau nggak dikenal: semua nol

        flag_blocks = []
        for c in self.flag_cols_:
            flag_blocks.append(X[c].isna().to_numpy(dtype=float)[:, None])

        return np.hstack([num] + onehot_blocks + flag_blocks)

    def _numeric_matrix(self, X: pd.DataFrame) -> np.ndarray:
        num = X[self.num_cols_].fillna(self.medians_)  # median dari train
        num[SKEWED] = np.log1p(num[SKEWED])
        return num.to_numpy(dtype=float)
