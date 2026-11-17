from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATEGORICAL, ENGINEERED, ID_COL, NUMERIC, SKEWED, TARGET


def load_data(data_dir="data"):
    root = Path(data_dir)
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required data file(s): " + ", ".join(missing))
    return pd.read_csv(train_path), pd.read_csv(test_path)

def _required_columns(include_target=True):
    columns = [ID_COL, *NUMERIC, *CATEGORICAL]
    if include_target:
        columns.append(TARGET)
    return columns

def validate_schema(train, test=None, sample_submission=None):
    if not isinstance(train, pd.DataFrame):
        raise TypeError("train must be a pandas DataFrame")
    required_train = set(_required_columns(include_target=True))
    missing_train = sorted(required_train - set(train.columns))
    if missing_train:
        raise ValueError(f"Train is missing required column(s): {missing_train}")

    if test is not None:
        required_test = set(_required_columns(include_target=False))
        missing_test = sorted(required_test - set(test.columns))
        if missing_test:
            raise ValueError(f"Test is missing required column(s): {missing_test}")
        train_features = set(train.columns) - {TARGET}
        if train_features != set(test.columns):
            raise ValueError("Train features and test columns do not match")

    if train[ID_COL].duplicated().any():
        raise ValueError("ID column contains duplicates in train")
    target_values = set(train[TARGET].dropna().unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1, found {sorted(target_values)}")

    if sample_submission is not None:
        expected = {ID_COL, TARGET}
        if set(sample_submission.columns) != expected:
            raise ValueError(
                f"Submission columns must be {sorted(expected)}, "
                f"found {list(sample_submission.columns)}"
            )

    return {
        "train_shape": tuple(train.shape),
        "test_shape": tuple(test.shape) if test is not None else None,
        "target_counts": train[TARGET].value_counts(dropna=False).to_dict(),
        "categorical": list(CATEGORICAL),
        "numeric": list(NUMERIC),
    }


def _invalid_masks(df):
    masks = {}
    nonnegative = [
        "num_adults",
        "num_children",
        "weekend_nights",
        "weekday_nights",
        "days_before_arrival",
        "prior_cancellations",
        "prior_completed_bookings",
        "nightly_rate_idr",
        "special_requests",
    ]
    for column in nonnegative:
        if column in df:
            masks[column] = df[column].notna() & (df[column] < 0)

    for column in ("needs_parking", "is_returning_guest"):
        if column in df:
            masks[column] = df[column].notna() & ~df[column].isin([0, 1])

    if {"arrival_year", "arrival_month", "arrival_day"}.issubset(df.columns):
        dates = pd.to_datetime(
            {
                "year": df["arrival_year"],
                "month": df["arrival_month"],
                "day": df["arrival_day"],
            },
            errors="coerce",
        )
        masks["arrival_date"] = dates.isna()

    if TARGET in df:
        masks[TARGET] = df[TARGET].notna() & ~df[TARGET].isin([0, 1])
    return masks


def data_quality_report(df, is_train=True):
    masks = _invalid_masks(df)
    invalid_by_rule = {name: int(mask.sum()) for name, mask in masks.items()}
    invalid_rows = int(pd.concat(list(masks.values()), axis=1).any(axis=1).sum()) if masks else 0
    duplicate_subset = [column for column in df.columns if column != ID_COL]
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
        "duplicate_rows_excluding_id": int(df.duplicated(subset=duplicate_subset).sum()),
        "duplicate_ids": int(df[ID_COL].duplicated().sum()) if ID_COL in df else None,
        "invalid_by_rule": invalid_by_rule,
        "invalid_rows": invalid_rows,
        "is_train": bool(is_train),
    }


def clean_train(df, return_report=False):
    before = data_quality_report(df, is_train=True)
    clean = df.copy()

    for column in CATEGORICAL:
        if column in clean:
            clean[column] = clean[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    masks = _invalid_masks(clean)
    for column, mask in masks.items():
        if column == "arrival_date":
            continue
        if column in clean:
            clean.loc[mask, column] = np.nan

    if {"arrival_year", "arrival_month", "arrival_day"}.issubset(clean.columns):
        dates = pd.to_datetime(
            {
                "year": clean["arrival_year"],
                "month": clean["arrival_month"],
                "day": clean["arrival_day"],
            },
            errors="coerce",
        )
        invalid_dates = dates.isna()
        clean.loc[invalid_dates, ["arrival_year", "arrival_month", "arrival_day"]] = np.nan

    duplicate_subset = [column for column in clean.columns if column != ID_COL]
    clean = clean.drop_duplicates(subset=duplicate_subset, keep="first").reset_index(drop=True)

    after = data_quality_report(clean, is_train=True)
    report = {"before": before, "after": after, "removed_rows": len(df) - len(clean)}
    return (clean, report) if return_report else clean


def split_xy(df):
    y = df[TARGET].to_numpy(dtype=int) if TARGET in df.columns else None
    drop = [column for column in (ID_COL, TARGET) if column in df.columns]
    return df.drop(columns=drop), y


def balanced_weights(y):
    y = np.asarray(y, dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    if not set(classes).issubset({0, 1}) or len(classes) < 2:
        raise ValueError("balanced_weights requires both binary classes 0 and 1")
    weights = {int(cls): len(y) / (len(classes) * count) for cls, count in zip(classes, counts)}
    return np.asarray([weights[int(value)] for value in y], dtype=float)


def mutual_info(X, y, bins=10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D with one target value per row")
    if not np.isfinite(X).all():
        raise ValueError("mutual_info requires finite X")

    n = len(y)
    py = {cls: float(np.mean(y == cls)) for cls in (0, 1)}
    result = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        column = X[:, j]
        edges = np.unique(np.quantile(column, np.linspace(0, 1, bins + 1)[1:-1]))
        values = np.searchsorted(edges, column, side="right")
        for value in np.unique(values):
            mask = values == value
            px = float(mask.mean())
            for cls in (0, 1):
                pxy = float(np.mean(mask & (y == cls)))
                if pxy > 0 and px > 0 and py[cls] > 0:
                    result[j] += pxy * np.log(pxy / (px * py[cls]))
    return result


class Preprocessor:
    def __init__(self, mode="matrix", missing_flags=True, select_k=None, pca_var=None):
        if mode not in ("matrix", "tree"):
            raise ValueError("mode must be 'matrix' or 'tree'")
        if select_k is not None and (int(select_k) <= 0):
            raise ValueError("select_k must be positive")
        if pca_var is not None and not 0 < float(pca_var) <= 1:
            raise ValueError("pca_var must be in (0, 1]")
        self.mode = mode
        self.missing_flags = bool(missing_flags)
        self.select_k = int(select_k) if select_k is not None else None
        self.pca_var = float(pca_var) if pca_var is not None else None

    def fit(self, X, y=None):
        X = self._prepare_frame(X)
        self.num_cols_ = [column for column in (*NUMERIC, *ENGINEERED) if column in X]
        missing_columns = set(_required_columns(include_target=False)) - set(X.columns) - {ID_COL}
        if missing_columns:
            raise ValueError(f"Input is missing feature column(s): {sorted(missing_columns)}")
        self.cat_levels_ = {
            column: sorted(X[column].dropna().unique().tolist(), key=str)
            for column in CATEGORICAL
        }

        if self.mode == "tree":
            self.feature_names_ = [*self.num_cols_, *CATEGORICAL]
            self.categorical_idx_ = list(range(len(self.num_cols_), len(self.feature_names_)))
            return self

        self.medians_ = X[self.num_cols_].median()
        flag_candidates = [*self.num_cols_, *CATEGORICAL]
        self.flag_cols_ = (
            [column for column in flag_candidates if X[column].isna().any()]
            if self.missing_flags
            else []
        )
        numeric = self._numeric_matrix(X)
        self.mean_ = numeric.mean(axis=0)
        self.std_ = numeric.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self.cat_feature_names_ = [
            f"{column}={level}"
            for column in CATEGORICAL
            for level in self.cat_levels_[column]
        ]
        self.full_feature_names_ = [
            *self.num_cols_,
            *self.cat_feature_names_,
            *[f"{column}_missing" for column in self.flag_cols_],
        ]
        self.categorical_idx_ = []

        full = self._matrix(X)
        self.keep_ = np.arange(full.shape[1], dtype=int)
        if self.select_k is not None:
            if y is None:
                raise ValueError("select_k requires y during fit")
            if self.select_k > full.shape[1]:
                raise ValueError(
                    f"select_k={self.select_k} exceeds {full.shape[1]} available features"
                )
            scores = mutual_info(full, y)
            self.mi_scores_ = scores
            order = np.lexsort((np.arange(len(scores)), -scores))
            self.keep_ = np.sort(order[: self.select_k])

        selected = full[:, self.keep_]
        self.pca_mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        if self.pca_var is not None:
            self.pca_mean_ = selected.mean(axis=0)
            centered = selected - self.pca_mean_
            _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
            variance = singular_values**2
            total = float(variance.sum())
            if total == 0:
                count = 1
                ratios = np.array([1.0])
            else:
                ratios = np.cumsum(variance) / total
                count = int(np.searchsorted(ratios, self.pca_var, side="left") + 1)
            self.components_ = vt[:count]
            self.explained_variance_ratio_ = (
                variance[:count] / total if total else np.array([1.0])
            )
            self.feature_names_ = [f"pc{i + 1}" for i in range(count)]
        else:
            self.feature_names_ = [self.full_feature_names_[i] for i in self.keep_]
        return self

    def transform(self, X):
        if not hasattr(self, "num_cols_"):
            raise RuntimeError("Call fit before transform")
        X = self._prepare_frame(X)
        if self.mode == "tree":
            numeric = X[self.num_cols_].to_numpy(dtype=float)
            categorical = []
            for column in CATEGORICAL:
                mapping = {level: index for index, level in enumerate(self.cat_levels_[column])}
                categorical.append(X[column].map(mapping).to_numpy(dtype=float))
            return np.column_stack([numeric, *categorical]).astype(float, copy=False)

        selected = self._matrix(X)[:, self.keep_]
        if self.components_ is not None:
            selected = (selected - self.pca_mean_) @ self.components_.T
        if not np.isfinite(selected).all():
            raise ValueError("Matrix transform produced non-finite values")
        return selected.astype(float, copy=False)

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    @staticmethod
    def _prepare_frame(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        frame = X.copy()
        if ID_COL in frame.columns:
            frame = frame.drop(columns=[ID_COL])
        if TARGET in frame.columns:
            frame = frame.drop(columns=[TARGET])
        missing = set(NUMERIC + CATEGORICAL) - set(frame.columns)
        if missing:
            raise ValueError(f"Input is missing feature column(s): {sorted(missing)}")
        for column in CATEGORICAL:
            frame[column] = frame[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
        frame["total_nights"] = frame["weekend_nights"] + frame["weekday_nights"]
        frame["total_guests"] = frame["num_adults"] + frame["num_children"]
        frame["is_free"] = (frame["nightly_rate_idr"] == 0).astype(float)
        return frame

    def _numeric_matrix(self, X):
        numeric = X[self.num_cols_].copy()
        numeric = numeric.fillna(self.medians_)
        for column in SKEWED:
            if column in numeric:
                numeric[column] = np.log1p(numeric[column].clip(lower=0))
        return numeric.to_numpy(dtype=float)

    def _matrix(self, X):
        numeric = (self._numeric_matrix(X) - self.mean_) / self.std_
        one_hot = [
            np.column_stack(
                [(X[column] == level).to_numpy(dtype=float) for level in self.cat_levels_[column]]
            )
            for column in CATEGORICAL
        ]
        flags = [X[column].isna().to_numpy(dtype=float)[:, None] for column in self.flag_cols_]
        parts = [numeric, *one_hot, *flags]
        return np.hstack(parts).astype(float, copy=False)

