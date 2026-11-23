import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer

NUMERIC = [
    'num_adults', 'num_children', 'weekend_nights', 'weekday_nights',
    'needs_parking', 'days_before_arrival', 'arrival_year', 'arrival_month',
    'arrival_day', 'is_returning_guest', 'prior_cancellations',
    'prior_completed_bookings', 'nightly_rate_idr', 'special_requests',
]
CATEGORICAL = ['meal_package', 'room_category', 'booking_channel']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'is_canceled'
SKEWED = ['days_before_arrival', 'nightly_rate_idr', 'prior_cancellations', 'prior_completed_bookings']
FEATURE_GROUPS = ('basic', 'stay', 'history', 'season', 'price', 'calendar', 'interactions')


def clean_features(frame):

    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing booking feature columns: {sorted(missing)}')
    data = frame[FEATURES].copy()
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    nonnegative = [c for c in NUMERIC if c not in ('arrival_year', 'arrival_month', 'arrival_day')]
    for col in nonnegative:
        data.loc[data[col] < 0, col] = np.nan
    for col in ('needs_parking', 'is_returning_guest'):
        data.loc[~data[col].isin([0, 1]), col] = np.nan
    dates = pd.to_datetime(dict(year=data.arrival_year, month=data.arrival_month,
                               day=data.arrival_day), errors='coerce')
    data.loc[dates.isna(), ['arrival_year', 'arrival_month', 'arrival_day']] = np.nan
    for col in CATEGORICAL:
        data[col] = data[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        data[col] = data[col].replace('', np.nan).where(data[col].notna(), np.nan).astype(object)
    return data


def add_features(frame, groups=('basic',)):
    data = clean_features(frame)
    if set(groups) - set(FEATURE_GROUPS):
        raise ValueError(f'Unknown feature group: {set(groups) - set(FEATURE_GROUPS)}')
    nights = data.weekend_nights + data.weekday_nights
    guests = data.num_adults + data.num_children
    if 'basic' in groups:
        data['total_nights'], data['total_guests'] = nights, guests
        data['is_free'] = np.where(data.nightly_rate_idr.isna(), np.nan,
                                   (data.nightly_rate_idr == 0).astype(float))
    if 'stay' in groups:
        data['stay_weekend_ratio'] = data.weekend_nights / nights.clip(lower=1)
        data['lead_time_per_night'] = data.days_before_arrival / nights.clip(lower=1)
        data['price_per_guest'] = data.nightly_rate_idr / guests.clip(lower=1)
        data['lead_time_per_guest'] = data.days_before_arrival / guests.clip(lower=1)
        data['has_children'] = data.num_children.gt(0).astype(float).where(data.num_children.notna())
    if 'history' in groups:
        prior = data.prior_cancellations + data.prior_completed_bookings
        data['prior_booking_total'] = prior
        data['prior_cancellation_ratio'] = data.prior_cancellations / prior.clip(lower=1)
    if 'season' in groups:
        angle = 2 * np.pi * (data.arrival_month - 1) / 12
        data['arrival_month_sin'], data['arrival_month_cos'] = np.sin(angle), np.cos(angle)
    if 'price' in groups:
        data['price_per_guest'] = data.nightly_rate_idr / guests.clip(lower=1)
        data['estimated_booking_value'] = data.nightly_rate_idr * nights
        data['has_special_requests'] = data.special_requests.gt(0).astype(float).where(data.special_requests.notna())
    if 'calendar' in groups:
        dates = pd.to_datetime(dict(year=data.arrival_year, month=data.arrival_month, day=data.arrival_day), errors='coerce')
        data['arrival_weekday'] = dates.dt.dayofweek.astype(float)
        data['arrival_day_of_year'] = dates.dt.dayofyear.astype(float)
        data['arrival_day_sin'] = np.sin(2 * np.pi * (data.arrival_day - 1) / 31)
        data['arrival_day_cos'] = np.cos(2 * np.pi * (data.arrival_day - 1) / 31)
    if 'interactions' in groups:
        for name in ('special_requests', 'num_adults', 'nightly_rate_idr'):
            data['lead_x_' + name] = np.log1p(data.days_before_arrival) * data[name]
        for channel in ('Online Travel Agent', 'Offline Agent', 'Corporate', 'Airline', 'Complimentary'):
            data['lead_channel_' + channel] = data.days_before_arrival * data.booking_channel.eq(channel)
    return data


def feature_groups(frame):

    return pd.util.hash_pandas_object(clean_features(frame), index=False).to_numpy()


def log_nonnegative(values):
    return np.log1p(np.maximum(values, 0))


def mutual_information(X, y):
    return mutual_info_classif(X, y, random_state=42)


class BookingPreprocessor(TransformerMixin, BaseEstimator):







    def __init__(self, view='matrix', feature_groups=('basic',), select_k=None):
        self.view = view
        self.feature_groups = feature_groups
        self.select_k = select_k

    def fit(self, X, y=None):
        if self.view not in ('matrix', 'tree', 'tree_imputed', 'bayes'):
            raise ValueError('view must be matrix, tree, tree_imputed, or bayes')
        raw = clean_features(X)
        observed = raw.days_before_arrival.dropna()
        self.lead_bin_edges_ = np.unique(observed.quantile([.2, .4, .6, .8]).to_numpy())
        data = self._features(X)
        self.numeric_ = [c for c in data.columns if c not in CATEGORICAL]
        if self.view == 'matrix':
            skewed = [c for c in self.numeric_ if c in SKEWED or c in ('price_per_guest', 'prior_booking_total')]
            plain = [c for c in self.numeric_ if c not in skewed]
            numeric = lambda log: Pipeline([
                ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                ('log', FunctionTransformer(log_nonnegative, feature_names_out='one-to-one') if log else 'passthrough'),
                ('scale', StandardScaler()),
            ])
            categorical = Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__', keep_empty_features=True)),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ])
            steps = [('skewed', numeric(True), skewed), ('plain', numeric(False), plain),
                     ('flags', MissingIndicator(features='all'), self.numeric_),
                     ('cat', categorical, CATEGORICAL)]
        else:
            ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,
                                     encoded_missing_value=np.nan)
            numeric_step = (SimpleImputer(strategy='median', keep_empty_features=True)
                            if self.view == 'tree_imputed' else
                            'passthrough' if self.view == 'tree' else StandardScaler())
            steps = [('num', numeric_step, self.numeric_)]
            if self.view == 'tree_imputed':
                steps.append(('flags', MissingIndicator(features='all'), self.numeric_))
            steps.append(('cat', ordinal, CATEGORICAL))
        self.columns_ = ColumnTransformer(steps)
        values = self.columns_.fit_transform(data, y)
        self.selector_ = None
        if self.view == 'matrix':
            self.variance_ = VarianceThreshold().fit(values)
            values = self.variance_.transform(values)
            if self.select_k is not None:
                if not isinstance(self.select_k, int) or self.select_k <= 0:
                    raise ValueError('select_k must be a positive integer or None')
                self.selector_ = SelectKBest(mutual_information, k=min(self.select_k, values.shape[1])).fit(values, y)
        elif self.select_k is not None:
            raise ValueError('MI selection is supported only for the matrix view')
        n_numeric = len(self.numeric_) * (2 if self.view == 'tree_imputed' else 1)
        self.categorical_idx_ = [] if self.view == 'matrix' else list(range(n_numeric, n_numeric + len(CATEGORICAL)))
        self.feature_names_ = list(self.get_feature_names_out()) if self.view == 'matrix' else (
            self.numeric_ + ([name + '_missing' for name in self.numeric_] if self.view == 'tree_imputed' else []) + CATEGORICAL)
        self.n_fit_rows_ = len(X)
        return self

    def transform(self, X):
        if not hasattr(self, 'columns_'):
            raise RuntimeError('Call fit before transform')
        values = self.columns_.transform(self._features(X))
        if self.view == 'matrix':
            values = self.variance_.transform(values)
            if self.selector_ is not None:
                values = self.selector_.transform(values)
        return np.asarray(values, dtype=float)

    def _features(self, X):
        data = add_features(X, self.feature_groups)
        if 'interactions' in self.feature_groups:
            lead = data.days_before_arrival
            data['lead_time_bin'] = np.searchsorted(self.lead_bin_edges_, lead).astype(float)
            data.loc[lead.isna(), 'lead_time_bin'] = np.nan
        return data

    def get_feature_names_out(self, input_features=None):

        if not hasattr(self, 'columns_'):
            raise RuntimeError('Call fit before requesting feature names')
        names = self.columns_.get_feature_names_out()
        if self.view == 'matrix':
            names = names[self.variance_.get_support()]
            if self.selector_ is not None:
                names = names[self.selector_.get_support()]
        return names
