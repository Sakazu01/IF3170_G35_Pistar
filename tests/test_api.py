import numpy as np
import pytest

from src.models.decision_tree import DecisionTree
from src.models.knn import KNN
from src.models.logistic_regression import LogisticRegressionScratch
from src.models.naive_bayes import NaiveBayes
from src.models.svm import SVMScratch

MODELS = [KNN, DecisionTree, LogisticRegressionScratch, SVMScratch, NaiveBayes]


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("Model", MODELS)
def test_fit_returns_self_and_sets_classes(Model, data):
    X, y = data
    model = Model()
    assert model.fit(X, y) is model
    np.testing.assert_array_equal(model.classes_, [0, 1])


@pytest.mark.parametrize("Model", MODELS)
def test_predict_proba_has_two_columns_summing_to_one(Model, data):
    X, y = data
    proba = Model().fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("Model", MODELS)
def test_predict_follows_positive_probability(Model, data):
    X, y = data
    model = Model().fit(X, y)
    expected = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    np.testing.assert_array_equal(model.predict(X), expected)
    assert model.score(X, y) == pytest.approx(np.mean(model.predict(X) == y))


@pytest.mark.parametrize("Model", MODELS)
def test_params_roundtrip(Model):
    model = Model()
    clone = Model(**model.get_params())
    assert clone.get_params() == model.get_params()


@pytest.mark.parametrize("Model", MODELS)
def test_set_params_changes_value_and_rejects_unknown(Model):
    model = Model()
    name = next(iter(model.get_params()))
    value = model.get_params()[name]
    assert model.set_params(**{name: value}) is model
    with pytest.raises(ValueError):
        model.set_params(not_a_param=1)


@pytest.mark.parametrize("Model", MODELS)
def test_save_and_load_keep_predictions(Model, data, tmp_path):
    X, y = data
    model = Model().fit(X, y)
    path = model.save(tmp_path / "model.pkl")
    loaded = Model.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(X), model.predict_proba(X))
