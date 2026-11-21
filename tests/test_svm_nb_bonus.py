import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.bonus.naive_bayes_graph import feature_strengths, plot_naive_bayes_graph
from src.bonus.svm_plot import fit_kernels, plot_svm_kernels
from src.models.naive_bayes import NaiveBayes


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    return X, y


def test_svm_plot_has_one_panel_per_kernel(tmp_path, data):
    X, y = data
    fig = plot_svm_kernels(X[:, :2], y, path=tmp_path / "svm.png")
    assert len(fig.axes) == 3
    assert (tmp_path / "svm.png").exists()


def test_svm_plot_rejects_non_2d(data):
    X, y = data
    with pytest.raises(ValueError):
        fit_kernels(X, y)


def test_nb_strength_ranks_informative_feature_first(tmp_path, data):
    X, y = data
    model = NaiveBayes().fit(X, y)
    strengths = feature_strengths(model, ["a", "b", "c"])
    assert max(strengths, key=strengths.get) == "a"
    fig = plot_naive_bayes_graph(model, ["a", "b", "c"], top_n=3, path=tmp_path / "nb.png")
    assert (tmp_path / "nb.png").exists() and fig is not None
