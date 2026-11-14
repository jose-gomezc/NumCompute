"""Unit tests for Pipeline.partial_fit and the streaming helpers it relies on."""

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def _stream(n=1000, seed=8):
    """Numeric data whose features drift over time, with a label from two of them."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) + np.linspace(0, 2, n)[:, None]
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    return X, y


def test_partial_fit_equals_chaining_the_steps_by_hand():
    X, y = _stream()
    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(buffer_size=300))])
    scaler, tree = StandardScaler(), DecisionTreeClassifier(buffer_size=300)

    for start in range(0, 1000, 100):
        Xc, yc = X[start:start + 100], y[start:start + 100]
        pipe.partial_fit(Xc, yc)
        scaler.partial_fit(Xc)
        tree.partial_fit(scaler.transform(Xc), yc)

    np.testing.assert_array_equal(pipe.steps[0][1].mean_, scaler.mean_)
    assert pipe.steps[1][1].n_retrains_ == tree.n_retrains_ == 3
    np.testing.assert_array_equal(pipe.predict(X), tree.predict(scaler.transform(X)))


def test_spec_pipeline_streams_predicts_and_flushes():
    X, y = _stream()
    X[::7, 2] = np.nan                                       # missing values for the imputer
    pipe = Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
    ])

    for start in range(0, 400, 100):
        pipe.partial_fit(X[start:start + 100], y[start:start + 100])
    with pytest.raises(RuntimeError, match="not fitted"):    # 400 < 500 rows: no tree yet
        pipe.predict(X[:5])

    for start in range(400, 700, 100):
        pipe.partial_fit(X[start:start + 100], y[start:start + 100])
    model = pipe.steps[-1][1]
    assert model.n_retrains_ == 1 and model.n_buffered_ == 200
    assert pipe.predict(X[:10]).shape == (10,)
    proba = pipe.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    pipe.flush()
    assert model.n_retrains_ == 2 and model.n_buffered_ == 0


def test_new_categories_widen_the_tree_input():
    first, y_first = [["a", "x"], ["b", "y"], ["a", "y"], ["b", "x"]], [0, 1, 0, 1]
    pipe = Pipeline([("onehot", OneHotEncoder()), ("model", DecisionTreeClassifier(buffer_size=6))])
    model = pipe.steps[-1][1]

    pipe.partial_fit(first, y_first)                          # 4 columns, 4 rows buffered
    pipe.partial_fit([["c", "x"]], [1])                       # new category mid-buffer: 5 columns
    pipe.partial_fit([["a", "x"]], [0])                       # 6 rows: retrain
    assert model.n_retrains_ == 1 and model.n_features_in_ == 5

    # Buffered rows were padded with zeros, which is what the encoder gives them now
    rows = first + [["c", "x"], ["a", "x"]]
    expected = DecisionTreeClassifier().fit(pipe.steps[0][1].transform(rows), y_first + [1, 0])
    assert model.export_text() == expected.export_text()

    pipe.partial_fit([["a", "z"]], [0])                       # new category after the retrain
    assert model.n_features_in_ == 6
    assert pipe.predict([["c", "z"], ["b", "y"]]).shape == (2,)

    with pytest.raises(ValueError, match="earlier data"):
        DecisionTreeClassifier().add_features(1)
    with pytest.raises(ValueError, match="n_new"):
        model.add_features(0)
    with pytest.raises(TypeError, match="n_new"):
        model.add_features(1.5)


class _BatchOnlyModel:
    """A model with fit and predict but no partial_fit."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_streaming_errors_and_empty_chunks():
    batch_only = Pipeline([("scale", StandardScaler()), ("model", _BatchOnlyModel())])
    with pytest.raises(ValueError, match="'model' does not provide partial_fit"):
        batch_only.partial_fit([[1.0], [2.0]], [0, 1])

    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier())])
    pipe.partial_fit(np.empty((0, 3)), [])                    # empty chunk: ignored
    assert not pipe.is_fitted_
    with pytest.raises(ValueError, match="partial_fit before predict_proba"):
        pipe.predict_proba([[1.0, 2.0, 3.0]])

    scale_only = Pipeline([("scale", StandardScaler())]).fit([[1.0], [2.0]])
    with pytest.raises(ValueError, match="predict_proba"):
        scale_only.predict_proba([[1.0]])
    with pytest.raises(ValueError, match="flush"):
        scale_only.flush()
