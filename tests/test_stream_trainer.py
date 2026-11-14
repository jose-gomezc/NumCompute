"""Unit tests for :mod:`numcompute_stream.stream`: StreamTrainer and the memory it reports."""

import numpy as np
import pytest

from numcompute_stream.metrics import StreamingAccuracy, StreamingPrecision, accuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def _data(n=1000, seed=9):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


def _chunks(X, y, size=100):
    for start in range(0, len(y), size):
        yield X[start:start + size], y[start:start + size]


def test_process_chunk_scores_before_training():
    X, y = _data()
    tree = DecisionTreeClassifier(buffer_size=300)
    trainer = StreamTrainer(tree)
    scored_true, scored_pred = [], []

    for Xc, yc in _chunks(X, y):
        expected = tree.predict(Xc) if hasattr(tree, "classes_") else None   # model before this chunk
        record = trainer.process_chunk(Xc, yc)

        if expected is None:
            assert np.isnan(record["accuracy"]) and np.isnan(record["cumulative_accuracy"])
        else:
            assert record["accuracy"] == accuracy(yc, expected)
            scored_true.append(yc)
            scored_pred.append(expected)

    assert np.isnan(trainer.history_["accuracy"][:3]).all()         # no tree before 300 rows
    assert trainer.history_["cumulative_accuracy"][-1] == pytest.approx(
        accuracy(np.concatenate(scored_true), np.concatenate(scored_pred))
    )


def test_score_chunk_and_fit_chunk_are_separate():
    X, y = _data()
    tree = DecisionTreeClassifier(buffer_size=200)
    trainer = StreamTrainer(tree)

    first = trainer.process_chunk(X[:100], y[:100])
    second = trainer.process_chunk(X[100:200], y[100:200])          # 200 rows: retrain
    assert not first["retrained"] and np.isnan(first["retrain_time"])
    assert second["retrained"] and second["retrain_time"] == tree.fit_time_ and second["n_retrains"] == 1

    record = trainer.score_chunk(X[200:300], y[200:300])            # scores only
    assert not np.isnan(record["accuracy"])
    assert tree.n_buffered_ == 0 and len(trainer.history_["chunk"]) == 2

    assert trainer.fit_chunk(X[200:300], y[200:300]) is trainer     # trains only
    assert tree.n_buffered_ == 100


def test_run_logs_every_chunk_rolling_accuracy_and_memory():
    X, y = _data()
    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(buffer_size=250))])
    history = StreamTrainer(pipe, window=3).run(_chunks(X, y))      # a generator

    assert all(len(values) == 10 for values in history.values())
    assert history["rolling_accuracy"][-1] == pytest.approx(np.mean(history["accuracy"][-3:]))

    # Buffer fills, empties at each retrain (chunks 2, 5, 8); the tree exists after the first one
    buffer = history["buffer_bytes"]
    assert buffer[0] < buffer[1] and buffer[2] == 0 and history["retrained"][2]
    assert history["tree_bytes"][0] == 0 and history["tree_bytes"][-1] > 0

    model = pipe.steps[-1][1]
    assert model.n_buffered_ == 0 and model.n_retrains_ == 4         # run flushed the last 100 rows


def test_auc_and_extra_metrics_depend_on_the_labels():
    X, y = _data()
    binary = StreamTrainer(DecisionTreeClassifier(buffer_size=200),
                           metrics={"precision": StreamingPrecision()}).run(_chunks(X, y))
    assert 0.5 < binary["cumulative_auc"][-1] <= 1.0
    assert 0.5 < binary["precision"][-1] <= 1.0

    y3 = np.digitize(X[:, 0], [-0.5, 0.5])                         # three classes: AUC not defined here
    multi = StreamTrainer(DecisionTreeClassifier(buffer_size=200)).run(_chunks(X, y3))
    assert np.isnan(multi["cumulative_auc"]).all()
    assert not np.isnan(multi["cumulative_f1"][-1])


class _NoFlushModel:
    """A streaming model without flush or memory reporting."""

    def partial_fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


@pytest.mark.parametrize(
    "call, error, match",
    [
        (lambda: StreamTrainer(StandardScaler()), TypeError, "predict"),
        (lambda: StreamTrainer(object()), TypeError, "partial_fit"),
        (lambda: StreamTrainer(DecisionTreeClassifier(), window=0), ValueError, "window"),
        (lambda: StreamTrainer(DecisionTreeClassifier(), metrics={"accuracy": StreamingAccuracy()}),
         ValueError, "clashes"),
        (lambda: StreamTrainer(_NoFlushModel()).flush(), ValueError, "flush"),
    ],
    ids=["no_predict", "no_partial_fit", "window", "metric_name", "no_flush"],
)
def test_invalid_setup_raises_and_empty_tree_uses_no_memory(call, error, match):
    with pytest.raises(error, match=match):
        call()
    assert DecisionTreeClassifier().memory_usage() == {"buffer": 0, "tree": 0}
