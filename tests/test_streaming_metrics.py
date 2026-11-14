"""Unit tests for the streaming metrics in :mod:`numcompute_stream.metrics`."""

import numpy as np
import pytest

from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingPrecision,
    StreamingRecall,
    accuracy,
    auc,
    confusion_matrix,
    f1,
    precision,
    recall,
    roc_curve,
)


def _chunks(k=3, n_chunks=4, size=25):
    """Split noisy predictions of k classes into equal chunks."""
    rng = np.random.default_rng(5)
    n = n_chunks * size
    y_true = rng.integers(0, k, size=n)
    y_pred = np.where(rng.random(n) < 0.7, y_true, rng.integers(0, k, size=n))
    return np.split(y_true, n_chunks), np.split(y_pred, n_chunks)


@pytest.mark.parametrize("average, k", [("binary", 2), ("macro", 3), ("micro", 3), ("weighted", 3)])
def test_streaming_scores_match_batch_metrics(average, k):
    trues, preds = _chunks(k=k)
    acc = StreamingAccuracy()
    streams = {
        precision: StreamingPrecision(average=average),
        recall: StreamingRecall(average=average),
        f1: StreamingF1(average=average),
    }
    for t, p in zip(trues, preds):
        acc.update(t, p)
        for metric in streams.values():
            metric.update(t, p)

    y_true, y_pred = np.concatenate(trues), np.concatenate(preds)
    assert acc.result() == pytest.approx(accuracy(y_true, y_pred))
    for batch, metric in streams.items():
        assert metric.result() == pytest.approx(batch(y_true, y_pred, average=average))


def test_confusion_matrix_accumulates_and_grows_with_new_labels():
    chunks = [([0, 1], [0, 1]), ([1, 1], [0, 1]), ([2, 0], [2, 1]), ([2], [0])]
    cm = StreamingConfusionMatrix()
    for t, p in chunks:
        cm.update(t, p)                     # label 2 first appears in the third chunk

    y_true = np.concatenate([t for t, _ in chunks])
    y_pred = np.concatenate([p for _, p in chunks])
    np.testing.assert_array_equal(cm.labels_, [0, 1, 2])
    np.testing.assert_array_equal(cm.result(), confusion_matrix(y_true, y_pred))

    # A fixed label order is kept as given
    fixed = StreamingConfusionMatrix(labels=["spam", "ham"]).update(["ham", "spam"], ["ham", "ham"])
    np.testing.assert_array_equal(fixed.result(), [[0, 1], [0, 1]])


def test_rolling_window_only_counts_the_last_chunks():
    trues, preds = _chunks(k=3)
    acc = StreamingAccuracy(window=2)
    macro_f1 = StreamingF1(average="macro", window=2)
    cm = StreamingConfusionMatrix(window=2)
    for t, p in zip(trues, preds):
        acc.update(t, p)
        macro_f1.update(t, p)
        cm.update(t, p)

    y_true, y_pred = np.concatenate(trues[-2:]), np.concatenate(preds[-2:])
    assert acc.n_seen_ == 50
    assert acc.result() == pytest.approx(accuracy(y_true, y_pred))
    assert macro_f1.result() == pytest.approx(f1(y_true, y_pred, average="macro"))
    np.testing.assert_array_equal(cm.result(), confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))


def test_empty_state_empty_chunks_and_reset():
    acc = StreamingAccuracy()
    assert np.isnan(acc.result()) and np.isnan(StreamingPrecision().result())
    assert StreamingConfusionMatrix().result().shape == (0, 0)

    acc.update([], [])
    assert acc.n_seen_ == 0
    acc.update([1, 0], [1, 1])
    assert acc.result() == 0.5

    acc.reset()
    assert np.isnan(acc.result()) and acc.labels_ is None and acc.n_seen_ == 0


def test_streaming_auc_matches_batch_auc():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=200)
    scores = np.round(np.clip(0.3 * y + 0.7 * rng.random(200), 0, 1), 1)    # on a 0.1 grid: exact
    roc, recent = StreamingAUC(), StreamingAUC(window=2)
    for t, s in zip(np.split(y, 4), np.split(scores, 4)):
        roc.update(t, s)
        recent.update(t, s)

    fpr, tpr, _ = roc_curve(y, scores)
    assert roc.result() == pytest.approx(auc(fpr, tpr))
    fpr, tpr, _ = roc_curve(y[100:], scores[100:])
    assert recent.result() == pytest.approx(auc(fpr, tpr))


def test_streaming_auc_edge_cases():
    roc = StreamingAUC().update([1, 1], [0.4, 0.9])
    assert np.isnan(roc.result())               # no negative sample yet
    roc.update([0], [0.1])
    assert roc.result() == 1.0

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        roc.update([0], [1.5])
    with pytest.raises(ValueError, match="binary"):
        roc.update([2], [0.5])
    assert roc.n_seen_ == 3                     # rejected chunks changed nothing


@pytest.mark.parametrize(
    "call, error, match",
    [
        (lambda: StreamingAccuracy().update([0, 1], [0]), ValueError, "same length"),
        (lambda: StreamingAccuracy().update([0, np.nan], [0, 1]), ValueError, "NaN"),
        (lambda: StreamingAccuracy().update([0, 1], [0, 1]).update(["a"], ["a"]), TypeError, "incompatible"),
        (lambda: StreamingConfusionMatrix(labels=[0, 1]).update([2], [0]), ValueError, "not in labels"),
        (lambda: StreamingConfusionMatrix(labels=[0, 0]), ValueError, "duplicates"),
        (lambda: StreamingPrecision().update([0, 1, 2], [0, 1, 2]).result(), ValueError, "binary"),
        (lambda: StreamingRecall(average="mean"), ValueError, "average"),
        (lambda: StreamingAccuracy(window=0), ValueError, "window"),
        (lambda: StreamingAccuracy(window=1.5), TypeError, "window"),
        (lambda: StreamingAUC(n_bins=0), ValueError, "n_bins"),
    ],
    ids=["length", "nan", "type_switch", "unknown_label", "duplicate_labels",
         "binary_3_labels", "average", "window_0", "window_float", "n_bins"],
)
def test_invalid_input_raises_informative_errors(call, error, match):
    with pytest.raises(error, match=match):
        call()
