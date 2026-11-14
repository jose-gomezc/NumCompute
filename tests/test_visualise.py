"""Unit tests for :mod:`numcompute_stream.visualise`."""

import numpy as np
import pytest

from numcompute_stream.metrics import confusion_matrix
from numcompute_stream.visualise import (
    PALETTE,
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


def test_plot_metric_over_time_lines_events_and_saving(tmp_path):
    values = [np.nan, np.nan, 0.8, 0.85, 0.9]                    # NaN before the first retrain
    ax = plot_metric_over_time(values, title="Accuracy", ylabel="Accuracy")
    (line,) = ax.get_lines()
    np.testing.assert_array_equal(line.get_ydata(), values)
    assert line.get_color() == PALETTE[0] and ax.get_title(loc="left") == "Accuracy"
    assert ax.get_legend() is None                               # one series: the title names it

    path = tmp_path / "metrics.png"
    ax = plot_metric_over_time({"a": [1, 2], "b": [2, 3], "c": [3, 4]}, events=[False, True],
                               events_label="retrain", save_path=path)
    assert len(ax.get_lines()) == 4                              # 3 series + 1 event line
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b", "c", "retrain"]
    assert path.stat().st_size > 0

    with pytest.raises(ValueError, match="non-empty 1-D"):
        plot_metric_over_time([])
    with pytest.raises(ValueError, match="non-empty 1-D"):
        plot_metric_over_time([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="numeric"):
        plot_metric_over_time(["high", "low"])


def test_compare_models_draws_two_labelled_lines():
    ax = compare_models([0.9, 0.8, 0.85], [0.7, 0.75], labels=("buffer_size=100", "buffer_size=500"))
    first, second = ax.get_lines()
    assert (first.get_label(), second.get_label()) == ("buffer_size=100", "buffer_size=500")
    assert (len(first.get_ydata()), len(second.get_ydata())) == (3, 2)
    assert (first.get_color(), second.get_color()) == PALETTE[:2]

    with pytest.raises(ValueError, match="2 different names"):
        compare_models([1.0], [2.0], labels=("only one",))
    with pytest.raises(ValueError, match="2 different names"):
        compare_models([1.0], [2.0], labels=("same", "same"))


def test_predictions_vs_ground_truth_sequence_and_confusion():
    y_true = np.array(["no", "yes", "yes", "no", "yes"])
    y_pred = np.array(["no", "no", "yes", "no", "no"])

    ax = plot_predictions_vs_ground_truth(y_true, y_pred)
    wrong = ax.collections[2]                                    # true, predicted, wrong
    assert len(wrong.get_offsets()) == 2
    assert ax.get_title(loc="left") == "Latest chunk: 3/5 correct"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["no", "yes"]

    ax = plot_predictions_vs_ground_truth(y_true, y_pred, kind="confusion")
    np.testing.assert_array_equal(ax.get_images()[0].get_array(), confusion_matrix(y_true, y_pred))

    with pytest.raises(ValueError, match="same length"):
        plot_predictions_vs_ground_truth(y_true, y_pred[:3])
    with pytest.raises(ValueError, match="kind"):
        plot_predictions_vs_ground_truth(y_true, y_pred, kind="bar")
