"""Train and evaluate a model on a stream of chunks, logging metrics and memory.

``StreamTrainer`` evaluates test-then-train: each chunk is scored by the
current model before the model learns from it, so every logged score is on
data the model has not seen. The log, ``history_``, is a dict of lists with
one entry per chunk, ready for the plotting functions in ``visualise``.
"""

import numpy as np

from .metrics import StreamingAccuracy, StreamingAUC, StreamingF1, accuracy
from .utils import _check_int

# All that can be imported from this module
__all__ = ["StreamTrainer"]

# Columns of history_, before any user metrics
_LOG_KEYS = (
    "chunk", "n_samples", "accuracy", "cumulative_accuracy", "rolling_accuracy",
    "cumulative_f1", "cumulative_auc", "retrained", "retrain_time", "n_retrains",
    "n_buffered", "buffer_bytes", "tree_bytes",
)


class StreamTrainer:
    """Run a streaming model chunk by chunk and log what happens.

    Parameters
    ----------
    model : object
        A ``Pipeline`` or an estimator with ``partial_fit(X, y)`` and
        ``predict(X)``. With ``predict_proba`` and binary labels, AUC is logged.
    window : int, default 5
        Number of recent scored chunks used for ``rolling_accuracy`` (>= 1).
    metrics : dict of str to streaming metric, optional
        Extra label metrics (``update(y_true, y_pred)`` / ``result()``), such
        as ``{"precision": StreamingPrecision()}``; each is logged under its name.
    verbose : bool, default False
        Print one summary line per processed chunk.

    Attributes
    ----------
    history_ : dict of str to list
        One entry per processed chunk for every key: ``chunk``, ``n_samples``,
        ``accuracy`` (this chunk), ``cumulative_accuracy``, ``rolling_accuracy``,
        ``cumulative_f1`` (macro), ``cumulative_auc``, ``retrained``,
        ``retrain_time`` (seconds), ``n_retrains``, ``n_buffered``,
        ``buffer_bytes``, ``tree_bytes``, plus the names in ``metrics``.

    Raises
    ------
    TypeError
        If ``model`` lacks ``partial_fit`` or ``predict``, or ``window`` is
        not an integer.
    ValueError
        If ``window`` < 1, or a metric name clashes with a log key or the
        metric lacks ``update``/``result``.

    Notes
    -----
    Until the model has trained once (its final estimator has ``classes_``),
    scores are NaN and the metrics are not updated. AUC counts only chunks
    scored once the tree knows both classes, and is NaN for good once a
    third label appears. Retrain details and
    memory come from the final estimator's ``n_retrains_``, ``fit_time_``,
    ``n_buffered_`` and ``memory_usage()`` when it has them, else NaN.

    Examples
    --------
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=4))
    >>> X, y = [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
    >>> history = trainer.run([(X, y), (X, y)])
    >>> history["accuracy"]
    [nan, 1.0]
    """

    def __init__(self, model, window: int = 5, metrics=None, verbose: bool = False):
        for method in ("partial_fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise TypeError(f"model must provide {method}, got {type(model).__name__}")

        _check_int(window, "window", minimum=1)
        metrics = {} if metrics is None else dict(metrics)

        for name, metric in metrics.items():
            if name in _LOG_KEYS:
                raise ValueError(f"metric name {name!r} clashes with a log key")
            if not (callable(getattr(metric, "update", None)) and callable(getattr(metric, "result", None))):
                raise ValueError(f"metric {name!r} must provide update and result")

        self.model = model
        self.window = window
        self.metrics = metrics
        self.verbose = verbose
        self.reset()

    def reset(self) -> "StreamTrainer":
        """Clear the log and the metrics; the model is left as it is.

        Returns
        -------
        StreamTrainer
            The instance (self).
        """
        self._accuracy = StreamingAccuracy()
        self._rolling = StreamingAccuracy(window=self.window)
        self._f1 = StreamingF1(average="macro")
        self._auc = None                  # created once the tree knows 2 classes, with its positive label
        self._auc_valid = True            # False once more than 2 labels appear
        self._labels = None               # labels seen in y and in the tree's classes_

        for metric in self.metrics.values():
            if callable(getattr(metric, "reset", None)):
                metric.reset()

        self._last_fit = {"retrained": False, "retrain_time": np.nan, "n_retrains": np.nan}
        self.history_ = {key: [] for key in (*_LOG_KEYS, *self.metrics)}

        return self

    def score_chunk(self, X, y) -> dict:
        """Score the current model on a chunk and update the metrics; nothing is trained.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the model.
        y : array_like, shape (n_samples,)
            True labels.

        Returns
        -------
        dict
            ``n_samples``, ``accuracy`` (NaN before the first training), the
            cumulative, rolling, F1 and AUC values, and each extra metric.

        Raises
        ------
        Exception
            Validation errors from the model or the metrics propagate.

        Notes
        -----
        Time: the model's predict cost plus O(n) for the metrics.
        """
        y = np.asarray(y)
        record = {"n_samples": int(y.size), "accuracy": np.nan}

        if y.size and self._is_ready():
            y_pred = self.model.predict(X)
            record["accuracy"] = accuracy(y, y_pred)

            for metric in (self._accuracy, self._rolling, self._f1, *self.metrics.values()):
                metric.update(y, y_pred)

            self._update_auc(X, y)

        record["cumulative_accuracy"] = self._accuracy.result()
        record["rolling_accuracy"] = self._rolling.result()
        record["cumulative_f1"] = self._f1.result()
        record["cumulative_auc"] = self._auc.result() if self._auc is not None and self._auc_valid else np.nan

        for name, metric in self.metrics.items():
            record[name] = metric.result()

        return record

    def fit_chunk(self, X, y) -> "StreamTrainer":
        """Train the model on a chunk with ``partial_fit`` and note whether it retrained.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the model.
        y : array_like, shape (n_samples,)
            Labels.

        Returns
        -------
        StreamTrainer
            The instance (self).

        Raises
        ------
        Exception
            Validation errors from the model propagate.
        """
        final = self._final_estimator()
        before = getattr(final, "n_retrains_", None)
        self.model.partial_fit(X, y)
        after = getattr(final, "n_retrains_", None)

        retrained = before is not None and after > before
        self._last_fit = {
            "retrained": retrained,
            "retrain_time": final.fit_time_ if retrained else np.nan,
            "n_retrains": np.nan if after is None else after,
        }

        return self

    def process_chunk(self, X, y) -> dict:
        """Score a chunk, then train on it, and append one record to ``history_``.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the model.
        y : array_like, shape (n_samples,)
            Labels.

        Returns
        -------
        dict
            The logged record (one value per ``history_`` key).

        Raises
        ------
        Exception
            Validation errors from the model or the metrics propagate.
        """
        record = self.score_chunk(X, y)
        self.fit_chunk(X, y)
        record.update(self._last_fit)
        record.update(self._memory())
        record["chunk"] = len(self.history_["chunk"])

        for key, values in self.history_.items():
            values.append(record[key])

        if self.verbose:
            print(_format(record))

        return record

    def run(self, chunks, flush: bool = True) -> dict:
        """Process every ``(X, y)`` chunk of an iterable, in order.

        Parameters
        ----------
        chunks : iterable of (array_like, array_like)
            The stream, e.g. a generator over a CSV file.
        flush : bool, default True
            At the end, let the model train on rows left in its buffer.

        Returns
        -------
        dict
            ``history_``.

        Notes
        -----
        The flush runs after the last record and is not logged: ``history_``
        keeps one entry per chunk, so its last ``n_retrains`` and
        ``n_buffered`` describe the model before the flush.
        """
        for X, y in chunks:
            self.process_chunk(X, y)

        if flush and callable(getattr(self.model, "flush", None)):
            self.model.flush()

        return self.history_

    def flush(self) -> "StreamTrainer":
        """Let the model train on rows left in its buffer.

        Returns
        -------
        StreamTrainer
            The instance (self).

        Raises
        ------
        ValueError
            If the model does not provide flush.
        """
        if not callable(getattr(self.model, "flush", None)):
            raise ValueError("the model does not provide flush")

        self.model.flush()

        return self

    def _final_estimator(self):
        """The model itself, or the last step of a Pipeline."""
        steps = getattr(self.model, "steps", None)

        return steps[-1][1] if steps else self.model

    def _is_ready(self) -> bool:
        """True once the final estimator has been trained (it has ``classes_``)."""
        return hasattr(self._final_estimator(), "classes_")

    def _update_auc(self, X, y: np.ndarray) -> None:
        """Feed positive-class scores to the AUC while the stream stays binary."""
        if not self._auc_valid or not callable(getattr(self.model, "predict_proba", None)):
            return

        classes = self._final_estimator().classes_
        labels = np.union1d(y, classes)
        self._labels = labels if self._labels is None else np.union1d(self._labels, labels)

        if self._labels.size > 2:
            self._auc_valid = False      # AUC here is binary only
            return

        if classes.size < 2:             # the tree knows one class: no score to rank yet
            return

        if self._auc is None:
            self._auc = StreamingAUC(pos_label=classes[1])

        self._auc.update(y, self.model.predict_proba(X)[:, 1])

    def _memory(self) -> dict:
        """Buffer size and memory of the final estimator, NaN when it does not report them."""
        final = self._final_estimator()
        usage = final.memory_usage() if callable(getattr(final, "memory_usage", None)) else {}

        return {
            "n_buffered": getattr(final, "n_buffered_", np.nan),
            "buffer_bytes": usage.get("buffer", np.nan),
            "tree_bytes": usage.get("tree", np.nan),
        }


def _format(record: dict) -> str:
    """One summary line for a logged record."""
    retrain = f"retrained {record['retrain_time'] * 1000:6.1f} ms" if record["retrained"] else "retrained      -  "

    return (
        f"chunk {record['chunk']:4d} | n {record['n_samples']:5d} | acc {record['accuracy']:.3f} "
        f"| cum {record['cumulative_accuracy']:.3f} | roll {record['rolling_accuracy']:.3f} "
        f"| {retrain} | buffer {record['buffer_bytes'] / 1024:7.1f} kB"
    )
