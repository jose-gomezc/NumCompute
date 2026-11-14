"""Accuracy, precision, recall, F1, MSE, confusion matrix, ROC, and AUC.

Classification metrics take 1-D label arrays (int, float, bool or str labels).
Precision, recall and F1 are all computed from one confusion matrix, so they
share the same counts. Scalar results are returned as Python floats.

The ``Streaming*`` classes accumulate the same metrics over chunks with
``update`` / ``result`` / ``reset``, either over the whole stream or over a
rolling window of the most recent chunks.
"""

from collections import deque
from collections.abc import Sequence

import numpy as np

from .utils import _check_int

# All that can be imported from this module
__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "mse",
    "roc_curve",
    "auc",
    "StreamingConfusionMatrix",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingAUC",
]

# Possible values for average
_AVERAGES = ("binary", "macro", "micro", "weighted", None)

# Label dtype kinds that can be compared with each other: NumPy compares any two
# numeric kinds by value (True == 1 == 1.0), but numbers never equal text ("0" != 0)
_LABEL_FAMILIES = {
    "b": "number",
    "i": "number",
    "u": "number",
    "f": "number",
    "U": "text",
    "S": "bytes",
}

# Dtype kinds accepted as numeric values (bool counts as 0/1)
_NUMERIC_KINDS = "biuf"


def accuracy(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
) -> float:
    """Fraction of predictions that match the true labels.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.

    Returns
    -------
    float
        Value in [0, 1].

    Raises
    ------
    TypeError
        If ``y_true`` and ``y_pred`` hold labels of incompatible or unsupported types.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN.

    Notes
    -----
    Time Complexity: O(n: number of samples).
    Space Complexity: O(n) for the boolean match array.

    Examples
    --------
    >>> accuracy([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    y_true = _check_labels(y_true, "y_true")
    y_pred = _check_labels(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))
    _check_same_family(y_true, y_pred, ("y_true", "y_pred"))

    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
    labels: np.ndarray | Sequence[int | str | float] | None = None,
) -> np.ndarray:
    """Count of samples for every (true label, predicted label) pair.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.
    labels : array_like, shape (k,), optional
        Labels in the order wanted for the rows and columns. May include labels
        that never appear. Default: the sorted labels found in either input.

    Returns
    -------
    numpy.ndarray of int, shape (k, k)
        ``C[i, j]`` is the number of samples whose true label is ``labels[i]``
        and whose predicted label is ``labels[j]``. Rows = true, columns =
        predicted, so the diagonal holds the correct predictions.

    Raises
    ------
    TypeError
        If the inputs (or ``labels``) hold labels of incompatible or unsupported types.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN;
        or ``labels`` is empty, has duplicates or misses a label in the data.

    Notes
    -----
    Time Complexity: O(n log k + k^2), n samples, k labels.
    Space Complexity: O(n + k^2).
    Every (true, predicted) pair is turned into one index ``t * k + p`` and all
    pairs are counted at once with ``np.bincount``, without a Python loop.

    Examples
    --------
    >>> confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
    array([[2, 0],
           [1, 1]])
    """
    y_true = _check_labels(y_true, "y_true")
    y_pred = _check_labels(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))
    _check_same_family(y_true, y_pred, ("y_true", "y_pred"))

    classes = _resolve_labels(y_true, y_pred, labels)

    return _confusion(y_true, y_pred, classes)


def precision(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
    average: str | None = "binary",
    pos_label: int | float | str = 1,
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """Precision: of the samples predicted as a class, the fraction that truly are.

    ``precision = tp / (tp + fp)`` per class.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        ``"binary"``: only the class ``pos_label`` (at most 2 classes).
        ``"macro"``: unweighted mean over classes.
        ``"micro"``: from the counts summed over all classes.
        ``"weighted"``: mean over classes weighted by their support (true count).
        ``None``: one value per class, in sorted label order.
    pos_label : integer, float, bool or string, default 1
        The positive class when ``average="binary"``.
    zero_division : float, default 0.0
        Value returned where the denominator is 0 (no sample predicted as that
        class), instead of NaN.

    Returns
    -------
    float, or numpy.ndarray of shape (k,) when ``average=None``
        Values in [0, 1] (or ``zero_division``).

    Raises
    ------
    TypeError
        If the inputs or ``pos_label`` hold labels of incompatible or unsupported types,
        ``pos_label`` is not a single label, or ``zero_division`` is not a number.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN;
        ``zero_division`` is NaN; ``average`` is unknown; or ``average="binary"``
        and there are more than 2 classes or ``pos_label`` is not one of the 2 labels.

    Notes
    -----
    Time Complexity: O(n log k + k^2), n samples, k labels.
    Space Complexity: O(n + k^2), dominated by the confusion matrix.

    Examples
    --------
    >>> precision([0, 1, 1, 0], [0, 1, 0, 0])
    1.0
    >>> precision([0, 1, 1, 0], [0, 1, 0, 0], average=None)
    array([0.66666667, 1.        ])
    """
    return _precision_recall_f1(y_true, y_pred, average, pos_label, zero_division)[0]


def recall(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
    average: str | None = "binary",
    pos_label: int | float | str = 1,
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """Recall: of the samples that truly are a class, the fraction predicted as it.

    ``recall = tp / (tp + fn)`` per class.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        How per-class values are combined, see :func:`precision`.
    pos_label : integer, float, bool or string, default 1
        The positive class when ``average="binary"``.
    zero_division : float, default 0.0
        Value returned where the denominator is 0 (no sample truly of that
        class), instead of NaN.

    Returns
    -------
    float, or numpy.ndarray of shape (k,) when ``average=None``
        Values in [0, 1] (or ``zero_division``).

    Raises
    ------
    TypeError
        If the inputs or ``pos_label`` hold labels of incompatible or unsupported types,
        ``pos_label`` is not a single label, or ``zero_division`` is not a number.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN;
        ``zero_division`` is NaN; ``average`` is unknown; or ``average="binary"``
        and there are more than 2 classes or ``pos_label`` is not one of the 2 labels.

    Notes
    -----
    Time Complexity: O(n log k + k^2), n samples, k labels.
    Space Complexity: O(n + k^2), dominated by the confusion matrix.
    ``recall(..., average="weighted")`` always equals :func:`accuracy`.

    Examples
    --------
    >>> recall([0, 1, 1, 0], [0, 1, 0, 0])
    0.5
    """
    return _precision_recall_f1(y_true, y_pred, average, pos_label, zero_division)[1]


def f1(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
    average: str | None = "binary",
    pos_label: int | float | str = 1,
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """F1 score: harmonic mean of precision and recall.

    Computed per class as ``2 * tp / (2 * tp + fp + fn)``, which equals
    ``2 * p * r / (p + r)`` and stays defined when only one of them is.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        How per-class values are combined, see :func:`precision`.
    pos_label : integer, float, bool or string, default 1
        The positive class when ``average="binary"``.
    zero_division : float, default 0.0
        Value returned where the denominator is 0 (the class never appears in
        ``y_true`` nor ``y_pred``), instead of NaN.

    Returns
    -------
    float, or numpy.ndarray of shape (k,) when ``average=None``
        Values in [0, 1] (or ``zero_division``).

    Raises
    ------
    TypeError
        If the inputs or ``pos_label`` hold labels of incompatible or unsupported types,
        ``pos_label`` is not a single label, or ``zero_division`` is not a number.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN;
        ``zero_division`` is NaN; ``average`` is unknown; or ``average="binary"``
        and there are more than 2 classes or ``pos_label`` is not one of the 2 labels.

    Notes
    -----
    Time Complexity: O(n log k + k^2), n samples, k labels.
    Space Complexity: O(n + k^2), dominated by the confusion matrix.
    ``f1(..., average="micro")`` always equals :func:`accuracy`.

    Examples
    --------
    >>> f1([0, 1, 1, 0], [0, 1, 0, 0])
    0.6666666666666666
    """
    return _precision_recall_f1(y_true, y_pred, average, pos_label, zero_division)[2]


def mse(
    y_true: np.ndarray | Sequence[int | float],
    y_pred: np.ndarray | Sequence[int | float],
) -> float:
    """Mean squared error between true and predicted values.

    Parameters
    ----------
    y_true : array_like of numbers, shape (n,)
        True values.
    y_pred : array_like of numbers, shape (n,)
        Predicted values.

    Returns
    -------
    float
        ``mean((y_true - y_pred) ** 2)`` over all samples, >= 0.

    Raises
    ------
    TypeError
        If an input is not numeric.
    ValueError
        If an input is not 1-D, is empty or contains NaN, or the lengths differ.

    Notes
    -----
    Time Complexity: O(n: number of samples).
    Space Complexity: O(n) for the difference array.

    Examples
    --------
    >>> mse([1, 2, 3], [1, 2, 5])
    1.3333333333333333
    """
    y_true = _check_values(y_true, "y_true")
    y_pred = _check_values(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))

    return float(np.mean(np.square(y_true - y_pred)))


def roc_curve(
    y_true: np.ndarray | Sequence[int | str | float],
    y_score: np.ndarray | Sequence[float],
    pos_label: int | float | str = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Receiver operating characteristic curve for a binary classifier.

    Each threshold ``t`` predicts positive for every sample with score >= ``t``;
    the curve gives the false and true positive rates for every distinct score.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels, at most 2 distinct values (int, float, bool or str).
    y_score : array_like of numbers, shape (n,)
        Score for the positive class (higher = more likely positive), e.g. a
        probability.
    pos_label : integer, float, bool or string, default 1
        The label of the positive class in ``y_true``.

    Returns
    -------
    fpr : numpy.ndarray, shape (m + 1,)
        False positive rate ``fp / n_negatives`` at each threshold, from 0 up to 1.
    tpr : numpy.ndarray, shape (m + 1,)
        True positive rate ``tp / n_positives`` at each threshold, from 0 up to 1.
    thresholds : numpy.ndarray, shape (m + 1,)
        The ``m`` distinct scores in decreasing order, after a first ``inf``
        (nothing predicted positive).

    Raises
    ------
    TypeError
        If ``y_true`` has an unsupported label dtype, ``y_score`` is not numeric,
        ``pos_label`` is not the same type of label as ``y_true``, or ``pos_label``
        is not a single label.
    ValueError
        If the inputs are empty, not 1-D or of different lengths; ``y_true`` or
        ``y_score`` contains NaN; ``y_true`` has more than 2 classes; or ``y_true``
        has no positive or no negative sample (a rate would divide by zero).

    Notes
    -----
    Time Complexity: O(n log n), for sorting the scores.
    Space Complexity: O(n).
    Samples with the same score cross a threshold together, so tied scores give
    one point on the curve.

    Examples
    --------
    >>> fpr, tpr, thresholds = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    """
    y_true = _check_labels(y_true, "y_true")
    y_score = _check_values(y_score, "y_score")
    _check_same_length(y_true, y_score, ("y_true", "y_score"))

    pos = _check_pos_label(pos_label, y_true)

    classes = np.unique(y_true)

    if classes.size > 2:
        raise ValueError(
            f"roc_curve is for binary labels, got {classes.size} classes: {classes.tolist()}"
        )

    positive = y_true == pos[0]
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"y_true needs positive and negative samples (pos_label={pos_label!r}), "
            f"got {n_pos} positive and {n_neg} negative"
        )

    # Highest score first; stable so equal scores keep their order
    order = np.argsort(-y_score, kind="stable")
    scores = y_score[order]

    tps = np.cumsum(positive[order])  # positives with score >= scores[i]
    fps = np.arange(1, scores.size + 1) - tps  # negatives with score >= scores[i]

    # Last position of each run of equal scores: tied samples cross the threshold together
    last = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])

    fpr = np.r_[0.0, fps[last] / n_neg]
    tpr = np.r_[0.0, tps[last] / n_pos]
    thresholds = np.r_[np.inf, scores[last]]

    return fpr, tpr, thresholds


def auc(
    x: np.ndarray | Sequence[int | float],
    y: np.ndarray | Sequence[int | float],
) -> float:
    """Area under a curve given by points (x, y), with the trapezoid rule.

    Parameters
    ----------
    x : array_like of numbers, shape (m,)
        x coordinates, all increasing or all decreasing (repeats allowed),
        e.g. ``fpr`` from :func:`roc_curve`.
    y : array_like of numbers, shape (m,)
        y coordinates, e.g. ``tpr`` from :func:`roc_curve`.

    Returns
    -------
    float
        The area. For decreasing ``x`` the sign is flipped, so the same curve
        gives the same area in either direction.

    Raises
    ------
    TypeError
        If an input is not numeric.
    ValueError
        If an input is not 1-D, contains NaN or has fewer than 2 points; the
        lengths differ; or ``x`` is not monotonic.

    Notes
    -----
    Time Complexity: O(m: number of points).
    Space Complexity: O(m).
    Written out instead of ``np.trapz``, which NumPy 2 renamed to ``np.trapezoid``.

    Examples
    --------
    >>> auc([0, 1], [0, 1])
    0.5
    """
    x = _check_values(x, "x")
    y = _check_values(y, "y")
    _check_same_length(x, y, ("x", "y"))

    if x.size < 2:
        raise ValueError(f"auc needs at least 2 points, got {x.size}")

    dx = np.diff(x)

    if (dx < 0).any() and (dx > 0).any():
        raise ValueError("x must be monotonic (all increasing or all decreasing)")

    direction = -1.0 if (dx < 0).any() else 1.0
    area = np.sum(dx * (y[1:] + y[:-1]) / 2)

    return float(direction * area)


# ---------------------------------------------------------------------------
# Streaming metrics
# ---------------------------------------------------------------------------

class _StreamingConfusion:
    """Base class: accumulates a confusion matrix over chunks, optionally over a window."""

    def __init__(self, labels=None, window: int | None = None):
        if labels is not None:
            labels = _check_labels(labels, "labels")

            if np.unique(labels).size != labels.size:
                raise ValueError(f"labels has duplicates: {labels.tolist()}")

        if window is not None:
            _check_int(window, "window", minimum=1)

        self.labels = labels
        self.window = window
        self.reset()

    def reset(self):
        """Forget every chunk seen so far.

        Returns
        -------
        self
            The emptied metric.
        """
        self.labels_ = self.labels
        self.n_seen_ = 0
        self._chunks = deque()

        if self.labels is None:
            self._total = None
        else:
            self._total = np.zeros((self.labels.size, self.labels.size), dtype=np.int64)

        return self

    def update(self, y_true, y_pred):
        """Add one chunk of true and predicted labels.

        Parameters
        ----------
        y_true : array_like, shape (n,)
            True labels of the chunk (int, float, bool or str).
        y_pred : array_like, shape (n,)
            Predicted labels, same kind as ``y_true``.

        Returns
        -------
        self
            The updated metric.

        Raises
        ------
        TypeError
            If the labels have unsupported or incompatible types, including a
            change of type during the stream.
        ValueError
            If the inputs are not 1-D, have different lengths or contain NaN,
            or hold a label missing from a fixed ``labels``.

        Notes
        -----
        An empty chunk is ignored; a rejected chunk leaves the state unchanged.
        Time: O(n log k + k**2) per chunk. Memory: O(k**2), or O(window * k**2).
        """
        if np.size(y_true) == 0 and np.size(y_pred) == 0:
            return self

        y_true = _check_labels(y_true, "y_true")
        y_pred = _check_labels(y_pred, "y_pred")
        _check_same_length(y_true, y_pred, ("y_true", "y_pred"))
        _check_same_family(y_true, y_pred, ("y_true", "y_pred"))

        if self.labels_ is not None:
            _check_same_family(self.labels_, y_true, ("earlier labels", "y_true"))

        if self.labels is not None:
            _resolve_labels(y_true, y_pred, self.labels)       # raises on an unknown label

        # All checks passed: only now change the state
        if self.labels is None:
            self._grow_labels(np.unique(np.concatenate([y_true, y_pred])))

        cm = _confusion(y_true, y_pred, self.labels_)
        self._total = self._total + cm
        self.n_seen_ += y_true.size

        if self.window is not None:
            self._chunks.append((cm, y_true.size))

            if len(self._chunks) > self.window:                # drop the oldest chunk
                old_cm, old_n = self._chunks.popleft()
                self._total = self._total - old_cm
                self.n_seen_ -= old_n

        return self

    def _grow_labels(self, found: np.ndarray) -> None:
        """Add new labels, moving the stored counts to their new positions."""
        if self.labels_ is None:
            self.labels_ = found
            self._total = np.zeros((found.size, found.size), dtype=np.int64)
            return

        new = np.union1d(self.labels_, found)

        if new.size == self.labels_.size:
            return

        self._total = _expand_confusion(self._total, self.labels_, new)
        self._chunks = deque((_expand_confusion(cm, self.labels_, new), n) for cm, n in self._chunks)
        self.labels_ = new


class StreamingConfusionMatrix(_StreamingConfusion):
    """Confusion matrix accumulated over chunks.

    Parameters
    ----------
    labels : array_like, shape (k,), optional
        Fixed label order for the rows and columns; a label outside it raises.
        Default: the sorted labels seen so far, growing as new ones arrive.
    window : int or None, default None
        Count only the most recent ``window`` chunks; None counts the whole stream.

    Attributes
    ----------
    labels_ : ndarray, shape (k,), or None
        Row and column labels; None before any data.
    n_seen_ : int
        Samples counted (inside the window, if any).

    Raises
    ------
    TypeError
        If ``labels`` has an unsupported dtype or ``window`` is not an integer.
    ValueError
        If ``labels`` is empty or has duplicates, or ``window`` < 1.

    Examples
    --------
    >>> cm = StreamingConfusionMatrix()
    >>> cm.update([0, 1], [0, 0]).update([1, 1], [1, 1]).result()
    array([[1, 0],
           [1, 2]])
    """

    def result(self) -> np.ndarray:
        """Current confusion matrix.

        Returns
        -------
        ndarray of int, shape (k, k)
            Rows are true labels and columns predicted labels, in ``labels_``
            order; shape (0, 0) before any data.
        """
        return np.zeros((0, 0), dtype=np.int64) if self._total is None else self._total.copy()


class StreamingAccuracy(_StreamingConfusion):
    """Accuracy accumulated over chunks.

    Parameters
    ----------
    window : int or None, default None
        Score only the most recent ``window`` chunks; None scores the whole stream.

    Attributes
    ----------
    labels_ : ndarray or None
        Labels seen so far.
    n_seen_ : int
        Samples counted (inside the window, if any).

    Raises
    ------
    TypeError
        If ``window`` is not an integer.
    ValueError
        If ``window`` < 1.

    Examples
    --------
    >>> acc = StreamingAccuracy()
    >>> acc.update([0, 1], [0, 0]).update([1, 1], [1, 1]).result()
    0.75
    """

    def __init__(self, window: int | None = None):
        super().__init__(window=window)

    def result(self) -> float:
        """Fraction of correct predictions so far.

        Returns
        -------
        float
            Value in [0, 1]; NaN before any data.
        """
        if self.n_seen_ == 0:
            return float("nan")

        return float(np.trace(self._total) / self.n_seen_)


class _StreamingPRF(_StreamingConfusion):
    """Base class for streaming precision, recall and F1 (one of the three scores)."""

    _score = 0     # index into (precision, recall, f1)

    def __init__(self, average: str | None = "binary", pos_label=1,
                 zero_division: float = 0.0, window: int | None = None):
        self.zero_division = _check_prf_options(average, zero_division)
        self.average = average
        self.pos_label = pos_label
        super().__init__(window=window)

    def result(self):
        """Current score.

        Returns
        -------
        float or ndarray of shape (k,)
            One value, or one per label with counts in the current window
            when ``average=None``; NaN before any data.

        Raises
        ------
        TypeError
            If ``pos_label`` is not the same kind of label as the data.
        ValueError
            If ``average="binary"`` and more than 2 labels were seen, or
            ``pos_label`` is not one of the 2 labels.
        """
        if self.n_seen_ == 0:
            return float("nan")

        # Only labels with counts now: one seen only in chunks that left the window is not a class
        present = (self._total.sum(axis=0) + self._total.sum(axis=1)) > 0
        cm = self._total[np.ix_(present, present)]

        scores = _prf_from_confusion(cm, self.labels_[present], self.average,
                                     self.pos_label, self.zero_division)

        return scores[self._score]


class StreamingPrecision(_StreamingPRF):
    """Precision accumulated over chunks: ``tp / (tp + fp)``.

    Parameters
    ----------
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        How per-label scores are combined, as in :func:`precision`.
    pos_label : int, float, bool or str, default 1
        Positive label when ``average="binary"``.
    zero_division : float, default 0.0
        Value used when a denominator is zero.
    window : int or None, default None
        Score only the most recent ``window`` chunks; None scores the whole stream.

    Attributes
    ----------
    labels_ : ndarray or None
        Labels seen so far.
    n_seen_ : int
        Samples counted (inside the window, if any).

    Raises
    ------
    TypeError
        If ``zero_division`` is not a number or ``window`` is not an integer.
    ValueError
        If ``average`` is unknown, ``zero_division`` is NaN or ``window`` < 1.
    """

    _score = 0


class StreamingRecall(_StreamingPRF):
    """Recall accumulated over chunks: ``tp / (tp + fn)``.

    Parameters
    ----------
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        How per-label scores are combined, as in :func:`recall`.
    pos_label : int, float, bool or str, default 1
        Positive label when ``average="binary"``.
    zero_division : float, default 0.0
        Value used when a denominator is zero.
    window : int or None, default None
        Score only the most recent ``window`` chunks; None scores the whole stream.

    Attributes
    ----------
    labels_ : ndarray or None
        Labels seen so far.
    n_seen_ : int
        Samples counted (inside the window, if any).

    Raises
    ------
    TypeError
        If ``zero_division`` is not a number or ``window`` is not an integer.
    ValueError
        If ``average`` is unknown, ``zero_division`` is NaN or ``window`` < 1.
    """

    _score = 1


class StreamingF1(_StreamingPRF):
    """F1 score accumulated over chunks: ``2 tp / (2 tp + fp + fn)``.

    Parameters
    ----------
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        How per-label scores are combined, as in :func:`f1`.
    pos_label : int, float, bool or str, default 1
        Positive label when ``average="binary"``.
    zero_division : float, default 0.0
        Value used when a denominator is zero.
    window : int or None, default None
        Score only the most recent ``window`` chunks; None scores the whole stream.

    Attributes
    ----------
    labels_ : ndarray or None
        Labels seen so far.
    n_seen_ : int
        Samples counted (inside the window, if any).

    Raises
    ------
    TypeError
        If ``zero_division`` is not a number or ``window`` is not an integer.
    ValueError
        If ``average`` is unknown, ``zero_division`` is NaN or ``window`` < 1.
    """

    _score = 2


class StreamingAUC:
    """Binary ROC AUC accumulated over chunks, from score histograms.

    Parameters
    ----------
    pos_label : int, float, bool or str, default 1
        Label of the positive class.
    n_bins : int, default 1000
        Number of equal-width score bins on [0, 1].
    window : int or None, default None
        Score only the most recent ``window`` chunks; None scores the whole stream.

    Attributes
    ----------
    labels_ : ndarray or None
        Labels seen so far (at most 2).
    n_seen_ : int
        Samples counted (inside the window, if any).

    Raises
    ------
    TypeError
        If ``pos_label`` is not a single supported label, or ``n_bins`` or
        ``window`` is not an integer.
    ValueError
        If ``n_bins`` or ``window`` < 1.

    Notes
    -----
    Positives and negatives are counted per score bin, so memory is
    O(n_bins) however long the stream. Scores in the same bin are treated as
    tied (half credit), so the result equals the exact AUC whenever no two
    distinct scores share a bin. Update: O(n + n_bins); result: O(n_bins).

    Examples
    --------
    >>> roc = StreamingAUC()
    >>> roc.update([0, 1], [0.2, 0.9]).update([0, 1], [0.6, 0.4]).result()
    0.75
    """

    def __init__(self, pos_label=1, n_bins: int = 1000, window: int | None = None):
        if np.ndim(pos_label) != 0:
            raise TypeError(f"pos_label must be a single label, got {pos_label!r}")

        _check_int(n_bins, "n_bins", minimum=1)

        if window is not None:
            _check_int(window, "window", minimum=1)

        self._pos_label = _check_labels([pos_label], "pos_label")
        self.pos_label = pos_label
        self.n_bins = n_bins
        self.window = window
        self.reset()

    def reset(self) -> "StreamingAUC":
        """Forget every chunk seen so far.

        Returns
        -------
        StreamingAUC
            The emptied metric.
        """
        self.labels_ = None
        self.n_seen_ = 0
        self._pos = np.zeros(self.n_bins, dtype=np.int64)
        self._neg = np.zeros(self.n_bins, dtype=np.int64)
        self._chunks = deque()

        return self

    def update(self, y_true, y_score) -> "StreamingAUC":
        """Add one chunk of true labels and positive-class scores.

        Parameters
        ----------
        y_true : array_like, shape (n,)
            True labels of the chunk; at most 2 distinct labels over the stream.
        y_score : array_like of float, shape (n,)
            Scores in [0, 1] for the positive class, e.g. ``predict_proba(X)[:, 1]``.

        Returns
        -------
        StreamingAUC
            The updated metric.

        Raises
        ------
        TypeError
            If the labels have an unsupported type or a different type than
            ``pos_label``, or ``y_score`` is not numeric.
        ValueError
            If the inputs are not 1-D, have different lengths or contain NaN,
            a score is outside [0, 1], or more than 2 labels are seen.

        Notes
        -----
        An empty chunk is ignored; a rejected chunk leaves the state unchanged.
        Time: O(n + n_bins).
        """
        if np.size(y_true) == 0 and np.size(y_score) == 0:
            return self

        y_true = _check_labels(y_true, "y_true")
        y_score = _check_values(y_score, "y_score")
        _check_same_length(y_true, y_score, ("y_true", "y_score"))
        _check_same_family(y_true, self._pos_label, ("y_true", "pos_label"))

        if (y_score < 0).any() or (y_score > 1).any():
            raise ValueError("y_score must be in [0, 1], e.g. probabilities from predict_proba")

        labels = np.unique(y_true) if self.labels_ is None else np.union1d(self.labels_, y_true)

        if labels.size > 2:
            raise ValueError(f"StreamingAUC is for binary labels, got {labels.tolist()}")

        # All checks passed: only now change the state
        self.labels_ = labels
        positive = y_true == self._pos_label[0]
        bins = np.minimum((y_score * self.n_bins).astype(np.int64), self.n_bins - 1)
        pos = np.bincount(bins[positive], minlength=self.n_bins)
        neg = np.bincount(bins[~positive], minlength=self.n_bins)

        self._pos = self._pos + pos
        self._neg = self._neg + neg
        self.n_seen_ += y_true.size

        if self.window is not None:
            self._chunks.append((pos, neg))

            if len(self._chunks) > self.window:                # drop the oldest chunk
                old_pos, old_neg = self._chunks.popleft()
                self._pos = self._pos - old_pos
                self._neg = self._neg - old_neg
                self.n_seen_ -= int(old_pos.sum() + old_neg.sum())

        return self

    def result(self) -> float:
        """Current area under the ROC curve.

        Returns
        -------
        float
            Value in [0, 1]; NaN until both a positive and a negative sample
            have been seen.
        """
        n_pos, n_neg = self._pos.sum(), self._neg.sum()

        if n_pos == 0 or n_neg == 0:
            return float("nan")

        # Highest-score bins first: each bin is one threshold of the ROC curve
        tpr = np.r_[0.0, np.cumsum(self._pos[::-1]) / n_pos]
        fpr = np.r_[0.0, np.cumsum(self._neg[::-1]) / n_neg]

        return auc(fpr, tpr)


def _check_labels(
    y: np.ndarray | Sequence[int | str | float],
    name: str,
) -> np.ndarray:
    """Validate a nonempty vector of numeric or string labels.

    Parameters
    ----------
    y : array_like, shape (n,)
        Labels of a supported numeric, boolean or text dtype.
    name : str
        Argument name included in error messages.

    Returns
    -------
    ndarray, shape (n,)
        Validated labels with their original dtype.

    Raises
    ------
    TypeError
        If label dtype is unsupported.
    ValueError
        If input is empty, not 1-D, ragged or contains floating-point NaN.

    Notes
    -----
    Time/space: O(n) worst case for conversion and NaN validation.
    """
    # Converting to a NumPy array if a list was received, keeping its dtype
    y = np.asarray(y)

    # Check if the array is 1-D: one label per sample
    if y.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got shape {y.shape}; use .ravel() for a column vector"
        )

    # Check if the dtype is a supported label kind: numbers, text or bytes (no objects, no dates)
    if y.dtype.kind not in _LABEL_FAMILIES:
        raise TypeError(
            f"{name} has unsupported label dtype {y.dtype}; use numbers or strings"
        )

    # Check if the array is not empty
    if y.size == 0:
        raise ValueError(f"{name} is empty")

    # Check for NaN, only on float labels: no other kind can hold it, and NaN never equals itself
    if y.dtype.kind == "f" and np.isnan(y).any():
        raise ValueError(f"{name} contains NaN")

    return y


def _check_values(
    y: np.ndarray | Sequence[int | float],
    name: str,
) -> np.ndarray:
    """Validate a nonempty numeric vector and convert it to float.

    Parameters
    ----------
    y : numeric array_like, shape (n,)
        Input values; numeric-looking strings are not accepted.
    name : str
        Argument name used in validation messages.

    Returns
    -------
    ndarray of float, shape (n,)
        Values without NaN; infinity is not rejected by this helper.

    Raises
    ------
    TypeError
        If dtype is not a supported numeric type.
    ValueError
        If input is empty, not 1-D, ragged or contains NaN.

    Notes
    -----
    Time and worst-case extra space: O(n); float64 input may be reused.
    """
    # Converting to a NumPy array if a list was received, keeping its dtype
    y = np.asarray(y)

    # Check if the array is 1-D: one value per sample
    if y.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got shape {y.shape}; use .ravel() for a column vector"
        )

    # Check if dtype is numeric before converting: astype(float) would silently turn "1" into 1.0
    if y.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(f"{name} must be numeric, got dtype {y.dtype}")

    # Check if the array is not empty
    if y.size == 0:
        raise ValueError(f"{name} is empty")

    # Converting to float for the arithmetic; no copy when the input is already float64
    y = y.astype(float, copy=False)

    # Check for NaN, after converting so it always runs on float data
    if np.isnan(y).any():
        raise ValueError(f"{name} contains NaN")

    return y


def _check_same_length(
    a: np.ndarray,
    b: np.ndarray,
    names: tuple[str, str],
) -> None:
    """Check that two validated vectors have equal lengths.

    Parameters
    ----------
    a, b : ndarray, shapes (n,), (m,)
        Vectors whose sizes are compared.
    names : pair of str
        Argument names included in the error message.

    Returns
    -------
    None
        Only validation is performed.

    Raises
    ------
    ValueError
        If the sizes differ.

    Notes
    -----
    Time and space: O(1).
    """
    # Check if both arrays have one entry per sample: no broadcasting
    if a.size != b.size:
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same length, got {a.size} and {b.size}"
        )


def _check_same_family(
    a: np.ndarray,
    b: np.ndarray,
    names: tuple[str, str],
) -> None:
    """Check that two validated label vectors use compatible label families.

    Parameters
    ----------
    a, b : ndarray, shapes (n,), (m,)
        Label vectors with supported dtypes.
    names : pair of str
        Names to include in the error message.

    Returns
    -------
    None
        Only validation is performed.

    Raises
    ------
    TypeError
        If one vector holds numeric labels and the other holds text labels.

    Notes
    -----
    Time and space: O(1).
    """
    # Check if both arrays hold the same family of labels: NumPy would compare 1 with "1" as unequal
    # (or silently promote 1 to "1"), giving wrong scores instead of an error
    if _LABEL_FAMILIES[a.dtype.kind] != _LABEL_FAMILIES[b.dtype.kind]:
        raise TypeError(
            f"{names[0]} ({a.dtype}) and {names[1]} ({b.dtype}) hold labels of incompatible types"
        )


def _check_pos_label(
    pos_label: int | float | str,
    y_true: np.ndarray
) -> np.ndarray:
    """Validate a scalar positive label against the true-label family.

    Parameters
    ----------
    pos_label : scalar
        Positive class identifier.
    y_true : ndarray, shape (n,)
        Previously validated true labels.

    Returns
    -------
    ndarray, shape (1,)
        Positive label with a compatible label family.

    Raises
    ------
    TypeError
        If pos_label is nonscalar, unsupported or incompatible with y_true.
    ValueError
        If pos_label contains NaN.

    Notes
    -----
    Time and space: O(1).
    """
    # Check if pos_label is a single label
    if np.ndim(pos_label) != 0:
        raise TypeError(f"pos_label must be a single label, got {pos_label!r}")

    pos = _check_labels([pos_label], "pos_label")
    _check_same_family(y_true, pos, ("y_true", "pos_label"))

    return pos


def _resolve_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | Sequence[int | str | float] | None = None,
) -> np.ndarray:
    """Resolve the confusion-matrix class order and validate optional labels.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (n,)
        Previously validated, compatible label vectors.
    labels : array_like, shape (k,), or None
        Desired class order. None selects the sorted union of observed labels.

    Returns
    -------
    ndarray, shape (k,)
        Unique class labels covering every observed label.

    Raises
    ------
    TypeError
        If supplied labels have unsupported or incompatible dtype.
    ValueError
        If labels are empty, not 1-D, duplicated, contain NaN or omit a class.

    Notes
    -----
    Time: O(n * log(n) + k * log(k) + n * log(k)); space O(n + k).
    """
    found = np.unique(np.concatenate([y_true, y_pred]))

    if labels is None:
        return found

    labels = _check_labels(labels, "labels")
    _check_same_family(y_true, labels, ("y_true", "labels"))

    if np.unique(labels).size != labels.size:
        raise ValueError(f"labels has duplicates: {labels.tolist()}")

    unknown = found[_label_index(found, labels) < 0]

    if unknown.size > 0:
        raise ValueError(
            f"the data has labels that are not in labels: {unknown.tolist()}"
        )

    return labels


def _label_index(
    y: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Map labels to positions in a possibly unsorted class vector.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Labels to look up.
    classes : ndarray, shape (k,)
        Nonempty unique classes with a compatible dtype.

    Returns
    -------
    ndarray of integers, shape (n,)
        Index in the original classes array, or -1 for an absent label.

    Raises
    ------
    TypeError
        If labels cannot be ordered or compared.
    IndexError
        If a nonempty y is searched in an empty classes array.

    Notes
    -----
    Time: O(k * log(k) + n * log(k)); space O(n + k).
    """
    order = np.argsort(classes, kind="stable")
    sorted_classes = classes[order]

    pos = np.searchsorted(sorted_classes, y)  # where each label is, or would be inserted
    pos = np.minimum(pos, classes.size - 1)  # keep the index valid

    found = sorted_classes[pos] == y

    return np.where(found, order[pos], -1)


def _confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Count checked true/predicted label pairs in the given class order.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (n,)
        Compatible labels, all present in classes.
    classes : ndarray, shape (k,)
        Nonempty unique classes in the desired output order.

    Returns
    -------
    ndarray of integers, shape (k, k)
        Rows are true classes and columns are predicted classes.

    Raises
    ------
    ValueError
        NumPy may reject invalid indices if unchecked labels violate the
        class-membership precondition.

    Notes
    -----
    Time: O(k * log(k) + n * log(k) + k**2); space O(n + k**2).
    """
    k = classes.size
    t = _label_index(y_true, classes)
    p = _label_index(y_pred, classes)

    # One index per (true, predicted) pair, all pairs counted at once
    counts = np.bincount(t * k + p, minlength=k * k)

    return counts.reshape(k, k)


def _expand_confusion(cm: np.ndarray, classes: np.ndarray, new_classes: np.ndarray) -> np.ndarray:
    """Copy ``cm`` (rows/columns in ``classes``) into a zero matrix ordered by ``new_classes``."""
    idx = _label_index(classes, new_classes)
    out = np.zeros((new_classes.size, new_classes.size), dtype=cm.dtype)
    out[np.ix_(idx, idx)] = cm

    return out


def _safe_divide(
    num: np.ndarray | Sequence[int | float],
    den: np.ndarray | Sequence[int | float],
    zero_division: float,
) -> np.ndarray:
    """Divide with a configured replacement for zero denominators.

    Parameters
    ----------
    num, den : numeric array_like
        Numerator and denominator with broadcast-compatible shapes.
    zero_division : float
        Value to return wherever den equals zero.

    Returns
    -------
    ndarray of float, broadcast shape of num and den
        Quotients with zero denominators replaced.

    Raises
    ------
    TypeError
        If arguments cannot be converted to floats.
    ValueError
        If conversion fails or shapes cannot be broadcast.

    Notes
    -----
    Time and space: O(B), B is the number of broadcast output elements.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), zero_division, dtype=float)

    np.divide(num, den, out=out, where=den != 0)

    return out


def _precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str | None,
    pos_label: int | float | str,
    zero_division: float,
) -> tuple:
    """Compute shared per-class or averaged precision, recall and F1.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        True and predicted labels from compatible families.
    average : {'binary', 'macro', 'micro', 'weighted'} or None
        Reduction rule; None returns per-class scores.
    pos_label : scalar
        Positive class for binary averaging.
    zero_division : number
        Replacement for undefined ratios; must not be NaN or boolean.

    Returns
    -------
    tuple of three floats or three ndarrays of shape (k,)
        Precision, recall and F1; per-class arrays use sorted class order.

    Raises
    ------
    TypeError
        If label families, pos_label or zero_division have invalid types.
    ValueError
        If labels, lengths, averaging mode or binary-class configuration
        are invalid, or zero_division is NaN.

    Notes
    -----
    Time: O(n * log(n) + n * log(k) + k**2); space O(n + k**2).
    """
    zero_division = _check_prf_options(average, zero_division)

    y_true = _check_labels(y_true, "y_true")
    y_pred = _check_labels(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))
    _check_same_family(y_true, y_pred, ("y_true", "y_pred"))

    classes = _resolve_labels(y_true, y_pred, None)

    return _prf_from_confusion(
        _confusion(y_true, y_pred, classes), classes, average, pos_label, zero_division
    )


def _check_prf_options(average: str | None, zero_division) -> float:
    """Validate ``average`` and ``zero_division``; return ``zero_division`` as a float."""
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")

    if isinstance(zero_division, bool) or not isinstance(
        zero_division, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"zero_division must be a number, got {zero_division!r}")

    zero_division = float(zero_division)

    if np.isnan(zero_division):
        raise ValueError("zero_division must not be NaN")

    return zero_division


def _prf_from_confusion(
    cm: np.ndarray,
    classes: np.ndarray,
    average: str | None,
    pos_label: int | float | str,
    zero_division: float,
) -> tuple:
    """Precision, recall and F1 from a confusion matrix whose rows and columns follow ``classes``."""
    if average == "binary":
        if classes.size > 2:
            raise ValueError(
                f"average='binary' needs at most 2 classes, got {classes.size}: "
                f"{classes.tolist()}; use 'macro', 'micro' or 'weighted'"
            )

        pos = _check_pos_label(pos_label, classes)

        # pos_label may be absent from the data (e.g. every label is 0): add it with zero counts
        if _label_index(pos, classes)[0] < 0:
            if classes.size == 2:
                raise ValueError(
                    f"pos_label {pos_label!r} is not one of the labels {classes.tolist()}"
                )

            new_classes = np.union1d(classes, pos)
            cm = _expand_confusion(cm, classes, new_classes)
            classes = new_classes

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp  # predicted as the class, truly another
    fn = cm.sum(axis=1) - tp  # truly the class, predicted as another
    support = cm.sum(axis=1)

    if average == "binary":
        idx = _label_index(pos, classes)[0]
        tp, fp, fn = tp[idx], fp[idx], fn[idx]

    elif average == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()

    scores = (
        _safe_divide(tp, tp + fp, zero_division),
        _safe_divide(tp, tp + fn, zero_division),
        _safe_divide(2 * tp, 2 * tp + fp + fn, zero_division),
    )

    if average is None:
        return scores

    if average == "macro":
        return tuple(float(s.mean()) for s in scores)

    if average == "weighted":
        return tuple(float(np.average(s, weights=support)) for s in scores)

    return tuple(float(s) for s in scores)  # binary / micro: already one value
