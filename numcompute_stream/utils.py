"""Shared helpers: input checks, batching, stable activations and distances."""
from collections.abc import Iterable, Iterator
from itertools import islice

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "iter_line_batches",
    "check_array",
    "iter_batches",
    "logsumexp",
    "softmax",
    "sigmoid",
    "pairwise_distances",
]


def _as_numeric_array(values: ArrayLike) -> np.ndarray:
    """Validate a one-dimensional real numeric sequence without changing its dtype.

    Parameters
    ----------
    values : array_like, shape (n,)
        Real numeric values; empty input and NaNs are allowed.

    Returns
    -------
    ndarray, shape (n,)
        Input array or an array converted from the sequence.

    Raises
    ------
    TypeError
        If values are nonnumeric or complex.
    ValueError
        If values are not one-dimensional or cannot form a regular array.

    Notes
    -----
    Time/space: O(n) when conversion allocates, otherwise O(1).
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional sequence")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError("values must contain real numeric values")
    return array

# ---- Jose: iterating line batches ----
def iter_line_batches(items: Iterable, batch_size: int) -> Iterator[list]:
    """Yield consecutive batches of at most ``batch_size`` items.

    Works on any iterable, including ones that can only be read once
    (open files, generators), without loading the whole input into memory.

    Parameters
    ----------
    items : iterable
        Source of items, consumed lazily.
    batch_size : int
        Maximum number of items per batch. Must be a positive integer.

    Returns
    -------
    Iterator[list]
        Lists of length ``batch_size``; the last one may be shorter.
        Nothing is yielded for an empty input.

    Raises
    ------
    TypeError
        If ``batch_size`` is not an integer.
    ValueError
        If ``batch_size < 1``.

    Notes
    -----
    Time Complexity: O(n: number of items), over all batches.
    Space Complexity: O(b: batch_size), only one batch is held at a time.

    Examples
    --------
    >>> list(iter_line_batches(range(5), 2))
    [[0, 1], [2, 3], [4]]
    """
    if not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an int, got {type(batch_size).__name__}")

    if batch_size < 1:
        raise ValueError(f"batch_size must be greater or equal than 1, got {batch_size}")

    it = iter(items)                          # one iterator: each islice continues from the last
    batch = list(islice(it, batch_size))
    while batch:
        yield batch
        batch = list(islice(it, batch_size))


# ---- Dat: validation and batching ----

def check_array(X, ndim: int = None) -> np.ndarray:
    """Convert input to a nonempty floating-point array and validate its dimensions.

    Parameters
    ----------
    X : array_like
        Values convertible to float; NaN and infinity are not rejected.
    ndim : int or None, default None
        Required number of dimensions; None allows any nonempty shape.

    Returns
    -------
    ndarray of float64, same shape as X
        May share storage with an existing float64 array.

    Raises
    ------
    TypeError
        If X cannot be converted to floating-point data.
    ValueError
        If X is empty or has a different number of dimensions.

    Notes
    -----
    Time/space: O(N) if conversion allocates, otherwise O(1), N=X.size.
    """
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("X must be numeric")

    if X.size == 0:
        raise ValueError("X is empty")
    if ndim is not None and X.ndim != ndim:
        raise ValueError(f"X must be {ndim}-D, got shape {X.shape}; use X.reshape(-1, 1) for one feature")
    return X


def _check_int(value, name: str, minimum: int) -> None:
    """Raise if ``value`` is not an integer >= ``minimum`` (bool rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")

    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")


def iter_batches(X: np.ndarray, batch_size: int):
    """Yield consecutive row slices, including a shorter final batch.

    Parameters
    ----------
    X : array_like, shape (n, ...)
        Input with a row axis; converted with np.asarray.
    batch_size : int
        Positive maximum row count per batch.

    Returns
    -------
    iterator of ndarray, shape (b, ...)
        Views of the converted array, with b <= batch_size. Empty input
        with a row axis yields no batches.

    Raises
    ------
    ValueError
        If batch_size is not positive.
    TypeError
        If batch_size is not usable as a range step.
    IndexError
        If X is scalar and has no row axis.

    Notes
    -----
    Time: O(ceil(n / batch_size)) iterations after conversion. Extra space
    is O(1) for array input; converting a sequence may allocate its full size.
    Validation occurs when the iterator is consumed.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    X = np.asarray(X)
    for start in range(0, X.shape[0], batch_size):
        yield X[start:start + batch_size]


# ---- Jose: numerically stable activations and distances ----

_EXP_FLOOR = -708.0             # exp() below this underflows past the smallest normal float
_BLOCK_FLOATS = 1_000_000       # floats per block of differences in pairwise_distances (8 MB)


def _exp_floored(z: np.ndarray) -> np.ndarray:
    """exp(z), with 0 where z is below _EXP_FLOOR, so exp never underflows or overflows for z <= 0."""
    return np.where(z > _EXP_FLOOR, np.exp(np.maximum(z, _EXP_FLOOR)), 0.0)


def _shifted_exp(x: np.ndarray, axis):
    """Per-slice peak, a mask of finite peaks, and exp(x - peak) (all zeros where the peak is not finite).

    The peak is the slice maximum: +inf if the slice holds +inf, -inf if it
    is all -inf, NaN if it holds NaN. Only slices with a finite peak are
    exponentiated, so exp never sees +inf or a value above 0.
    """
    peak = np.max(x, axis=axis, keepdims=True)
    finite = np.isfinite(peak)
    z = np.where(finite, x - np.where(finite, peak, 0.0), -np.inf)
    return peak, finite, _exp_floored(z)


def logsumexp(x: ArrayLike, axis: int = None):
    """Compute log(sum(exp(x))) without overflow or underflow.

    Uses the max-shift identity log(sum(exp(x))) = m + log(sum(exp(x - m))),
    with m = max(x), so the largest term is exp(0) = 1. A slice whose
    maximum is not finite (it holds +inf or NaN, or is all -inf) gets its
    result directly and is never exponentiated, so no step overflows, even
    under ``np.errstate(all="raise")``.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Values converted to float, e.g. log-probabilities or logits.
    axis : int or None, default None
        Axis to reduce; None reduces all values.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the input shape with the
        axis removed. A slice of only -inf gives -inf, +inf gives +inf and
        NaN propagates.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N) for the shifted temporary array.

    Examples
    --------
    >>> logsumexp([1000.0, 1000.0])          # the naive formula overflows to inf
    1000.6931471805599
    """
    x = check_array(x)
    peak, finite, exp = _shifted_exp(x, axis)
    total = np.where(finite, np.sum(exp, axis=axis, keepdims=True), 1.0)    # >= 1 where finite
    out = np.where(finite, np.log(total) + np.where(finite, peak, 0.0), peak)

    if axis is None:
        return float(out.reshape(()))
    return np.squeeze(out, axis=axis)


def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Convert scores to probabilities that sum to 1 along an axis.

    Subtracts the maximum of each slice before exponentiating (max-shifted
    softmax), so large scores do not overflow. The result does not change
    when a constant is added to every score in a slice. Slices holding +inf
    are resolved before any exponential, so no step overflows, even under
    ``np.errstate(all="raise")``.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Scores (logits) converted to float; for (n, k) input each row is one
        sample's k class scores.
    axis : int, default -1
        Axis along which the probabilities sum to 1 (the last axis by default).

    Returns
    -------
    ndarray of float, same shape as x
        Values in [0, 1]. In a slice containing +inf, the +inf entries share
        the probability equally and the rest get 0 (the limit of large
        scores). A slice containing NaN, or made only of -inf, gives NaN.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N) for the exponentials and the output.

    Examples
    --------
    >>> softmax([1000.0, 1000.0])            # naive exp(1000) overflows
    array([0.5, 0.5])
    """
    x = check_array(x)
    peak, finite, exp = _shifted_exp(x, axis)
    p = exp / np.where(finite, np.sum(exp, axis=axis, keepdims=True), 1.0)  # sum >= 1 where finite

    # +inf slices: the +inf entries share the mass; NaN or all -inf slices: undefined
    top = x == np.inf
    shared = top / np.maximum(np.sum(top, axis=axis, keepdims=True), 1)
    p = np.where(peak == np.inf, shared, p)
    return np.where(np.isnan(peak) | (peak == -np.inf), np.nan, p)


def sigmoid(x: ArrayLike):
    """Compute the logistic function 1 / (1 + exp(-x)) without overflow.

    Uses z = exp(-|x|), which never exceeds 1: x >= 0 gives 1 / (1 + z) and
    x < 0 gives z / (1 + z), the same value written in a safe form. z is
    set to 0 below exp's underflow point, so no step overflows or underflows,
    even under ``np.errstate(all="raise")``.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Values converted to float.

    Returns
    -------
    float or ndarray of float
        A Python float for scalar input; otherwise the same shape as x.
        Values are in [0, 1]; NaN propagates.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N) for temporaries and the output.

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> float(sigmoid([-1000.0])[0])          # naive exp(1000) overflows
    0.0
    """
    x = check_array(x)
    z = _exp_floored(-np.abs(x))
    out = np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
    out = np.where(np.isnan(x), np.nan, out)          # the floor would turn NaN into 0

    if out.ndim == 0:
        return float(out)
    return out


def pairwise_distances(X: ArrayLike, Y: ArrayLike = None) -> np.ndarray:
    """Euclidean distance between every row of X and every row of Y.

    Works on the direct differences x - y, so close points stay accurate
    however widely spread the data is (the expansion ||x||^2 + ||y||^2 - 2 x.y
    cancels catastrophically when one point is far away). Each difference
    vector is divided by its largest absolute entry before squaring, as in
    ``hypot``, so large values do not overflow. Rows of X are processed in
    vectorised blocks of about 8 MB of differences each.

    Parameters
    ----------
    X : numeric array_like, shape (n, d)
        First set of points, one per row.
    Y : numeric array_like, shape (m, d), or None, default None
        Second set of points; None compares X with itself.

    Returns
    -------
    ndarray of float, shape (n, m)
        Entry [i, j] is the distance between X[i] and Y[j]. With Y=None the
        result is (n, n), symmetric, with an exact zero diagonal. A pair with
        an infinite difference gives inf; NaN propagates.

    Raises
    ------
    TypeError
        If X or Y cannot be converted to numeric data.
    ValueError
        If X or Y is empty or not 2-D, or they have different feature counts.

    Notes
    -----
    Time Complexity: O(n * m * d).
    Space Complexity: O(n * m) for the output plus O(b * m * d) for one block
    of b rows, with b * m * d about 1e6.

    Examples
    --------
    >>> pairwise_distances([[0.0, 0.0], [3.0, 4.0]])
    array([[0., 5.],
           [5., 0.]])
    """
    X = check_array(X, ndim=2)
    same = Y is None
    Y = X if same else check_array(Y, ndim=2)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"X and Y need the same number of features, got {X.shape[1]} and {Y.shape[1]}")

    out = np.empty((X.shape[0], Y.shape[0]))
    rows = max(1, _BLOCK_FLOATS // (Y.shape[0] * Y.shape[1]))

    for start in range(0, X.shape[0], rows):
        diff = np.abs(X[start:start + rows, None, :] - Y[None, :, :])        # (b, m, d)
        scale = diff.max(axis=2)                                             # (b, m)
        safe = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)        # 0 and inf: no division
        out[start:start + rows] = scale * np.sqrt(((diff / safe[..., None]) ** 2).sum(axis=2))

    if same:
        np.fill_diagonal(out, 0.0)
    return out


if __name__ == "__main__":
    assert check_array([[1, 2]]).dtype == float

    sizes = [batch.shape[0] for batch in iter_batches(np.zeros((10, 1)), 4)]
    assert sizes == [4, 4, 2]

    try:
        check_array([1.0], ndim=2)
        assert False, "1-D input should be rejected"
    except ValueError:
        pass

    print("utils ok")
