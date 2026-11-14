"""Descriptive statistics, quantiles, histogram and streaming (Welford) statistics.

axis follows NumPy: None reduces everything to one float, axis=0 on (n, d) gives (d,).
skip_nan=False lets NaN propagate like NumPy; skip_nan=True ignores NaN values.

``StreamingStats`` keeps per-feature mean, variance, min, max, quantiles and
histograms over chunks with ``update_stats``, in fixed memory, optionally over
a sliding window of recent chunks.
"""

from collections import deque

import numpy as np

from .utils import _check_int, check_array


def mean(x: np.ndarray, axis: int = None, skip_nan: bool = False):
    """Compute the mean along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the reduced array with
        the selected axis removed (possibly a zero-dimensional array scalar).

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).

    Notes
    -----
    Time: O(N), N=x.size. Space: O(N) in the worst case for conversion,
    NaN handling and the output; float inputs may avoid a conversion copy.
    All-NaN slices return NaN with NumPy's warning when skip_nan=True.
    """
    x = check_array(x)
    if skip_nan:
        result = np.nanmean(x, axis=axis)
    else:
        result = np.mean(x, axis=axis)

    if axis is None:
        return float(result)
    return result


def median(x: np.ndarray, axis: int = None, skip_nan: bool = False):
    """Compute the median along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the reduced array with
        the selected axis removed (possibly a zero-dimensional array scalar).

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).

    Notes
    -----
    Median has a conservative O(N * log(N)) time bound and O(N) space.
    All-NaN slices return NaN with NumPy's warning when skip_nan=True.
    """
    x = check_array(x)
    if skip_nan:
        result = np.nanmedian(x, axis=axis)
    else:
        result = np.median(x, axis=axis)

    if axis is None:
        return float(result)
    return result


def std(x: np.ndarray, axis: int = None, ddof: int = 0, skip_nan: bool = False):
    """Compute the standard deviation along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    ddof : int, default 0
        Divisor is the number of included values minus ddof.
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        Float for axis=None; otherwise the input shape with axis removed.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).
        Also raised when any reduced group has no more than ddof values.

    Notes
    -----
    Time and worst-case space: O(N), N=x.size, including conversion and
    centred/NaN temporary arrays.
    """
    x = check_array(x)
    if skip_nan:
        count = np.sum(~np.isnan(x), axis=axis)
    else:
        count = x.size if axis is None else x.shape[axis]
    if np.any(count <= ddof):
        raise ValueError(f"std needs more than ddof={ddof} values")

    if skip_nan:
        result = np.nanstd(x, axis=axis, ddof=ddof)
    else:
        result = np.std(x, axis=axis, ddof=ddof)

    if axis is None:
        return float(result)
    return result


def min(x: np.ndarray, axis: int = None, skip_nan: bool = False):
    """Compute the minimum along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the reduced array with
        the selected axis removed (possibly a zero-dimensional array scalar).

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).

    Notes
    -----
    Time: O(N), N=x.size. Space: O(N) in the worst case for conversion,
    NaN handling and the output; float inputs may avoid a conversion copy.
    All-NaN slices return NaN with NumPy's warning when skip_nan=True.
    """
    x = check_array(x)
    if skip_nan:
        result = np.nanmin(x, axis=axis)
    else:
        result = np.min(x, axis=axis)

    if axis is None:
        return float(result)
    return result


def max(x: np.ndarray, axis: int = None, skip_nan: bool = False):
    """Compute the maximum along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the reduced array with
        the selected axis removed (possibly a zero-dimensional array scalar).

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).

    Notes
    -----
    Time: O(N), N=x.size. Space: O(N) in the worst case for conversion,
    NaN handling and the output; float inputs may avoid a conversion copy.
    All-NaN slices return NaN with NumPy's warning when skip_nan=True.
    """
    x = check_array(x)
    if skip_nan:
        result = np.nanmax(x, axis=axis)
    else:
        result = np.max(x, axis=axis)

    if axis is None:
        return float(result)
    return result


def quantile(x: np.ndarray, q, axis: int = None, method: str = "linear", skip_nan: bool = False):
    """Compute quantiles along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Input data converted to float.
    q : float or 1-D array_like
        One or more probabilities in [0, 1].
    axis : int or None, default None
        Axis to reduce; None flattens the data.
    method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        Quantile interpolation method; default 'linear'.
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        Scalar q and axis=None return a float. Otherwise scalar q removes
        the reduced axis; a sequence prepends an axis of length len(q).

    Raises
    ------
    TypeError
        If x or q cannot be converted to numbers.
    ValueError
        If x/q is empty, q is invalid, method is unsupported or axis is invalid.

    Notes
    -----
    Conservative time: O(N * log(N) + output_size); space O(N + output_size),
    where N=x.size. NumPy partition-based selection can be faster.
    NaN handling follows NumPy; all-NaN slices may emit a warning.
    """
    x = check_array(x)
    q = np.asarray(q, dtype=float)
    if q.size == 0 or np.any(q < 0) or np.any(q > 1):
        raise ValueError("q must lie in [0, 1]")
    if method not in ("linear", "lower", "higher", "midpoint", "nearest"):
        raise ValueError(f"unknown method {method!r}")

    if skip_nan:
        result = np.nanquantile(x, q, axis=axis, method=method)
    else:
        result = np.quantile(x, q, axis=axis, method=method)

    if axis is None and q.ndim == 0:
        return float(result)
    return result


def histogram(x: np.ndarray, bins=10, range: tuple = None, density: bool = False):
    """Bin flattened data after discarding NaN values.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Input flattened to one dimension.
    bins : int or sequence of edges, default 10
        Number of equal-width bins or explicit increasing edges.
    range : pair of numbers or None, default None
        Bounds for automatically chosen edges; values outside are ignored.
    density : bool, default False
        If true, normalize counts so the density integral is one.

    Returns
    -------
    tuple of (ndarray, ndarray)
        Counts or densities of shape (b,) and bin edges of shape (b+1,).
        All bins except the last exclude the right endpoint.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data or bins has invalid type.
    ValueError
        If x is empty, all values are NaN, or bins/range is invalid.

    Notes
    -----
    Conservative time: O(N * log(N) + N * log(b) + b); space O(N + b),
    N=x.size. NumPy uses faster paths for equal-width bins.
    """
    x = check_array(x).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("histogram needs at least one non-NaN value")
    return np.histogram(x, bins=bins, range=range, density=density)


class RunningStats:
    """Accumulate per-feature streaming moments, minima and maxima by batches.

    Parameters
    ----------
    skip_nan : bool, default False
        Drop entire rows containing NaN when true. Otherwise NaN propagates
        into the affected feature statistics.

    Attributes
    ----------
    n : int
        Number of retained rows, initially zero.
    mean : ndarray, shape (d,)
        Running means, created after the first retained batch.
    m2 : ndarray, shape (d,)
        Running sums of squared deviations from the means.
    min, max : ndarray, shape (d,)
        Running minima and maxima.

    Raises
    ------
    ValueError
        If a batch has invalid dimensions or a different feature count, or
        variance/standard deviation is requested with n <= ddof.
    RuntimeError
        If variance/standard deviation is requested before any retained data.

    Notes
    -----
    Uses Chan/Welford moment merging. Updating b rows and d features takes
    O(b*d) time and temporary space; persistent state is O(d), independent
    of the total number of rows.
    """

    def __init__(self, skip_nan: bool = False):
        """Initialize an empty streaming accumulator.

        Parameters
        ----------
        skip_nan : bool, default False
            If true, discard each incoming row containing any NaN. Otherwise
            NaN propagates into the statistics for its feature.

        Returns
        -------
        None
            Set n=0; feature arrays are created by the first retained batch.

        Notes
        -----
        Time and space: O(1).
        """
        self.skip_nan = skip_nan
        self.n = 0

    def update(self, x: np.ndarray) -> "RunningStats":
        """Merge an incoming batch into the stored statistics.

        Parameters
        ----------
        x : numeric scalar or array_like, shape (b,) or (b, d)
            A scalar is one sample with one feature. A 1-D array is b samples of
            one feature, not one sample of b features. Later batches must use d
            features. With skip_nan=True, discard complete rows containing NaN.

        Returns
        -------
        RunningStats
            The updated instance (self). An empty or fully discarded batch does
            not change state. n counts retained rows; feature statistics are (d,).

        Raises
        ------
        TypeError
            If x cannot be converted to a numeric array.
        ValueError
            If conversion fails, x has more than two dimensions, or a nonempty
            batch has a different feature count.

        Notes
        -----
        Time: O(b * d). Temporary space: O(b * d); persistent state O(d).
        Combines batch moments with stored moments using the Chan/Welford formula.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim < 2:
            x = x.reshape(-1, 1)
        if x.ndim > 2:
            raise ValueError(f"batch must be at most 2-D, got shape {x.shape}")
        if self.skip_nan:
            x = x[~np.isnan(x).any(axis=1)]

        b, d = x.shape
        if b == 0:
            return self
        if self.n == 0:
            self.mean = np.zeros(d)
            self.m2 = np.zeros(d)
            self.min = np.full(d, np.inf)
            self.max = np.full(d, -np.inf)
        if d != self.mean.shape[0]:
            raise ValueError(f"expected {self.mean.shape[0]} features, got {d}")

        # merge the stored statistics with this batch (Chan et al. formula)
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - self.mean
        n_new = self.n + b

        self.mean = self.mean + delta * b / n_new
        self.m2 = self.m2 + batch_m2 + delta ** 2 * self.n * b / n_new
        self.min = np.minimum(self.min, x.min(axis=0))
        self.max = np.maximum(self.max, x.max(axis=0))
        self.n = n_new
        return self

    def var(self, ddof: int = 0) -> np.ndarray:
        """Return the accumulated per-feature variance.

        Parameters
        ----------
        ddof : int, default 0
            Degrees-of-freedom correction: denominator is n - ddof.
            Use 0 for population variance or 1 for a sample estimate.

        Returns
        -------
        ndarray, shape (d,)
            One value per feature over retained rows.

        Raises
        ------
        RuntimeError
            If no samples have been accumulated.
        ValueError
            If n <= ddof.

        Notes
        -----
        Time and output space: O(d).
        """
        if self.n == 0:
            raise RuntimeError("no data yet; call update() first")
        if self.n <= ddof:
            raise ValueError(f"var needs more than ddof={ddof} rows, seen {self.n}")
        return self.m2 / (self.n - ddof)

    def std(self, ddof: int = 0) -> np.ndarray:
        """Return the accumulated per-feature standard deviation.

        Parameters
        ----------
        ddof : int, default 0
            Degrees-of-freedom correction: denominator is n - ddof.
            Use 0 for population variance or 1 for a sample estimate.

        Returns
        -------
        ndarray, shape (d,)
            One value per feature over retained rows.

        Raises
        ------
        RuntimeError
            If no samples have been accumulated.
        ValueError
            If n <= ddof.

        Notes
        -----
        Time and output space: O(d).
        """
        return np.sqrt(self.var(ddof))


class StreamingStats:
    """Per-feature statistics of a stream of chunks, in fixed memory.

    Mean, variance, min and max are exact. Quantiles and histograms come from
    an adaptive histogram per feature. NaN values are ignored cell by cell.

    Parameters
    ----------
    n_bins : int, default 256
        Histogram bins per feature (>= 1).
    window : int or None, default None
        Keep only the most recent ``window`` chunks; None keeps the whole stream.

    Attributes
    ----------
    n_ : int
        Rows counted (inside the window, if any).
    count_ : ndarray of int, shape (d,)
        Non-NaN values counted per feature.
    n_features_in_ : int
        Number of features, fixed by the first chunk.

    Raises
    ------
    TypeError
        If ``n_bins`` or ``window`` is not an integer.
    ValueError
        If ``n_bins`` or ``window`` < 1.

    Notes
    -----
    Moments are merged with the Chan/Welford formula, so values far from zero
    keep their precision. Each feature's histogram starts on the range of its
    first values; when data falls outside, the bin width doubles and
    neighbouring bins merge, so no count is lost. Quantiles interpolate inside
    a bin and are clipped to the exact min and max: error <= one bin width.
    Update: O(n * d) time. Memory: O(d * n_bins), or O(window * d * n_bins).

    Examples
    --------
    >>> s = StreamingStats().update_stats([[1.0], [2.0]]).update_stats([[3.0], [4.0]])
    >>> s.mean(), s.var()
    (array([2.5]), array([1.25]))
    """

    def __init__(self, n_bins: int = 256, window: int | None = None):
        _check_int(n_bins, "n_bins", minimum=1)

        if window is not None:
            _check_int(window, "window", minimum=1)

        self.n_bins = n_bins
        self.window = window
        self.reset()

    def reset(self) -> "StreamingStats":
        """Forget every chunk seen so far.

        Returns
        -------
        StreamingStats
            The emptied instance.
        """
        self.n_ = 0
        self._chunks = deque()
        self._state = None          # (count, mean, m2, min, max, hist) totals
        self._lo = self._width = None

        for name in ("count_", "n_features_in_"):
            self.__dict__.pop(name, None)

        return self

    def update_stats(self, X_chunk) -> "StreamingStats":
        """Add one chunk to the statistics.

        Parameters
        ----------
        X_chunk : array_like, shape (n_samples, n_features)
            Numeric data; NaN marks a missing value.

        Returns
        -------
        StreamingStats
            The updated instance (self).

        Raises
        ------
        TypeError
            If the chunk is not numeric.
        ValueError
            If the chunk is not 2-D, contains infinity, or has a different
            number of features than the first chunk.

        Notes
        -----
        An empty chunk (0 rows) is ignored; a rejected chunk changes nothing.
        Time: O(n * d), plus O(d * n_bins) when a histogram widens.
        """
        if np.ndim(X_chunk) == 2 and np.shape(X_chunk)[0] == 0:
            return self

        X = check_array(X_chunk, ndim=2)

        if np.isinf(X).any():
            raise ValueError("X_chunk contains infinity; only NaN is accepted as missing")

        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got shape {X.shape}")

        # All checks passed: only now change the state
        self.n_features_in_ = X.shape[1]
        self._fit_range(X)
        chunk = (*_chunk_moments(X), np.fmin.reduce(X, axis=0), np.fmax.reduce(X, axis=0),
                 self._chunk_hist(X), X.shape[0])

        if self.window is None:
            self._state = chunk[:6] if self._state is None else _merge_chunk(self._state, chunk[:6])
            self.n_ += X.shape[0]
        else:
            self._chunks.append(chunk)

            if len(self._chunks) > self.window:                # drop the oldest chunk
                self._chunks.popleft()

            self._state = self._chunks[0][:6]
            for c in list(self._chunks)[1:]:
                self._state = _merge_chunk(self._state, c[:6])
            self.n_ = sum(c[6] for c in self._chunks)

        self.count_ = self._state[0]
        return self

    def mean(self) -> np.ndarray:
        """Per-feature mean of the non-NaN values.

        Returns
        -------
        ndarray, shape (d,)
            NaN for a feature with no values yet.

        Raises
        ------
        RuntimeError
            If no data has been added.
        """
        count, mean = self._check_data()[:2]
        return np.where(count > 0, mean, np.nan)

    def var(self, ddof: int = 0) -> np.ndarray:
        """Per-feature variance of the non-NaN values.

        Parameters
        ----------
        ddof : int, default 0
            Delta degrees of freedom: the divisor is ``count - ddof``.

        Returns
        -------
        ndarray, shape (d,)
            NaN where ``count <= ddof``.

        Raises
        ------
        RuntimeError
            If no data has been added.
        """
        count, _, m2 = self._check_data()[:3]
        return np.divide(m2, count - ddof, out=np.full(m2.shape, np.nan), where=count > ddof)

    def std(self, ddof: int = 0) -> np.ndarray:
        """Per-feature standard deviation of the non-NaN values.

        Parameters
        ----------
        ddof : int, default 0
            Delta degrees of freedom: the divisor is ``count - ddof``.

        Returns
        -------
        ndarray, shape (d,)
            NaN where ``count <= ddof``.

        Raises
        ------
        RuntimeError
            If no data has been added.
        """
        return np.sqrt(self.var(ddof))

    def min(self) -> np.ndarray:
        """Per-feature minimum, ignoring NaN.

        Returns
        -------
        ndarray, shape (d,)
            NaN for a feature with no values yet.

        Raises
        ------
        RuntimeError
            If no data has been added.
        """
        return self._check_data()[3].copy()

    def max(self) -> np.ndarray:
        """Per-feature maximum, ignoring NaN.

        Returns
        -------
        ndarray, shape (d,)
            NaN for a feature with no values yet.

        Raises
        ------
        RuntimeError
            If no data has been added.
        """
        return self._check_data()[4].copy()

    def quantile(self, q) -> np.ndarray:
        """Approximate per-feature quantiles from the histograms.

        Parameters
        ----------
        q : float or 1-D array_like
            Probabilities in [0, 1].

        Returns
        -------
        ndarray
            Shape (d,) for a scalar ``q``, else (len(q), d). NaN for a feature
            with no values yet.

        Raises
        ------
        RuntimeError
            If no data has been added.
        ValueError
            If ``q`` is empty or outside [0, 1].

        Notes
        -----
        Error is at most one bin width (see ``histogram``). Time: O(len(q) * d * n_bins).
        """
        count, _, _, lo_val, hi_val, hist = self._check_data()
        q = np.asarray(q, dtype=float)

        if q.size == 0 or np.any(q < 0) or np.any(q > 1):
            raise ValueError("q must lie in [0, 1]")

        qs = np.atleast_1d(q)[:, None]                          # (m, 1)
        cdf = np.cumsum(hist, axis=1)                           # (d, B)
        target = qs * count                                     # (m, d)

        # First bin whose cumulative count reaches the target, then interpolate inside it
        j = np.minimum((cdf[None] < target[..., None]).sum(axis=-1), self.n_bins - 1)
        before = np.where(j > 0, np.take_along_axis(cdf, np.maximum(j - 1, 0).T, axis=1).T, 0)
        in_bin = np.take_along_axis(hist, j.T, axis=1).T
        frac = np.divide(target - before, in_bin, out=np.zeros(target.shape), where=in_bin > 0)
        values = np.clip(self._lo + (j + frac) * self._width, lo_val, hi_val)
        values = np.where(count > 0, values, np.nan)

        return values[0] if q.ndim == 0 else values

    def histogram(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature histogram counts and bin edges.

        Returns
        -------
        counts : ndarray of int, shape (d, n_bins)
            Values per bin.
        edges : ndarray, shape (d, n_bins + 1)
            Bin edges; the last bin includes its right edge.

        Raises
        ------
        RuntimeError
            If no data has been added.
        """
        hist = self._check_data()[5]
        edges = self._lo[:, None] + self._width[:, None] * np.arange(self.n_bins + 1)
        return hist.copy(), edges

    def _check_data(self) -> tuple:
        """Return the state totals, or raise RuntimeError before any data."""
        if self._state is None:
            raise RuntimeError("no data yet; call update_stats(X_chunk) first")
        return self._state

    def _fit_range(self, X: np.ndarray) -> None:
        """Start or widen each feature's histogram range so it covers the chunk."""
        d, B = X.shape[1], self.n_bins
        x_min, x_max = np.fmin.reduce(X, axis=0), np.fmax.reduce(X, axis=0)

        if self._lo is None:
            self._lo = np.full(d, np.nan)
            self._width = np.full(d, np.nan)

        # A feature's first values set its range; a constant start gets a unit-wide range
        new = np.isnan(self._lo) & ~np.isnan(x_min)
        span = x_max - x_min
        self._width[new] = np.where(span[new] > 0, span[new] / B, 1.0 / B)
        self._lo[new] = np.where(span[new] > 0, x_min[new], x_min[new] - 0.5)

        # Features whose new values fall outside the current range (a short loop: rare event)
        hi = self._lo + B * self._width
        for f in np.flatnonzero((x_min < self._lo) | (x_max > hi)):
            self._widen(f, np.fmin(x_min[f], self._lo[f]), np.fmax(x_max[f], hi[f]))

    def _widen(self, f: int, low: float, high: float) -> None:
        """Double feature f's bin width until [low, high] fits, merging old bins."""
        B, lo, w = self.n_bins, self._lo[f], self._width[f]
        t = np.ceil((lo - low) / w)                  # old-width bins added on the left
        new_lo = lo - t * w
        k = int(np.maximum(1, np.ceil(np.log2((high - new_lo) / (B * w)))))

        while new_lo + B * w * 2.0 ** k < high:     # guard against rounding in log2
            k += 1

        # Old bin i lies wholly inside new bin (t + i) // 2**k
        target = np.minimum(np.floor((t + np.arange(B)) / 2.0 ** k), B - 1).astype(np.int64)

        def rebin(row):
            return np.bincount(target, weights=row, minlength=B).astype(np.int64)

        # The window's totals are rebuilt from its chunks after each update, so rebin one or the other
        if self.window is None:
            if self._state is not None:
                self._state[5][f] = rebin(self._state[5][f])
        else:
            for chunk in self._chunks:
                chunk[5][f] = rebin(chunk[5][f])

        self._lo[f], self._width[f] = new_lo, w * 2.0 ** k

    def _chunk_hist(self, X: np.ndarray) -> np.ndarray:
        """Histogram counts of one chunk on the current edges, shape (d, n_bins)."""
        d, B = X.shape[1], self.n_bins
        valid = ~np.isnan(X)
        idx = np.clip(np.floor((X - self._lo) / self._width), 0, B - 1)     # top edge -> last bin
        flat = (idx + np.arange(d) * B)[valid].astype(np.int64)
        return np.bincount(flat, minlength=d * B).reshape(d, B)


def _chunk_moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature non-NaN count, mean and sum of squared deviations of a 2-D chunk."""
    valid = ~np.isnan(X)
    count = valid.sum(axis=0)
    mean = np.divide(np.where(valid, X, 0.0).sum(axis=0), count,
                     out=np.zeros(X.shape[1]), where=count > 0)
    m2 = (np.where(valid, X - mean, 0.0) ** 2).sum(axis=0)
    return count, mean, m2


def _merge_moments(n_a, mean_a, m2_a, n_b, mean_b, m2_b) -> tuple:
    """Combine two per-feature (count, mean, m2) triples with the Chan/Welford formula."""
    n = n_a + n_b
    safe = np.where(n > 0, n, 1)
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / safe
    m2 = m2_a + m2_b + delta ** 2 * n_a * n_b / safe
    return n, mean, m2


def _merge_chunk(a: tuple, b: tuple) -> tuple:
    """Merge two (count, mean, m2, min, max, hist) states."""
    return (*_merge_moments(*a[:3], *b[:3]), np.fmin(a[3], b[3]), np.fmax(a[4], b[4]), a[5] + b[5])


if __name__ == "__main__":
    from .utils import iter_batches

    rng = np.random.default_rng(0)
    X = rng.normal(1e6, 1, (1000, 3))
    X[5, 1] = np.nan

    assert np.isnan(mean(X, axis=0)[1])
    assert np.allclose(mean(X, axis=0, skip_nan=True), np.nanmean(X, axis=0))
    assert isinstance(mean(X[:, 0]), float)
    assert quantile(X, [0.25, 0.5], axis=0, skip_nan=True).shape == (2, 3)
    assert histogram(X[:, 1], bins=5)[0].sum() == 999

    running = RunningStats(skip_nan=True)
    for batch in iter_batches(X, 64):
        running.update(batch)
    clean = X[~np.isnan(X).any(axis=1)]
    assert running.n == 999
    assert np.allclose(running.mean, clean.mean(axis=0))
    assert np.allclose(running.var(), clean.var(axis=0))

    print("stats ok")
