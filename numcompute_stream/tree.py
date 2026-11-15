"""Decision tree classifier built from scratch, with Gini or entropy splits.

Each internal node asks ``X[:, feature] <= threshold``; rows that answer yes go
left. The split search is vectorised: each feature is sorted once and a
cumulative sum of one-hot labels gives the class counts on both sides of every
threshold, so all candidate splits are scored without a Python loop.

For streams, ``partial_fit`` buffers incoming chunks and retrains the tree
once the buffer holds at least ``buffer_size`` rows.

Shapes: ``X`` is (n_samples, n_features); ``y`` is (n_samples,).
"""

import sys
import time

import numpy as np

from .metrics import _check_labels, _check_same_family, accuracy
from .utils import _check_int, check_array

# All that can be imported from this module
__all__ = [
    "gini",
    "entropy",
    "Node",
    "DecisionTreeClassifier",
]

_CRITERIA = ("gini", "entropy")
_MAX_FEATURES_STR = ("sqrt", "log2")

# Gains closer than this count as equal, so float rounding cannot decide a tie
_GAIN_TOL = 1e-12


# ---------------------------------------------------------------------------
# Impurity
# ---------------------------------------------------------------------------

def gini(y) -> float:
    """Gini impurity of a group of labels.

    Parameters
    ----------
    y : array_like, shape (n,)
        Labels (int, float, bool or str). May be empty.

    Returns
    -------
    float
        ``1 - sum(p**2)`` over the class proportions ``p``: 0 for a pure group,
        0.5 for two equally sized classes, 0 for an empty group.

    Raises
    ------
    TypeError
        If the label dtype is unsupported.
    ValueError
        If ``y`` is not 1-D or contains NaN.

    Notes
    -----
    Time: O(n log n).
    Space: O(n).

    Examples
    --------
    >>> gini([0, 1, 1, 0])
    0.5
    """
    counts = _class_counts(y)
    return float(_gini_from_counts(counts)) if counts.size else 0.0


def entropy(y) -> float:
    """Entropy in bits of a group of labels.

    Parameters
    ----------
    y : array_like, shape (n,)
        Labels (int, float, bool or str). May be empty.

    Returns
    -------
    float
        ``-sum(p * log2(p))`` over the nonzero class proportions ``p``: 0 for a
        pure group, 1 for two equally sized classes, 0 for an empty group.

    Raises
    ------
    TypeError
        If the label dtype is unsupported.
    ValueError
        If ``y`` is not 1-D or contains NaN.

    Notes
    -----
    Time: O(n log n).
    Space: O(n).

    Examples
    --------
    >>> entropy([0, 1, 1, 0])
    1.0
    """
    counts = _class_counts(y)
    return float(_entropy_from_counts(counts)) if counts.size else 0.0


def _class_counts(y) -> np.ndarray:
    """Validated count of each distinct label; empty input gives an empty array."""
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}; use .ravel() for a column vector")

    if y.size == 0:
        return np.zeros(0, dtype=np.int64)

    y = _check_labels(y, "y")

    return np.unique(y, return_counts=True)[1]


def _gini_from_counts(counts) -> np.ndarray:
    """Gini impurity from class counts of shape (..., k); 0 for groups with no rows."""
    counts = np.asarray(counts, dtype=float)

    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)   # no division by zero

    return np.where(n[..., 0] > 0, 1.0 - (p ** 2).sum(axis=-1), 0.0)


def _entropy_from_counts(counts) -> np.ndarray:
    """Entropy in bits from class counts of shape (..., k), with 0 * log2(0) = 0."""
    counts = np.asarray(counts, dtype=float)

    n = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)

    log_p = np.zeros_like(p)
    np.log2(p, out=log_p, where=p > 0)          # log2(0) is undefined: leave 0

    return 0.0 - (p * log_p).sum(axis=-1)       # 0.0 - x turns -0.0 into 0.0


_IMPURITY_FROM_COUNTS = {"gini": _gini_from_counts, "entropy": _entropy_from_counts}


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

class Node:
    """One node of a fitted tree: an internal split or a leaf.

    Parameters
    ----------
    value : ndarray of int, shape (n_classes,)
        Class counts of the training rows that reached the node.
    feature : int or None, default None
        Feature tested by the split ``X[:, feature] <= threshold``; None for a leaf.
    threshold : float or None, default None
        Split point; None for a leaf.
    left, right : Node or None, default None
        Children for rows with ``X[:, feature] <= threshold`` and the rest.

    Attributes
    ----------
    label : int or None
        On a leaf, the index of the majority class (ties go to the lowest
        index); None on an internal node.
    """

    __slots__ = ("feature", "threshold", "left", "right", "value", "label")

    def __init__(self, value, feature=None, threshold=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.label = int(np.argmax(value)) if feature is None else None

    @property
    def is_leaf(self) -> bool:
        """True if the node has no split."""
        return self.feature is None


class DecisionTreeClassifier:
    """Depth-limited classification tree with Gini or entropy splits.

    Parameters
    ----------
    max_depth : int, default 5
        Maximum depth of the tree (>= 1).
    buffer_size : int, default 500
        Rows ``partial_fit`` collects before retraining (>= 1).
    min_samples_split : int, default 2
        Nodes with fewer rows become leaves (>= 2).
    max_features : None, int, float, "sqrt" or "log2", default None
        Features tried at each node: all (None), a count (int), a fraction in
        (0, 1] (float), or sqrt/log2 of the feature count. Drawn without
        replacement from the features that are not constant in the node.
    criterion : {"gini", "entropy"}, default "gini"
        Impurity used for the information gain.
    random_state : int or None, default None
        Seed for drawing features when ``max_features`` is set.

    Attributes
    ----------
    root_ : Node
        Root of the fitted tree.
    classes_ : ndarray, shape (n_classes,)
        Sorted class labels of the current tree. With ``partial_fit``, every
        label seen in the stream up to the last retrain.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    max_features_ : int
        Number of features tried at each node.
    depth_ : int
        Depth of the fitted tree (0 for a single leaf).
    n_leaves_ : int
        Number of leaves.
    n_nodes_ : int
        Number of nodes, leaves included.
    fit_time_ : float
        Seconds taken by the last training.
    n_buffered_ : int
        Rows waiting in the buffer.
    n_retrains_ : int
        Retrains triggered by ``partial_fit`` or ``flush`` since the last ``fit``.

    Raises
    ------
    TypeError
        If a hyperparameter has the wrong type.
    ValueError
        If a hyperparameter is out of range.

    Notes
    -----
    Splits maximise information gain. Ties go to the lowest feature index,
    then the lowest threshold; thresholds are midpoints between consecutive
    distinct values. Fit: O(depth * d * n log n) time and O(n * d * k) extra
    space for n rows, d features and k classes. Predict: O(n * depth).

    Streaming: each ``partial_fit`` call appends its chunk to a buffer. Once
    the buffer holds at least ``buffer_size`` rows, the tree is retrained on
    all buffered rows and the buffer is cleared; between retrains, ``predict``
    uses the last tree. Buffer memory: O(buffer_size * d).

    Examples
    --------
    >>> X = [[1.0], [2.0], [10.0], [11.0]]
    >>> tree = DecisionTreeClassifier(max_depth=2).fit(X, ["a", "a", "b", "b"])
    >>> tree.predict([[3.0], [9.0]]).tolist()
    ['a', 'b']
    """

    def __init__(
        self,
        max_depth: int = 5,
        buffer_size: int = 500,
        min_samples_split: int = 2,
        max_features=None,
        criterion: str = "gini",
        random_state: int | None = None,
    ):
        # Validate hyperparameters before storing them
        _check_int(max_depth, "max_depth", minimum=1)
        _check_int(buffer_size, "buffer_size", minimum=1)
        _check_int(min_samples_split, "min_samples_split", minimum=2)
        _check_max_features(max_features)

        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")

        self.max_depth = max_depth
        self.buffer_size = buffer_size
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self._reset_stream()

    # -- training ------------------------------------------------------------

    def fit(self, X, y) -> "DecisionTreeClassifier":
        """Train the tree from scratch on the given data and clear the stream buffer.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Finite numeric features.
        y : array_like, shape (n_samples,)
            Class labels (int, float, bool or str).

        Returns
        -------
        DecisionTreeClassifier
            The fitted instance (self).

        Raises
        ------
        TypeError
            If X is not numeric or y has an unsupported dtype.
        ValueError
            If X is not 2-D, is empty, contains NaN or infinity, or has a
            different number of rows than y.

        Notes
        -----
        Time: O(depth * d * n log n).
        Extra space: O(n * d * k).
        """
        X, y = _check_Xy(X, y)

        self._reset_stream()
        self._train(X, y, np.unique(y), np.random.default_rng(self.random_state))
        self._seen_classes = self.classes_

        return self

    def partial_fit(self, X, y) -> "DecisionTreeClassifier":
        """Add a chunk to the buffer; retrain once it holds at least ``buffer_size`` rows.

        The retrain uses every buffered row, then clears the buffer. Between
        retrains, ``predict`` keeps using the last tree. An empty chunk
        (0 rows) is ignored.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Finite numeric features of the chunk.
        y : array_like, shape (n_samples,)
            Class labels of the chunk.

        Returns
        -------
        DecisionTreeClassifier
            The instance (self).

        Raises
        ------
        TypeError
            If X is not numeric, y has an unsupported dtype, or the labels
            change type (e.g. int to str) during the stream.
        ValueError
            If X is not 2-D, contains NaN or infinity, has a different number
            of rows than y, or a different number of features than earlier data.

        Notes
        -----
        A rejected chunk leaves the buffer unchanged. Chunks are copied, so
        later changes to the caller's arrays do not affect the buffer.
        Time: O(n) per call, plus a retrain when the buffer fills.
        """
        if np.ndim(X) in (1, 2) and np.shape(X)[0] == 0 and np.size(y) == 0:
            return self

        X, y = _check_Xy(X, y)
        expected = self._expected_n_features()

        if expected is not None and X.shape[1] != expected:
            raise ValueError(f"expected {expected} features, got shape {X.shape}")

        if self._seen_classes is not None:
            _check_same_family(self._seen_classes, y, ("earlier labels", "y"))

        # All checks passed: only now change the stream state
        if self._seen_classes is None:
            self._seen_classes = np.unique(y)

        else:
            self._seen_classes = np.union1d(self._seen_classes, y)    # classes are never dropped

        self._buffer_X.append(X.copy())
        self._buffer_y.append(y.copy())
        self.n_buffered_ += X.shape[0]

        if self.n_buffered_ >= self.buffer_size:
            self._retrain()

        return self

    def flush(self) -> "DecisionTreeClassifier":
        """Retrain on the rows left in the buffer, if any, and clear it.

        Useful at the end of a stream, when fewer than ``buffer_size`` rows
        are waiting.

        Returns
        -------
        DecisionTreeClassifier
            The instance (self).

        Notes
        -----
        Does nothing when the buffer is empty.
        """
        if self.n_buffered_:
            self._retrain()

        return self

    def memory_usage(self) -> dict:
        """Memory held by the stream buffer and the fitted tree, in bytes.

        Returns
        -------
        dict
            ``{"buffer": int, "tree": int}``. The buffer figure is the exact
            size of the buffered arrays; the tree figure is an estimate (each
            node object plus its class counts), 0 before any training.

        Notes
        -----
        Time: O(n_nodes + number of buffered chunks).
        """
        buffer = sum(a.nbytes for a in self._buffer_X) + sum(a.nbytes for a in self._buffer_y)
        tree = _tree_bytes(self.root_) if hasattr(self, "root_") else 0

        return {"buffer": int(buffer), "tree": int(tree)}

    def add_features(self, n_new: int) -> "DecisionTreeClassifier":
        """Accept ``n_new`` extra columns at the end of every later chunk.

        Used when an upstream one-hot encoder learns new categories. Rows
        already in the buffer get zeros in the new columns (they did not have
        those categories), and the current tree keeps predicting, since its
        splits only use the earlier columns.

        Parameters
        ----------
        n_new : int
            Number of columns added (>= 1).

        Returns
        -------
        DecisionTreeClassifier
            The instance (self).

        Raises
        ------
        TypeError
            If ``n_new`` is not an integer.
        ValueError
            If ``n_new`` < 1, or no data has been seen yet.

        Notes
        -----
        Time and extra space: O(n_buffered * n_new).
        """
        _check_int(n_new, "n_new", minimum=1)

        if self._expected_n_features() is None:
            raise ValueError("add_features needs earlier data; the first chunk sets the feature count")

        self._buffer_X = [np.hstack([chunk, np.zeros((chunk.shape[0], n_new))]) for chunk in self._buffer_X]

        if hasattr(self, "n_features_in_"):
            self.n_features_in_ += n_new

        return self

    def _train(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray, rng) -> None:
        """Build a new tree on X, y with the given sorted class list and random generator."""
        start = time.perf_counter()

        # Train on class indices 0..k-1; classes_ maps them back to labels
        self.classes_ = classes
        self.n_classes_ = classes.size
        self.n_features_in_ = X.shape[1]
        self.max_features_ = _resolve_max_features(self.max_features, self.n_features_in_)
        self._impurity = _IMPURITY_FROM_COUNTS[self.criterion]
        self._rng = rng

        self.root_ = self._build(X, np.searchsorted(classes, y), 0)
        self.depth_, self.n_leaves_, self.n_nodes_ = _tree_size(self.root_)
        self.fit_time_ = time.perf_counter() - start

    def _retrain(self) -> None:
        """Train on all buffered rows, then clear the buffer."""
        X = np.concatenate(self._buffer_X)
        y = np.concatenate(self._buffer_y)

        if self._stream_rng is None:
            self._stream_rng = np.random.default_rng(self.random_state)

        self._train(X, y, self._seen_classes, self._stream_rng)
        self._buffer_X, self._buffer_y, self.n_buffered_ = [], [], 0
        self.n_retrains_ += 1

    def _reset_stream(self) -> None:
        """Clear the buffer and the stream state."""
        self._buffer_X, self._buffer_y = [], []
        self.n_buffered_ = 0
        self.n_retrains_ = 0
        self._seen_classes = None
        self._stream_rng = None

    def _expected_n_features(self) -> int | None:
        """Feature count fixed by the current tree or the first buffered chunk, else None."""
        if hasattr(self, "n_features_in_"):
            return self.n_features_in_

        return self._buffer_X[0].shape[1] if self._buffer_X else None

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Return a leaf if a stop rule applies, else split and build both children."""
        counts = np.bincount(y, minlength=self.n_classes_)

        if (depth >= self.max_depth
                or y.size < self.min_samples_split
                or np.count_nonzero(counts) == 1):        # pure node
            return Node(counts)

        split = self._best_split(X, y, counts)

        if split is None:                                 # no split has positive gain
            return Node(counts)

        f, t = split
        mask = X[:, f] <= t

        return Node(counts, feature=f, threshold=t,
                    left=self._build(X[mask], y[mask], depth + 1),
                    right=self._build(X[~mask], y[~mask], depth + 1))

    def _best_split(self, X: np.ndarray, y: np.ndarray, counts: np.ndarray):
        """Return ``(feature, threshold)`` with the highest gain, or None if no gain > 0."""
        features = self._candidate_features(X)

        if features.size == 0:
            return None

        n = y.size
        Xf = X[:, features]                                         # (n, m)
        order = np.argsort(Xf, axis=0, kind="stable")
        xs = np.take_along_axis(Xf, order, axis=0)                  # sorted values
        onehot = np.eye(self.n_classes_, dtype=np.int64)[y[order]]  # (n, m, k)

        # Row i: class counts left of a split between sorted rows i and i+1
        left = np.cumsum(onehot, axis=0)[:-1]                       # (n-1, m, k)
        right = counts - left
        n_left = np.arange(1, n)[:, None]

        after = (n_left * self._impurity(left) + (n - n_left) * self._impurity(right)) / n
        gain = self._impurity(counts) - after                       # (n-1, m)
        gain = np.where(xs[1:] > xs[:-1], gain, -np.inf)            # only between distinct values

        # Feature-major order: the first near-best entry has the lowest feature, then threshold
        gain = gain.T
        best = gain.max()

        if best <= _GAIN_TOL:
            return None

        fi, i = divmod(int(np.flatnonzero(gain >= best - _GAIN_TOL)[0]), n - 1)

        # Midpoint, halving first to avoid overflow; keep it strictly below the upper value
        lo, hi = xs[i, fi], xs[i + 1, fi]
        t = lo / 2 + hi / 2

        if not t < hi:
            t = lo

        return int(features[fi]), float(t)

    def _candidate_features(self, X: np.ndarray) -> np.ndarray:
        """Sorted indices of non-constant features, subsampled to ``max_features_``."""
        usable = np.flatnonzero(np.ptp(X, axis=0) > 0)

        if usable.size <= self.max_features_:
            return usable

        return np.sort(self._rng.choice(usable, size=self.max_features_, replace=False))

    # -- prediction ----------------------------------------------------------

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities for each row, from the class counts of its leaf.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Finite numeric features.

        Returns
        -------
        ndarray of float, shape (n_samples, n_classes)
            Rows sum to 1; columns follow ``classes_``.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        TypeError
            If X is not numeric.
        ValueError
            If X is not 2-D, is empty, contains NaN or infinity, or has a
            different number of features than during fit.

        Notes
        -----
        Rows move down the tree as index arrays, so there is no loop over rows.
        Time: O(n * depth).
        Space: O(n * k).
        """
        X = self._check_predict_X(X)
        proba = np.empty((X.shape[0], self.n_classes_))
        _route(self.root_, X, np.arange(X.shape[0]), proba)

        return proba

    def predict(self, X) -> np.ndarray:
        """Predicted class label for each row.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Finite numeric features.

        Returns
        -------
        ndarray, shape (n_samples,)
            Labels from ``classes_``; ties go to the lowest class.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        TypeError
            If X is not numeric.
        ValueError
            If X is not 2-D, is empty, contains NaN or infinity, or has a
            different number of features than during fit.

        Notes
        -----
        Time: O(n * depth).
        Space: O(n * k).
        """
        proba = self.predict_proba(X)          # validate before reading classes_

        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y) -> float:
        """Accuracy of the predictions for X against y.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Finite numeric features.
        y : array_like, shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Value in [0, 1].

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        TypeError
            If X is not numeric or the labels are incompatible.
        ValueError
            If X or y is invalid or their lengths differ.

        Notes
        -----
        Time: O(n * depth).
        """
        return accuracy(y, self.predict(X))

    # -- inspection ----------------------------------------------------------

    def export_text(self, feature_names=None, decimals: int = 3) -> str:
        """Text representation of the fitted tree, one split or leaf per line.

        Parameters
        ----------
        feature_names : sequence of str or None, default None
            One name per feature; None uses ``x0, x1, ...``.
        decimals : int, default 3
            Decimal places shown for thresholds.

        Returns
        -------
        str
            Lines such as ``|--- x0 <= 0.500`` and ``|   |--- class: 1``.

        Raises
        ------
        RuntimeError
            If the tree is not fitted.
        TypeError
            If ``decimals`` is not an integer.
        ValueError
            If ``feature_names`` has the wrong length or ``decimals`` < 0.

        Notes
        -----
        Time and space: O(n_nodes).
        """
        self._check_fitted()
        _check_int(decimals, "decimals", minimum=0)

        if feature_names is None:
            feature_names = [f"x{j}" for j in range(self.n_features_in_)]

        elif len(feature_names) != self.n_features_in_:
            raise ValueError(
                f"feature_names needs {self.n_features_in_} names, got {len(feature_names)}"
            )

        lines = []

        def walk(node: Node, indent: str) -> None:
            if node.is_leaf:
                lines.append(f"{indent}|--- class: {self.classes_[node.label]}")
                return
            name, t = feature_names[node.feature], f"{node.threshold:.{decimals}f}"
            lines.append(f"{indent}|--- {name} <= {t}")
            walk(node.left, indent + "|   ")
            lines.append(f"{indent}|--- {name} >  {t}")
            walk(node.right, indent + "|   ")

        walk(self.root_, "")

        return "\n".join(lines)

    # -- validation ----------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise RuntimeError if ``fit`` has not been called."""
        if not hasattr(self, "root_"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; call fit(X, y), or partial_fit "
                f"until {self.buffer_size} rows are buffered"
            )

    def _check_predict_X(self, X) -> np.ndarray:
        """Validate X for prediction, including the fitted feature count."""
        self._check_fitted()
        X = _check_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got shape {X.shape}")

        return X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _route(node: Node, X: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
    """Move rows ``idx`` down from ``node`` and write their leaf probabilities into ``out``."""
    if idx.size == 0:
        return

    if node.is_leaf:
        out[idx] = node.value / node.value.sum()
        return

    yes = X[idx, node.feature] <= node.threshold
    _route(node.left, X, idx[yes], out)
    _route(node.right, X, idx[~yes], out)


def _tree_size(node: Node) -> tuple[int, int, int]:
    """Return ``(depth, n_leaves, n_nodes)`` of the subtree rooted at ``node``."""
    if node.is_leaf:
        return 0, 1, 1

    dl, ll, nl = _tree_size(node.left)
    dr, lr, nr = _tree_size(node.right)

    return 1 + max(dl, dr), ll + lr, 1 + nl + nr


def _tree_bytes(node: Node) -> int:
    """Approximate bytes of the subtree rooted at ``node``: each node object plus its counts."""
    size = sys.getsizeof(node) + node.value.nbytes

    if not node.is_leaf:
        size += _tree_bytes(node.left) + _tree_bytes(node.right)

    return size


def _check_X(X) -> np.ndarray:
    """Return X as a nonempty, finite 2-D float array or raise."""
    X = check_array(X, ndim=2)

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinity; impute first, e.g. SimpleImputer in a Pipeline")

    return X


def _check_Xy(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate X and y for training and check they have the same number of rows."""
    X = _check_X(X)
    y = _check_labels(y, "y")

    if X.shape[0] != y.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size} labels")

    return X, y



def _check_max_features(max_features) -> None:
    """Raise if ``max_features`` is not None, "sqrt", "log2", an int >= 1 or a float in (0, 1]."""
    if max_features is None or max_features in _MAX_FEATURES_STR:
        return

    if isinstance(max_features, (float, np.floating)):
        if not 0.0 < max_features <= 1.0:
            raise ValueError(f"max_features as a float must be in (0, 1], got {max_features}")
        return

    if isinstance(max_features, str):
        raise ValueError(f"max_features must be 'sqrt' or 'log2' as a string, got {max_features!r}")

    _check_int(max_features, "max_features", minimum=1)


def _resolve_max_features(max_features, n_features: int) -> int:
    """Convert the ``max_features`` option to a count in [1, n_features]."""
    if max_features is None:
        m = n_features
    elif max_features == "sqrt":
        m = int(np.sqrt(n_features))
    elif max_features == "log2":
        m = int(np.log2(n_features))
    elif isinstance(max_features, (float, np.floating)):
        m = int(max_features * n_features)
    else:
        m = int(max_features)

    return min(max(m, 1), n_features)
