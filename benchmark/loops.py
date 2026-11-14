"""Plain-Python versions of NumCompute functions: the "loop" side of the benchmarks.

They loop over Python lists one element at a time, with no NumPy inside the loop,
so the benchmarks show what vectorisation gains. Results are turned into NumPy
arrays only at the end, so they can be checked against the NumCompute versions.
"""

import csv
import math

import numpy as np

from numcompute_stream.io import DEFAULT_MISSING
from numcompute_stream.tree import _GAIN_TOL, DecisionTreeClassifier


# io / metrics
def load_csv_loop(path, delimiter=","):
    """Numeric CSV with a header row -> float array"""
    missing = set(DEFAULT_MISSING)

    rows = []
    
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Skip the header row (only numbers are kept)
        next(reader)

        # Read row by row and convert values accordingly
        for row in reader:
            values = []
            for cell in row:
                cell = cell.strip()
                values.append(math.nan if cell in missing else float(cell))
            rows.append(values)
    
    return np.array(rows, dtype=float)


def accuracy_loop(y_true, y_pred):
    matches = 0

    # Loop through the arrays to count the matching values
    for t, p in zip(y_true, y_pred):
        if t == p:
            matches += 1
    
    return matches / len(y_true)


def confusion_matrix_loop(y_true, y_pred):
    """True labels in rows. Predicted labels in columns. Labels sorted."""
    labels = sorted(set(y_true) | set(y_pred))
    
    index = {label: i for i, label in enumerate(labels)}
    
    counts = [[0] * len(labels) for _ in labels]
    
    for t, p in zip(y_true, y_pred):
        counts[index[t]][index[p]] += 1
    
    return np.array(counts)


def mse_loop(y_true, y_pred):
    total = 0.0

    # Sum the squared error of each prediction
    for t, p in zip(y_true, y_pred):
        total += (t - p) ** 2
    
    return total / len(y_true)


def auc_loop(y_true, y_score, pos_label=1):
    """Share of pairs (positive, negative) where the positive scores higher, a tie counting half."""

    pos = [s for t, s in zip(y_true, y_score) if t == pos_label]
    neg = [s for t, s in zip(y_true, y_score) if t != pos_label]
    
    wins = 0.0

    # Compare every positive score with every negative score
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5

    # Share of pairs the positive wins
    return wins / (len(pos) * len(neg))


# Preprocessing / stats
def standard_scale_loop(X):
    """Column-wise z-score with the population std (ddof=0); X is a list of rows."""
    n_rows, n_cols = len(X), len(X[0])
    out = [[0.0] * n_cols for _ in range(n_rows)]
    
    for j in range(n_cols):
        # Mean computation
        mean = 0.0
        for i in range(n_rows):
            mean += X[i][j]
        mean /= n_rows

        # Standard deviation computation
        var = 0.0
        for i in range(n_rows):
            var += (X[i][j] - mean) ** 2
        std = math.sqrt(var / n_rows) or 1.0  # or 1.0 to avoid zero division
    
        for i in range(n_rows):
            out[i][j] = (X[i][j] - mean) / std

    return np.array(out)


def histogram_loop(x, bins):
    """Counts in `bins` equal-width bins from min(x) to max(x); the last bin includes max(x)."""
    low, high = min(x), max(x)

    width = (high - low) / bins
    
    counts = [0] * bins
    
    for v in x:
        i = int((v - low) / width)
        counts[min(i, bins - 1)] += 1                 # max(x) lands on the last edge
    
    return np.array(counts)


# sort_search / rank
def average_rank_loop(x):
    """1-based ranks; tied values share the average of their ranks. O(n^2)."""
    ranks = []
    
    for v in x:
        less = 0
        equal = 0
        
        for w in x:
            if w < v:
                less += 1
            elif w == v:
                equal += 1
        
        ranks.append(less + (equal + 1) / 2)
    
    return np.array(ranks)


def topk_loop(x, k):
    """The k largest values, largest first, and their indices, by k passes over the data."""
    taken = [False] * len(x)
    values, indices = [], []
    for _ in range(k):
        best = -1
        for i in range(len(x)):
            if not taken[i] and (best == -1 or x[i] > x[best]):
                best = i
        
        taken[best] = True
        values.append(x[best])
        indices.append(best)
    
    return np.array(values), np.array(indices)


def stable_sort_loop(x):
    """Ascending sort by merge sort; equal values keep their input order. O(n log n)."""
    return np.array(_merge_sort(x))


def _merge_sort(x):
    """Sorted copy of the list x: sort each half, then merge them."""
    if len(x) <= 1:
        return list(x)
    
    middle = len(x) // 2
    left = _merge_sort(x[:middle])
    right = _merge_sort(x[middle:])
    
    merged = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if right[j] < left[i]:                    # strict: on a tie the left value goes first
            merged.append(right[j])
            j += 1
        
        else:
            merged.append(left[i])
            i += 1
    
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged


# Streaming tree and statistics (Assignment 2.2)

def _impurity_loop(counts, n, criterion):
    """Gini or entropy of one group from its class counts, one class at a time."""
    if n == 0:
        return 0.0

    if criterion == "gini":
        total = 0.0
        for c in counts:
            total += (c / n) ** 2
        return 1.0 - total

    total = 0.0
    for c in counts:
        if c:
            p = c / n
            total -= p * math.log2(p)
    
    return total


def best_split_loop(columns, labels, n_classes, criterion="gini"):
    """Best (column index, threshold) by information gain, or None; plain-Python sweep.

    Same algorithm as the vectorised search: sort each column once, then move
    the threshold one row at a time, keeping running class counts. Ties go to
    the lowest column, then the lowest threshold.
    """
    n = len(labels)
    total = [0] * n_classes
    
    for label in labels:
        total[label] += 1

    parent = _impurity_loop(total, n, criterion)
    best_gain, best = _GAIN_TOL, None

    for j, column in enumerate(columns):
        pairs = sorted(zip(column, labels))
        left = [0] * n_classes

        for i in range(n - 1):
            value, label = pairs[i]
            left[label] += 1
            upper = pairs[i + 1][0]

            if upper == value:                     # a threshold must separate two values
                continue

            n_left = i + 1
            right = [t - l for t, l in zip(total, left)]
            after = (n_left * _impurity_loop(left, n_left, criterion)
                     + (n - n_left) * _impurity_loop(right, n - n_left, criterion)) / n
            gain = parent - after

            if gain > best_gain + _GAIN_TOL:
                threshold = value / 2 + upper / 2
                best_gain, best = gain, (j, threshold if threshold < upper else value)

    return best


class LoopSplitTree(DecisionTreeClassifier):
    """DecisionTreeClassifier whose split search is the plain-Python sweep above."""

    def _best_split(self, X, y, counts):
        """Same contract as the vectorised search, computed with Python loops."""
        features = self._candidate_features(X)

        if features.size == 0:
            return None

        columns = [X[:, f].tolist() for f in features]
        best = best_split_loop(columns, y.tolist(), self.n_classes_, self.criterion)

        return None if best is None else (int(features[best[0]]), float(best[1]))


def predict_loop(tree, X):
    """Predict one row at a time by walking it down the fitted tree."""
    labels = []

    for row in np.asarray(X, dtype=float).tolist():
        node = tree.root_
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        labels.append(tree.classes_[node.label])

    return np.array(labels)


def welford_loop(chunks):
    """Per-feature mean and population variance, updated one value at a time (Welford)."""
    count = mean = m2 = None

    for chunk in chunks:
        for row in chunk.tolist():
            if count is None:
                d = len(row)
                count, mean, m2 = [0] * d, [0.0] * d, [0.0] * d

            for j, x in enumerate(row):
                count[j] += 1
                delta = x - mean[j]
                mean[j] += delta / count[j]
                m2[j] += delta * (x - mean[j])

    return np.array(mean), np.array([m / c for m, c in zip(m2, count)])
