"""NumCompute-Stream: a NumPy-only toolkit for streaming, decision tree-based learning.

Extends NumCompute (Assignment 2.1) so that models, preprocessing, statistics
and metrics can learn from data that arrives in chunks.

Modules
-------
io            CSV reading, including chunked streaming with ``iter_csv_chunks``.
preprocessing Scalers, imputer and one-hot encoder.
stats         Descriptive statistics and ``RunningStats``.
metrics       Classification, regression and ROC metrics.
pipeline      Chains transformers and a final estimator.
sort_search   Sorting, top-k, quickselect and binary search.
rank          Ranking and percentiles.
optim         Finite-difference gradients and Jacobians.
utils         Array checks, batching and numerically stable helpers.
benchmarking  Timing helpers for loop vs vectorised comparisons.
"""

__version__ = "0.2.0"
