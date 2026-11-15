"""NumCompute-Stream: a NumPy-only toolkit for streaming, decision tree-based learning.

Extends NumCompute (Assignment 2.1) so that models, preprocessing, statistics
and metrics can learn from data that arrives in chunks.

Modules
-------
stream        ``StreamTrainer``: test-then-train over chunks, logging metrics and memory.
tree          ``DecisionTreeClassifier`` with buffered retraining through ``partial_fit``.
preprocessing Scalers, imputer and one-hot encoder, each with ``partial_fit``.
stats         Descriptive statistics, ``RunningStats`` and ``StreamingStats.update_stats``.
metrics       Classification, regression and ROC metrics, plus streaming metrics
              with ``update``/``result``/``reset`` and optional rolling windows.
pipeline      Chains transformers and a final estimator; ``partial_fit`` streams.
visualise     Matplotlib plots for metrics over time, model comparison and predictions.
io            CSV reading, including chunked streaming with ``iter_csv_chunks``.
sort_search   Sorting, top-k, quickselect and binary search.
rank          Ranking and percentiles.
optim         Finite-difference gradients and Jacobians.
utils         Array checks, batching and numerically stable helpers.
benchmarking  Timing helpers for loop vs vectorised comparisons.
"""

__version__ = "0.2.0"
