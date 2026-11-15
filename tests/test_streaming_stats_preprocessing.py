"""Unit tests for StreamingStats and the partial_fit methods of the preprocessing transformers."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.stats import StreamingStats


def _drifting_chunks(n_chunks=6, size=200):
    """Chunks whose mean and spread grow over time, with some NaNs."""
    rng = np.random.default_rng(7)
    chunks = [rng.normal(loc=3 * i, scale=1 + i, size=(size, 3)) for i in range(n_chunks)]
    chunks[0][:5, 1] = np.nan
    chunks[2][:, 2] = np.nan           # a whole chunk missing in one feature
    return chunks


# StreamingStats

def test_streaming_moments_match_batch_values():
    chunks = _drifting_chunks()
    chunks[1][:, 0] += 1e9             # far from zero: Welford keeps the precision
    X = np.vstack(chunks)
    stats = StreamingStats()
    for chunk in chunks:
        stats.update_stats(chunk)

    assert stats.n_ == X.shape[0]
    np.testing.assert_array_equal(stats.count_, (~np.isnan(X)).sum(axis=0))
    np.testing.assert_allclose(stats.mean(), np.nanmean(X, axis=0))
    np.testing.assert_allclose(stats.var(), np.nanvar(X, axis=0))
    np.testing.assert_allclose(stats.std(ddof=1), np.nanstd(X, axis=0, ddof=1))
    np.testing.assert_array_equal(stats.min(), np.nanmin(X, axis=0))
    np.testing.assert_array_equal(stats.max(), np.nanmax(X, axis=0))


def test_histogram_widens_with_drift_and_quantiles_stay_within_one_bin():
    chunks = _drifting_chunks()
    X = np.vstack(chunks)
    stats = StreamingStats(n_bins=64)
    for chunk in chunks:
        stats.update_stats(chunk)

    counts, edges = stats.histogram()
    np.testing.assert_array_equal(counts.sum(axis=1), (~np.isnan(X)).sum(axis=0))   # no count lost
    assert (edges[:, 0] <= np.nanmin(X, axis=0)).all() and (edges[:, -1] >= np.nanmax(X, axis=0)).all()

    q = [0.0, 0.1, 0.5, 0.9, 1.0]
    width = edges[:, 1] - edges[:, 0]
    error = np.abs(stats.quantile(q) - np.nanquantile(X, q, axis=0))
    assert (error <= width).all()
    np.testing.assert_array_equal(stats.quantile([0.0, 1.0]), [np.nanmin(X, axis=0), np.nanmax(X, axis=0)])

    rising, falling = StreamingStats(n_bins=64), StreamingStats(n_bins=64)
    for i in range(6):                 # new room goes on the side the data overflowed
        rising.update_stats(np.linspace(i, i + 1.0, 50)[:, None])
        falling.update_stats(np.linspace(-i - 1.0, -i, 50)[:, None])
    np.testing.assert_allclose(rising.histogram()[1][0, [0, -1]], [0.0, 8.0])
    np.testing.assert_allclose(falling.histogram()[1][0, [0, -1]], [-8.0, 0.0])


def test_window_keeps_only_the_last_chunks():
    chunks = _drifting_chunks()
    recent = np.vstack(chunks[-2:])
    stats = StreamingStats(window=2)
    for chunk in chunks:
        stats.update_stats(chunk)

    assert stats.n_ == recent.shape[0]
    np.testing.assert_allclose(stats.mean(), recent.mean(axis=0))
    np.testing.assert_allclose(stats.var(), recent.var(axis=0))
    np.testing.assert_array_equal(stats.min(), recent.min(axis=0))
    np.testing.assert_array_equal(stats.histogram()[0].sum(axis=1), [400, 400, 400])


def test_streaming_stats_edge_cases():
    stats = StreamingStats()
    with pytest.raises(RuntimeError, match="no data"):
        stats.quantile(0.5)

    stats.update_stats(np.empty((0, 3)))                      # empty chunk: ignored
    stats.update_stats([[5.0, np.nan, 1.0], [5.0, np.nan, 2.0]])
    stats.update_stats([[5.0, np.nan, 3.0]])
    assert stats.var()[0] == 0.0                              # constant feature
    assert np.isnan(stats.mean()[1]) and np.isnan(stats.quantile(0.5)[1])   # no values yet

    with pytest.raises(ValueError, match="expected 3 features"):
        stats.update_stats([[1.0, 2.0]])
    with pytest.raises(ValueError, match="infinity"):
        stats.update_stats([[1.0, np.inf, 2.0]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        stats.quantile(1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        stats.quantile(np.nan)
    with pytest.raises(ValueError, match="1-D"):
        stats.quantile([[0.5]])
    with pytest.raises(TypeError, match="number"):
        stats.quantile("median")
    stats.update_stats([])                                    # empty chunk: ignored
    assert stats.n_ == 3                                      # rejected chunks changed nothing


# Preprocessing

def test_standard_scaler_partial_fit_matches_fit():
    chunks = _drifting_chunks()
    chunks[3][:, 0] += 1e9
    for chunk in chunks:
        chunk[:, 2] = np.where(np.isnan(chunk[:, 2]), np.nan, 4.0)      # constant column
    X = np.vstack(chunks)

    stream = StandardScaler()
    for chunk in chunks:
        stream.partial_fit(chunk)
    batch = StandardScaler().fit(X)

    np.testing.assert_allclose(stream.mean_, batch.mean_)
    np.testing.assert_allclose(stream.scale_, batch.scale_)
    np.testing.assert_allclose(stream.scale_, [np.nanstd(X[:, 0]), np.nanstd(X[:, 1]), 1.0])


def test_min_max_scaler_partial_fit_widens_the_range():
    chunks = _drifting_chunks()
    stream = MinMaxScaler()
    stream.partial_fit(chunks[0])
    first_max = stream.data_max_.copy()
    for chunk in chunks[1:]:
        stream.partial_fit(chunk)

    batch = MinMaxScaler().fit(np.vstack(chunks))
    assert (stream.data_max_ > first_max).all()
    np.testing.assert_array_equal(stream.data_min_, batch.data_min_)
    np.testing.assert_array_equal(stream.data_range_, batch.data_range_)


def test_simple_imputer_partial_fit():
    chunks = _drifting_chunks()
    X = np.vstack(chunks)

    mean = SimpleImputer("mean")
    for chunk in chunks:
        mean.partial_fit(chunk)
    np.testing.assert_allclose(mean.statistics_, SimpleImputer("mean").fit(X).statistics_)

    # A column with no values yet uses fill_value, then its mean once values arrive
    late = SimpleImputer("mean", fill_value=-1).partial_fit([[1.0, np.nan], [3.0, np.nan]])
    np.testing.assert_allclose(late.statistics_, [2.0, -1.0])
    late.partial_fit([[5.0, 10.0]])
    np.testing.assert_allclose(late.statistics_, [3.0, 10.0])

    # Streaming median: within one bin width of the exact median
    median, stats = SimpleImputer("median"), StreamingStats()     # same default bins
    for chunk in chunks:
        median.partial_fit(chunk)
        stats.update_stats(chunk)
    edges = stats.histogram()[1]
    width = edges[:, 1] - edges[:, 0]
    assert (np.abs(median.statistics_ - np.nanmedian(X, axis=0)) <= width).all()

    # Even count: the two middle values are averaged, as fit (np.median) does
    small = SimpleImputer("median").partial_fit([[1.0], [2.0]]).partial_fit([[3.0], [np.nan], [4.0]])
    np.testing.assert_allclose(small.statistics_, [2.5], atol=0.02)

    constant = SimpleImputer("constant", fill_value=0).partial_fit(chunks[0])
    np.testing.assert_array_equal(constant.statistics_, [0.0, 0.0, 0.0])


def test_one_hot_encoder_appends_new_categories_at_the_end():
    first = [["red", "S"], ["blue", "M"]]
    enc = OneHotEncoder().partial_fit(first)
    np.testing.assert_array_equal(enc.transform(first), OneHotEncoder().fit(first).transform(first))
    old = enc.transform(first)

    enc.partial_fit([["green", "M"], ["red", "L"]])            # new: green (feature 0), L (feature 1)
    assert [c.tolist() for c in enc.categories_] == [["blue", "red", "green"], ["M", "S", "L"]]
    assert enc.n_features_out_ == 6

    # Old columns keep their positions; old rows are zero in the new columns
    np.testing.assert_array_equal(enc.transform(first), np.hstack([old, np.zeros((2, 2))]))
    np.testing.assert_array_equal(enc.transform([["green", "L"]]), [[0, 0, 0, 0, 1, 1]])


@pytest.mark.parametrize(
    "transformer, X",
    [
        (StandardScaler(), [[1.0, 2.0], [3.0, 4.0]]),
        (MinMaxScaler(), [[1.0, 2.0], [3.0, 4.0]]),
        (SimpleImputer("median"), [[1.0, np.nan], [3.0, 4.0]]),
        (OneHotEncoder(), [["a", "x"], ["b", "y"]]),
    ],
    ids=["standard", "minmax", "imputer", "onehot"],
)
def test_partial_fit_rejects_bad_chunks_without_changing_state(transformer, X):
    with pytest.raises(RuntimeError, match="not fitted"):
        transformer.transform(X)

    transformer.partial_fit(X)
    before = transformer.transform(X)
    with pytest.raises(ValueError, match="expected 2 features"):
        transformer.partial_fit([[1.0, 2.0, 3.0]])
    if not isinstance(transformer, OneHotEncoder):             # "inf" is just a category to the encoder
        with pytest.raises(ValueError, match="infinity"):
            transformer.partial_fit([[np.inf, 1.0]])
    transformer.partial_fit(np.empty((0, 2)))                  # empty chunks: ignored
    transformer.partial_fit([])
    np.testing.assert_array_equal(transformer.transform(X), before)
