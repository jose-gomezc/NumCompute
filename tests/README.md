# Tests

Unit and end-to-end tests for `numcompute_stream`, run with pytest. What each file covers is listed in the Tests section of the main [README](../README.md#tests).

## Running

From the project root, after `python -m pip install -e ".[test]"`:

```bash
python -m pytest tests -q                              # everything
python -m pytest tests/test_tree.py -q                 # one file
python -m pytest tests -q -k "partial_fit"             # tests whose name matches
```

## Layout

- **2.1 files** (`test_io.py`, `test_preprocessing.py`, `test_sort_search.py`, `test_rank.py`, `test_stats.py`, `test_metrics.py`, `test_optim.py`, `test_pipeline.py`, `test_utils.py`, `test_benchmarking.py`): the batch API, kept and extended where 2.2 changed a behaviour.
- **2.2 files** (`test_tree.py`, `test_streaming_metrics.py`, `test_streaming_stats_preprocessing.py`, `test_stream_pipeline.py`, `test_stream_trainer.py`, `test_visualise.py`, `test_stream_benchmarks.py`, `test_end_to_end.py`): the streaming API, one file per area, plus a CSV-to-trainer run and a top-to-bottom run of `demo/stream_demo.ipynb`.

## Conventions

- `conftest.py` selects matplotlib's off-screen Agg backend and closes all figures after each test.
- Random data uses fixed seeds, and the end-to-end tests read the committed CSV files, so expected numbers are fixed.
- Streaming tests compare against the batch result where one exists (`partial_fit` over chunks against `fit` on all rows).
- Invalid-input cases check both the exception type and its message, and that a rejected chunk leaves the state unchanged.
- Parametrised cases carry `ids=` so a failure names the case.
- Benchmark tests check results and report generation, never a fixed runtime, since timings depend on the machine.
