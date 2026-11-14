"""Unit tests for the streaming benchmarks: the loop baselines and the runner.

Timings depend on the machine, so nothing here asserts a speed: the tests check
that each loop version gives the same result as the NumPy version it is timed
against, and that the runner writes its report and figures.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmark import loops
from numcompute_stream.stats import StreamingStats
from numcompute_stream.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_loop_baselines_match_the_vectorised_code(criterion):
    rng = np.random.default_rng(10)
    X = rng.integers(0, 5, size=(300, 4)).astype(float)          # many tied values
    y = rng.integers(0, 3, size=300)

    loop_tree = loops.LoopSplitTree(max_depth=4, criterion=criterion).fit(X, y)
    tree = DecisionTreeClassifier(max_depth=4, criterion=criterion).fit(X, y)
    assert loop_tree.export_text() == tree.export_text()
    np.testing.assert_array_equal(loops.predict_loop(tree, X), tree.predict(X))

    chunks = np.array_split(rng.normal(1e3, 1.0, size=(500, 3)), 5)
    stats = StreamingStats()
    for chunk in chunks:
        stats.update_stats(chunk)
    mean, var = loops.welford_loop(chunks)
    np.testing.assert_allclose(mean, stats.mean())
    np.testing.assert_allclose(var, stats.var())


def test_stream_runner_quick_writes_report_and_figures(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "benchmark.run_stream_benchmarks", "--quick", "--output-dir", str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert done.returncode == 0, done.stderr

    report = (tmp_path / "stream_results.md").read_text(encoding="utf-8")
    for section in ("## Environment", "## Loop vs vectorised", "## buffer_size sweep", "## Figures"):
        assert section in report
    for case in ("tree root split", "tree fit", "tree predict", "streaming mean/var"):
        assert case in report
    for name in ("rolling_accuracy.png", "retrain_time.png", "accuracy_after_drift.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 0
