"""End-to-end tests: a CSV file streamed through a pipeline, and the demo notebook.

These use the committed demo data (demo/data/stream.csv, 20,000 rows with a
concept drift at row 10,000), so the expected numbers are fixed.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream import io
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM_CSV = REPO_ROOT / "demo" / "data" / "stream.csv"
FEATURES = ["x0", "x1", "x2", "x3", "x4"]


def test_csv_stream_trains_pipeline_and_recovers_from_drift():
    chunks = ((chunk[:, :-1], chunk[:, -1].astype(int))
              for chunk in io.iter_csv_chunks(STREAM_CSV, 100, usecols=[*FEATURES, "label"], dtype=float))
    pipe = Pipeline([
        ("impute", SimpleImputer()),                     # x2 has missing cells
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
    ])
    history = StreamTrainer(pipe, window=10).run(chunks, flush=False)
    accuracy = np.asarray(history["accuracy"])

    assert all(len(values) == 200 for values in history.values())
    np.testing.assert_array_equal(np.flatnonzero(history["retrained"]), np.arange(4, 200, 5))
    assert np.isnan(accuracy[:5]).all() and not np.isnan(accuracy[5:]).any()
    assert history["cumulative_accuracy"][-1] > 0.8
    assert accuracy[100:105].mean() < 0.65 < 0.8 < accuracy[110:].mean()      # drift at chunk 100, then recovery


def test_new_category_in_csv_stream_grows_encoder_and_tree():
    encoder, tree = OneHotEncoder(), DecisionTreeClassifier(max_depth=3, buffer_size=500)
    pipe = Pipeline([("encode", encoder), ("tree", tree)])
    widths = []

    for chunk in io.iter_csv_chunks(STREAM_CSV, 100, usecols=["segment", "label"]):
        pipe.partial_fit(chunk[:, :1], chunk[:, 1].astype(int))
        widths.append(encoder.n_features_out_)

    assert widths == [3] * 100 + [4] * 100                # category D first appears at chunk 100
    np.testing.assert_array_equal(encoder.categories_[0], ["A", "B", "C", "D"])
    assert tree.n_features_in_ == 4
    np.testing.assert_array_equal(encoder.transform(np.array([["D"]])), [[0, 0, 0, 1]])
    assert np.isin(pipe.predict(np.array([["D"], ["A"]])), [0, 1]).all()


def test_demo_notebook_runs_top_to_bottom(monkeypatch):
    notebook = json.loads((REPO_ROOT / "demo" / "stream_demo.ipynb").read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert notebook["nbformat"] == 4 and cells

    monkeypatch.chdir(REPO_ROOT / "demo")                  # the notebook finds the root from demo/ too
    monkeypatch.setattr(sys, "path", list(sys.path))      # undo the notebook's sys.path insert
    namespace = {"__name__": "__main__"}

    for i, cell in enumerate(cells):
        source = "".join(cell["source"])
        assert not any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()), f"cell {i} uses a magic"
        exec(compile(source, f"stream_demo.ipynb code cell {i}", "exec"), namespace)
        plt.close("all")

    assert len(namespace["history"]["chunk"]) == 200
