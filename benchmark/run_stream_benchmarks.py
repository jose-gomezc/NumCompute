"""Benchmark the streaming tree: Python loops vs NumPy, and the effect of buffer_size.

Run from the repository root:

    python -m benchmark.run_stream_benchmarks            # full sizes (about a minute)
    python -m benchmark.run_stream_benchmarks --quick    # small sizes, a few seconds

Writes benchmark/stream_results.md and three PNG figures in benchmark/figures/.
Data comes from demo/data/stream.csv (see demo/generate_stream.py), read with
numcompute_stream.io, and from np.random.default_rng(SEED).
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                    # figures are only saved, never shown

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import loops  # noqa: E402
from numcompute_stream.benchmarking import compare, environment_info, format_table  # noqa: E402
from numcompute_stream.io import load_csv  # noqa: E402
from numcompute_stream.pipeline import Pipeline  # noqa: E402
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler  # noqa: E402
from numcompute_stream.stats import StreamingStats  # noqa: E402
from numcompute_stream.stream import StreamTrainer  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier  # noqa: E402
from numcompute_stream.visualise import plot_metric_over_time  # noqa: E402

SEED = 42
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "demo" / "data" / "stream.csv"
DRIFT_ROW = 10_000
FEATURES = ["x0", "x1", "x2", "x3", "x4"]
PLOT_SIZES = (100, 500, 2000, 5000)                       # four lines: the colour-blind-safe set

# --quick is for smoke tests, not for reporting
FULL = {"fit_n": 5_000, "predict_n": 20_000, "welford_shape": (100_000, 5), "welford_chunk": 1_000,
        "repeat": 7, "step": 1, "chunk": 100, "buffer_sizes": (100, 250, 500, 1000, 2000, 5000),
        "sweep_repeat": 3}
QUICK = {"fit_n": 300, "predict_n": 1_000, "welford_shape": (5_000, 5), "welford_chunk": 500,
         "repeat": 2, "step": 10, "chunk": 100, "buffer_sizes": (100, 500), "sweep_repeat": 1}


def load_stream(step: int = 1):
    """Features (NaN where missing) and labels of stream.csv, keeping every ``step``-th row."""
    X = load_csv(DATA, usecols=FEATURES, dtype=float)
    y = load_csv(DATA, usecols="label", dtype=int).ravel()
    
    return X[::step], y[::step]


def loop_cases(cfg, X, y, rng):
    """Return the loop-vs-vectorised cases, each (operation, loop_func, vec_func, notes)."""
    n = cfg["fit_n"]
    X_fit = SimpleImputer().fit_transform(X[:n])          # the tree needs complete rows
    y_fit = y[:n]
    fitted = DecisionTreeClassifier(max_depth=5).fit(X_fit, y_fit)
    X_pred = SimpleImputer().fit_transform(X[:cfg["predict_n"]])
    chunks = np.array_split(rng.normal(1e3, 1.0, size=cfg["welford_shape"]),
                            cfg["welford_shape"][0] // cfg["welford_chunk"])

    def welford_vec():
        stats = StreamingStats()
        for chunk in chunks:
            stats.update_stats(chunk)
        return stats.mean(), stats.var()

    return [
        (f"tree root split (max_depth=1) n={n:,} d=5",
         lambda: loops.LoopSplitTree(max_depth=1).fit(X_fit, y_fit).export_text(),
         lambda: DecisionTreeClassifier(max_depth=1).fit(X_fit, y_fit).export_text(),
         "Sort once per feature, then sweep thresholds with running counts: Python lists vs cumsum"),
        (f"tree fit max_depth=5 n={n:,} d=5",
         lambda: loops.LoopSplitTree(max_depth=5).fit(X_fit, y_fit).export_text(),
         lambda: DecisionTreeClassifier(max_depth=5).fit(X_fit, y_fit).export_text(),
         "Same tree (export_text checked); only the split search differs"),
        (f"tree predict n={len(X_pred):,}",
         lambda: loops.predict_loop(fitted, X_pred),
         lambda: fitted.predict(X_pred),
         "Each row walks the tree vs index arrays moving one level at a time"),
        (f"streaming mean/var {cfg['welford_shape'][0]:,} x {cfg['welford_shape'][1]} "
         f"in chunks of {cfg['welford_chunk']:,}",
         lambda: loops.welford_loop(chunks),
         welford_vec,
         "Welford per value vs one Chan/Welford merge per chunk"),
    ]


def sweep(X, y, sizes, chunk, repeat, drift_chunk):
    """Stream X, y through imputer -> scaler -> tree for each buffer_size; return rows and histories."""
    rows, histories = [], {}

    for size in sizes:
        runs = []
        for _ in range(repeat):
            pipe = Pipeline([
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("model", DecisionTreeClassifier(max_depth=5, buffer_size=size)),
            ])
            trainer = StreamTrainer(pipe, window=10)
            stream = ((X[s:s + chunk], y[s:s + chunk]) for s in range(0, len(y), chunk))
            runs.append(trainer.run(stream, flush=False))

        history = runs[0]                                  # accuracy is the same in every run
        totals = [np.nansum(h["retrain_time"]) for h in runs]
        n_retrains = int(history["n_retrains"][-1])
        total_ms = float(np.median(totals)) * 1000
        histories[size] = history
        rows.append({
            "buffer_size": size,
            "retrains": n_retrains,
            "total_ms": total_ms,
            "mean_ms": total_ms / n_retrains if n_retrains else np.nan,
            "final_accuracy": history["cumulative_accuracy"][-1],
            "after_drift": float(np.nanmean(history["accuracy"][drift_chunk:])),
            "peak_kb": float(np.nanmax(history["buffer_bytes"])) / 1024,
        })

    return rows, histories


def sweep_table(rows) -> str:
    """Markdown table of the buffer_size sweep."""
    lines = ["| buffer_size | Retrains | Total retrain time | Mean retrain time | Final cumulative accuracy "
             "| Mean accuracy after drift | Peak buffer |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['buffer_size']:,} | {r['retrains']} | {r['total_ms']:.1f} ms | {r['mean_ms']:.2f} ms "
              f"| {r['final_accuracy']:.3f} | {r['after_drift']:.3f} | {r['peak_kb']:.1f} kB |" for r in rows]
    
    return "\n".join(lines)


def save_figures(rows, histories, drift_chunk, figure_dir: Path) -> list[str]:
    """Draw the three sweep figures; return their file names."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    sizes = [r["buffer_size"] for r in rows]

    ax = plot_metric_over_time(
        {f"buffer_size={s:,}": histories[s]["rolling_accuracy"] for s in PLOT_SIZES if s in histories},
        title="Rolling accuracy (last 10 chunks), drift marked", ylabel="Accuracy",
        events=[drift_chunk], events_label="concept drift", save_path=figure_dir / "rolling_accuracy.png")
    plt.close(ax.figure)

    for key, title, ylabel, name in (("total_ms", "Total retraining time", "Milliseconds", "retrain_time.png"),
                                     ("after_drift", "Mean accuracy after the drift", "Accuracy",
                                      "accuracy_after_drift.png")):
        ax = plot_metric_over_time([r[key] for r in rows], x=sizes, title=title, ylabel=ylabel, xlabel="buffer_size")
        ax.get_lines()[0].set(marker="o", markersize=6)        # one point per buffer size
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{s:,}" for s in sizes])
        ax.minorticks_off()
        ax.figure.savefig(figure_dir / name, dpi=150, bbox_inches="tight", facecolor=ax.figure.get_facecolor())
        plt.close(ax.figure)

    return ["rolling_accuracy.png", "retrain_time.png", "accuracy_after_drift.png"]


def run(quick: bool = False, output_dir: Path = Path(__file__).parent) -> str:
    """Run both benchmarks, save the figures; return the Markdown report."""
    cfg = QUICK if quick else FULL
    rng = np.random.default_rng(SEED)
    X, y = load_stream(cfg["step"])
    drift_chunk = DRIFT_ROW // cfg["step"] // cfg["chunk"]
    start = time.perf_counter()

    loop_rows = []
    for case, loop_func, vec_func, notes in loop_cases(cfg, X, y, rng):
        print(f"  {case} ...", flush=True)
        row = compare(case, loop_func, vec_func, repeat=cfg["repeat"], rtol=1e-6)
        row["notes"] = notes
        loop_rows.append(row)

    print("  buffer_size sweep ...", flush=True)
    sweep_rows, histories = sweep(X, y, cfg["buffer_sizes"], cfg["chunk"], cfg["sweep_repeat"], drift_chunk)
    figures = save_figures(sweep_rows, histories, drift_chunk, output_dir / "figures")
    elapsed = time.perf_counter() - start

    command = "python -m benchmark.run_stream_benchmarks" + (" --quick" if quick else "")
    lines = [
        "# Streaming benchmark results",
        "",
        f"Generated by `{command}` (seed {SEED}, {elapsed:.1f} s in total).",
        "",
        "## Environment",
        "",
        *[f"- {key}: {value}" for key, value in environment_info().items()],
        "",
        "## Loop vs vectorised",
        "",
        f"Median time per call over {cfg['repeat']} repeats. Both versions run the same algorithm and "
        "are checked to give the same result before they are timed; the speed-up is loop time / NumPy time.",
        "",
        format_table(loop_rows),
        "",
        "## buffer_size sweep",
        "",
        f"`demo/data/stream.csv` ({len(y):,} rows, concept drift at chunk {drift_chunk}) in chunks of "
        f"{cfg['chunk']}, through SimpleImputer -> StandardScaler -> DecisionTreeClassifier(max_depth=5), "
        "evaluated test-then-train with StreamTrainer (each chunk is scored before the model learns from it). "
        f"Retraining times are the median of {cfg['sweep_repeat']} run(s); rows left in the buffer at the "
        "end are not flushed. Peak buffer is the largest buffer logged after a chunk.",
        "",
        sweep_table(sweep_rows),
        "",
        "## Figures",
        "",
        *[f"![{name}](figures/{name})" for name in figures],
    ]

    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark the streaming tree.")
    parser.add_argument("--quick", action="store_true", help="small sizes for a fast check")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent,
                        help="folder for stream_results.md and figures/ (default: benchmark/)")
    args = parser.parse_args(argv)

    report = run(quick=args.quick, output_dir=args.output_dir)
    path = args.output_dir / "stream_results.md"
    path.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
