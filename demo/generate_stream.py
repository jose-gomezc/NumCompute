"""Generate 20,000 synthetic rows for the streaming demo and benchmarks.

Columns: x0, x1, x2, x3, x4 (numeric features), segment (category), label (0/1).

- Abrupt concept drift at row 10,000: the label is ``x0 + x1 > 0`` before it
  and ``x2 - x4 > 0`` after it, so a model must retrain to recover.
- Gradual drift: the mean of x3 slides from 0 to 3 over the stream.
- About 3% of x2 is missing (empty cell), so a pipeline needs an imputer.
- About 5% of labels are flipped, so accuracy cannot reach 100%.
- segment is A, B or C; category D appears only after the drift (it is
  independent of the label and is used to show OneHotEncoder.partial_fit).

Run: python demo/generate_stream.py
Output: demo/data/stream.csv. Fixed seed 42 makes the file reproducible.
"""

from pathlib import Path

import numpy as np

N_ROWS = 20_000
DRIFT_ROW = 10_000
SEED = 42


def generate_stream(n_rows: int = N_ROWS, drift_row: int = DRIFT_ROW, seed: int = SEED) -> np.ndarray:
    """Return the rows as strings, shape (n_rows, 7), in the CSV column order."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, 5))
    X[:, 3] += np.linspace(0.0, 3.0, n_rows)                       # gradual drift in x3

    before = np.arange(n_rows) < drift_row
    label = np.where(before, X[:, 0] + X[:, 1] > 0, X[:, 2] - X[:, 4] > 0).astype(int)
    flip = rng.random(n_rows) < 0.05
    label[flip] = 1 - label[flip]                                  # label noise

    segment = np.where(before, rng.choice(["A", "B", "C"], n_rows),
                       rng.choice(["A", "B", "C", "D"], n_rows))

    text = np.char.mod("%.4f", X).astype("U16")
    text[rng.random(n_rows) < 0.03, 2] = ""                        # missing x2 after the label is set

    return np.column_stack([text, segment, label.astype(str)])


def main():
    """Write demo/data/stream.csv."""
    rows = generate_stream()
    destination = Path(__file__).resolve().parent / "data" / "stream.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, rows, fmt="%s", delimiter=",",
               header="x0,x1,x2,x3,x4,segment,label", comments="", encoding="utf-8")
    print(f"Generated {len(rows)} rows: {destination}")
    print(f"Drift at row {DRIFT_ROW}; missing x2 cells: {(rows[:, 2] == '').sum()}")


if __name__ == "__main__":
    main()
