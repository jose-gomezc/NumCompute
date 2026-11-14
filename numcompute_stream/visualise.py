"""Reusable matplotlib plots for streaming results.

Each function draws on ``ax`` (or on a new figure), returns the Axes, and can
save the figure (``save_path``) or show it (``show=True``); in a notebook the
figure displays inline. Styling is applied to each Axes, so importing this
module never changes matplotlib's global settings.

Line colours follow the Okabe-Ito colour-blind-safe palette (Okabe & Ito, 2008).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .metrics import _check_labels, _check_same_family, _check_same_length, confusion_matrix

# All that can be imported from this module
__all__ = [
    "PALETTE",
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
]

# Okabe-Ito order; the first four were checked as neighbouring line colours
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")

ERROR = "#d03b3b"          # reserved for wrong predictions, never a series colour
_TEXT, _TEXT_2, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
_LINE_WIDTH = 1.5          # points: about 2 px on screen
_BLUES = LinearSegmentedColormap.from_list("okabe_blue", ["#eef4fa", "#0072B2", "#003a5c"])


def plot_metric_over_time(metric_values, title: str = "Metric over time", ylabel: str = "Value",
                          xlabel: str = "Chunk", events=None, events_label: str = "event",
                          ax=None, save_path=None, show: bool = False):
    """Plot one or more metrics across chunks.

    Parameters
    ----------
    metric_values : sequence of float, or dict of str to sequence
        One value per chunk (NaN leaves a gap), or several named series
        drawn as separate lines (at most 8).
    title, ylabel, xlabel : str
        Plot title and axis labels.
    events : sequence of bool or of int, optional
        Chunks to mark with a thin vertical line, as a boolean mask (e.g.
        ``history_["retrained"]``) or as chunk indices.
    events_label : str, default "event"
        Legend entry for the event lines.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created when omitted.
    save_path : str or path, optional
        Save the figure there; the format follows the extension.
    show : bool, default False
        Call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib Axes
        The Axes drawn on.

    Raises
    ------
    TypeError
        If a series is not numeric.
    ValueError
        If a series is empty or not 1-D, or there are more than 8 series.

    Examples
    --------
    >>> ax = plot_metric_over_time([0.8, 0.85, 0.9], title="Accuracy", ylabel="Accuracy")
    >>> len(ax.get_lines())
    1
    """
    series = _as_series(metric_values, ylabel)

    if len(series) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} series can be told apart, got {len(series)}")

    ax = _get_axes(ax)

    for (label, values), colour in zip(series.items(), PALETTE):
        ax.plot(np.arange(values.size), values, color=colour, linewidth=_LINE_WIDTH,
                label=label, solid_capstyle="round", solid_joinstyle="round")

    if events is not None:
        events = np.asarray(events)
        positions = np.flatnonzero(events) if events.dtype == bool else events.ravel()

        for i, x in enumerate(positions):
            ax.axvline(x, color=_MUTED, linewidth=0.8, linestyle=(0, (2, 3)),
                       label=events_label if i == 0 else None)

    legend = len(series) > 1 or (events is not None and len(positions) > 0)
    
    return _finish(ax, title, xlabel, ylabel, legend, save_path, show)


def compare_models(metric1, metric2, labels=("Model A", "Model B"), title: str = "Model comparison",
                   ylabel: str = "Value", xlabel: str = "Chunk", ax=None, save_path=None,
                   show: bool = False):
    """Plot the same metric for two models across chunks.

    Parameters
    ----------
    metric1, metric2 : sequence of float
        One value per chunk for each model; the lengths may differ.
    labels : pair of str, default ("Model A", "Model B")
        Legend names of the two models, e.g. ``("buffer_size=100", "buffer_size=500")``.
    title, ylabel, xlabel : str
        Plot title and axis labels.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created when omitted.
    save_path : str or path, optional
        Save the figure there; the format follows the extension.
    show : bool, default False
        Call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib Axes
        The Axes drawn on.

    Raises
    ------
    TypeError
        If a series is not numeric.
    ValueError
        If ``labels`` does not hold two different names, or a series is
        empty or not 1-D.

    Notes
    -----
    For more than two models, pass a dict to :func:`plot_metric_over_time`.
    """
    labels = list(labels)

    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError(f"labels must hold 2 different names, got {labels}")

    return plot_metric_over_time({labels[0]: metric1, labels[1]: metric2}, title=title,
                                 ylabel=ylabel, xlabel=xlabel, ax=ax, save_path=save_path, show=show)


def plot_predictions_vs_ground_truth(y_true, y_pred, kind: str = "sequence", title=None,
                                     ax=None, save_path=None, show: bool = False):
    """Show predicted against true labels, e.g. for the latest chunk.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        True and predicted class labels (int, float, bool or str).
    kind : {"sequence", "confusion"}, default "sequence"
        ``"sequence"``: one column per sample, true labels as hollow circles,
        predictions as dots, wrong predictions marked with a red cross.
        ``"confusion"``: a confusion-matrix heatmap with the counts written in.
    title : str, optional
        Default: ``"Latest chunk: <correct>/<n> correct"``.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created when omitted.
    save_path : str or path, optional
        Save the figure there; the format follows the extension.
    show : bool, default False
        Call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib Axes
        The Axes drawn on.

    Raises
    ------
    TypeError
        If the labels have unsupported or incompatible types.
    ValueError
        If ``kind`` is unknown, or the inputs are empty, not 1-D, of
        different lengths, or contain NaN.
    """
    if kind not in ("sequence", "confusion"):
        raise ValueError(f"kind must be 'sequence' or 'confusion', got {kind!r}")

    y_true = _check_labels(y_true, "y_true")
    y_pred = _check_labels(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))
    _check_same_family(y_true, y_pred, ("y_true", "y_pred"))

    classes = np.unique(np.concatenate([y_true, y_pred]))
    wrong = y_true != y_pred

    if title is None:
        title = f"Latest chunk: {int((~wrong).sum())}/{y_true.size} correct"

    ax = _get_axes(ax)

    if kind == "sequence":
        idx = np.arange(y_true.size)
        true_row, pred_row = np.searchsorted(classes, y_true), np.searchsorted(classes, y_pred)
        ax.scatter(idx, true_row, s=64, facecolors="none", edgecolors=PALETTE[0], linewidths=1.5,
                   label="true", zorder=2)
        ax.scatter(idx, pred_row, s=20, color=PALETTE[1], label="predicted", zorder=3)
        ax.scatter(idx[wrong], pred_row[wrong], s=80, marker="x", color=ERROR, linewidths=2,
                   label="wrong", zorder=4)
        ax.set_yticks(np.arange(classes.size), [str(c) for c in classes])
        ax.set_ylim(-0.6, classes.size - 0.4)
        return _finish(ax, title, "Sample in chunk", "Class", True, save_path, show)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    ax.imshow(cm, cmap=_BLUES, vmin=0)

    for (i, j), count in np.ndenumerate(cm):       # dark cells get white text
        ax.text(j, i, str(count), ha="center", va="center", fontsize=10,
                color="#ffffff" if count > cm.max() / 2 else _TEXT)

    ticks = np.arange(classes.size)
    ax.set_xticks(ticks, [str(c) for c in classes])
    ax.set_yticks(ticks, [str(c) for c in classes])
    ax = _finish(ax, title, "Predicted label", "True label", False, save_path=None, show=False)
    ax.grid(False)
    ax.spines[:].set_visible(False)                  # the cells already frame the matrix

    return _save_show(ax, save_path, show)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_series(metric_values, default_label: str) -> dict:
    """Return ``{label: 1-D float array}`` from one sequence or a dict of sequences."""
    if isinstance(metric_values, dict):
        if not metric_values:
            raise ValueError("metric_values is empty")
        items = metric_values.items()
    else:
        items = [(default_label, metric_values)]

    series = {}

    for label, values in items:
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise TypeError(f"series {label!r} must be numeric") from None

        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"series {label!r} must be a non-empty 1-D sequence, got shape {values.shape}")

        series[label] = values

    return series


def _get_axes(ax):
    """Return ``ax``, or the Axes of a new figure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.set_facecolor(_SURFACE)

    return ax


def _finish(ax, title, xlabel, ylabel, legend, save_path, show):
    """Apply the shared style, titles and legend, then save and/or show."""
    ax.set_facecolor(_SURFACE)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)

    ax.tick_params(colors=_TEXT_2, labelsize=9)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=_TEXT, fontsize=12)
    ax.set_xlabel(xlabel, color=_TEXT_2)
    ax.set_ylabel(ylabel, color=_TEXT_2)

    if legend:
        ax.legend(frameon=False, labelcolor=_TEXT, fontsize=9)

    return _save_show(ax, save_path, show)


def _save_show(ax, save_path, show):
    """Save the Axes' figure if a path is given, and show it if asked."""
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=ax.figure.get_facecolor())

    if show:
        plt.show()

    return ax
