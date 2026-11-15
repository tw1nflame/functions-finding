"""Matplotlib figures without global style changes or implicit show/save."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from ._validation import validate_xy
from .fitting import _positive_integer
from .results import FitResult, SelectionResult


def plot_fit(
    result: FitResult | SelectionResult,
    x: ArrayLike,
    y: ArrayLike,
    *,
    show_split: bool = True,
    top_n: int = 1,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
) -> Figure:
    """Return a figure with observations and one or more fitted curves.

    Supply x/y in the same row order used for selection when show_split=True.
    The selected model is always shown; remaining successful candidates are
    ordered by validation RMSE. Failed candidates are skipped. No refitting occurs.
    """
    x, y = validate_xy(x, y)
    _positive_integer(top_n, "top_n")
    if isinstance(result, SelectionResult):
        best = result.best_result
        others = sorted(
            (
                candidate
                for i, candidate in enumerate(result.candidates)
                if candidate.success and i != result.best_index
            ),
            key=lambda candidate: candidate.validation_metrics["rmse"],
        )
        candidates = [best, *others][:top_n]
    else:
        if not result.success:
            raise ValueError(f"Cannot plot failed fit: {result.message}")
        candidates = [result]
    protocol = result.split_protocol
    if show_split and protocol is not None:
        all_indices = np.concatenate([protocol.train_indices, protocol.validation_indices])
        if len(all_indices) != len(x) or not np.array_equal(
            np.sort(all_indices), np.arange(len(x))
        ):
            raise ValueError("x/y must match the original selection rows to show the split.")
    grid = np.linspace(float(x.min()), float(x.max()), 400)
    # Evaluate before opening a figure, so prediction failures don't leak figures.
    curves = [(candidate, candidate.predict(grid)) for candidate in candidates]
    figure, axes = plt.subplots(figsize=(9, 5), layout="constrained")
    if show_split and protocol is not None:
        for positions, label, marker in (
            (protocol.train_indices, "Train", "o"),
            (protocol.validation_indices, "Validation", "x"),
        ):
            axes.scatter(x[positions], y[positions], label=label, marker=marker, alpha=0.65)
    else:
        axes.scatter(x, y, label="Observations", alpha=0.65)
    for index, (candidate, prediction) in enumerate(curves):
        label = candidate.family_name + (" (selected)" if index == 0 else "")
        axes.plot(grid, prediction, label=label, linewidth=2 if index == 0 else 1.2)
    axes.set(xlabel=xlabel, ylabel=ylabel, title=title or "Parametric regression")
    axes.grid(alpha=0.2)
    axes.legend()
    return figure
