"""Figures for trajectory fits, coefficient dependencies and held-out predictions."""

from collections.abc import Hashable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .._validation import as_vector, validate_xy
from ..fitting import _positive_integer
from .results import Conditions, MultiExperimentResult


def plot_experiments(
    result: MultiExperimentResult,
    *,
    partition: str = "train",
    trajectory_ids: Sequence[Hashable] | None = None,
    max_trajectories: int = 6,
) -> Figure:
    """Show observations, available local train fits and the general-model curves.

    Uses stored trajectories; no fitting is performed for validation/test plots.
    Failed trajectories are annotated. Returns a Figure without showing/saving it.
    """
    _positive_integer(max_trajectories, "max_trajectories")
    if partition not in ("train", "validation", "test"):
        raise ValueError("partition must be train, validation or test.")
    items = [item for item in result._trajectories if item.partition == partition]
    if trajectory_ids is not None:
        requested = set(trajectory_ids)
        if requested - {item.identifier for item in items}:
            raise ValueError("Requested trajectory IDs are not in this partition.")
        items = [item for item in items if item.identifier in requested]
    items = items[:max_trajectories]
    if not items:
        raise ValueError("No trajectories to plot in this partition.")
    columns = min(2, len(items))
    rows = int(np.ceil(len(items) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 3.5 * rows), squeeze=False, layout="constrained"
    )
    for axes_item, item in zip(axes.flat, items, strict=False):
        axes_item.set(
            title=f"{item.identifier} ({partition})",
            xlabel="Elapsed time" if result.time_family.requires_initial_value else "Time",
            ylabel="Response",
        )
        if item.error:
            axes_item.text(0.05, 0.5, item.error, transform=axes_item.transAxes, wrap=True)
            continue
        axes_item.scatter(item.time, item.target, label="Observed", alpha=0.6, s=18)
        grid = np.linspace(item.time.min(), item.time.max(), 200)
        local = result.train_fits.get(item.identifier)
        if local is not None and local.success:
            axes_item.plot(grid, local.predict(grid), "--", label="Individual train fit")
        if result.success:
            try:
                values = result.predict(
                    grid,
                    dict(zip(result.condition_columns, item.conditions, strict=True)),
                    initial_value=item.initial_value,
                    warn_extrapolation=False,
                )
                axes_item.plot(grid, values, label="General model")
            except (ValueError, FloatingPointError) as error:
                axes_item.text(0.05, 0.1, str(error), transform=axes_item.transAxes, wrap=True)
        axes_item.legend(fontsize="small")
        axes_item.grid(alpha=0.2)
    for unused in list(axes.flat)[len(items) :]:
        unused.set_visible(False)
    return figure


def plot_parameter_dependency(
    result: MultiExperimentResult,
    parameter: str,
    factor: str,
    *,
    reference_conditions: Mapping[str, float] | None = None,
) -> Figure:
    """Show train estimates and a model slice along one environmental factor.

    Other factors are held at supplied values or training medians. Scatter points
    include all observed conditions and therefore need not lie on that slice.
    """
    result._require_fit()
    if parameter not in result.parameter_models or factor not in result.condition_columns:
        raise ValueError("Unknown parameter or condition factor.")
    others = set(result.condition_columns) - {factor}
    if reference_conditions is not None and set(reference_conditions) != others:
        raise ValueError(f"reference_conditions must specify exactly {others}.")
    reference = (
        dict(reference_conditions)
        if reference_conditions is not None
        else {
            name: float(np.median(result.training_conditions[:, i]))
            for i, name in enumerate(result.condition_columns)
            if name != factor
        }
    )
    grid = np.linspace(*result.condition_ranges[factor], 200)
    conditions = pd.DataFrame(
        {**{name: np.full(len(grid), value) for name, value in reference.items()}, factor: grid}
    )
    values = result.predict_parameters(conditions)[parameter].to_numpy()
    table = result.experiment_parameters
    fitted = table[(table["partition"] == "train") & table["fit_success"]]
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for eligible, label, marker in (
        (True, "Train estimates (all conditions)", "o"),
        (False, "Excluded estimates", "x"),
    ):
        points = fitted[fitted["used_for_parameter_models"] == eligible]
        if not points.empty:
            axes.scatter(
                points[f"condition:{factor}"],
                points[f"param:{parameter}"],
                label=label,
                marker=marker,
                alpha=0.7,
            )
    slice_label = "Model slice" + (
        ": " + ", ".join(f"{k}={v:.4g}" for k, v in reference.items()) if reference else ""
    )
    axes.plot(grid, values, label=slice_label)
    axes.set(xlabel=factor, ylabel=parameter, title=f"{parameter} dependency")
    axes.legend(fontsize="small")
    axes.grid(alpha=0.2)
    return figure


def plot_new_trajectory(
    result: MultiExperimentResult,
    time: ArrayLike,
    conditions: Conditions,
    *,
    initial_value: float | ArrayLike | None = None,
    truth: ArrayLike | None = None,
) -> Figure:
    """Plot prediction for supplied conditions, optionally compared to a known true curve."""
    time = as_vector(time, "time")
    prediction = result.predict(time, conditions, initial_value=initial_value)
    if truth is not None:
        _, truth = validate_xy(time, truth)
    order = np.argsort(time)
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.plot(time[order], prediction[order], label="General-model prediction")
    if truth is not None:
        axes.plot(time[order], truth[order], "--", label="True curve")
    axes.set(
        xlabel="Elapsed time" if result.time_family.requires_initial_value else "Time",
        ylabel="Response",
        title="Prediction for new conditions",
    )
    axes.grid(alpha=0.2)
    axes.legend()
    return figure


def plot_prediction_comparison(
    result: MultiExperimentResult,
    *,
    partition: str = "validation",
) -> Figure:
    """Plot observed versus predicted held-out y, excluding conditioning baselines."""
    if partition not in ("validation", "test") or partition not in result.evaluations:
        raise ValueError("Choose an evaluated validation or test partition.")
    evaluation = result.evaluations[partition]
    if evaluation.predictions.empty:
        raise ValueError("No predictions are available for this partition.")
    frame = evaluation.predictions[evaluation.predictions["scored"]]
    figure, axes = plt.subplots(figsize=(6, 6), layout="constrained")
    for identifier, points in frame.groupby("experiment_id", sort=False):
        axes.scatter(
            points["observed"], points["predicted"], label=str(identifier), alpha=0.6, s=20
        )
    low = float(frame[["observed", "predicted"]].min().min())
    high = float(frame[["observed", "predicted"]].max().max())
    axes.plot([low, high], [low, high], "k--", label="Identity")
    title = f"{partition}: observed vs predicted"
    if not evaluation.complete:
        title += " (incomplete coverage)"
    axes.set(xlabel="Observed", ylabel="Predicted", title=title)
    axes.legend(fontsize="small")
    axes.grid(alpha=0.2)
    return figure
