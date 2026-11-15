"""Independent experiment splits, trajectory preparation and honest curve scoring."""

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupShuffleSplit

from .._validation import FloatArray, validate_xy
from ..families import BaseFunctionFamily
from ..metrics import regression_metrics


@dataclass(frozen=True)
class ExperimentSplit:
    """Partition of every original row and independent experiment, created before fitting."""

    row_partitions: NDArray[np.str_]
    experiments: pd.DataFrame
    method: str
    random_state: int | None
    validation_size: float | None = None
    test_size: float | None = None


@dataclass
class Trajectory:
    """Prepared observations; held-out trajectories are never fitted."""

    identifier: Hashable
    experiment: Hashable
    partition: str
    positions: NDArray[np.int64]
    time: FloatArray
    target: FloatArray
    conditions: FloatArray
    initial_value: float | None = None
    time_origin: float = 0.0
    scored: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=bool))
    error: str = ""


def _identifier_column(df: pd.DataFrame, column: str) -> None:
    values = df[column]
    if values.isna().any():
        raise ValueError(f"Identifier column {column!r} contains missing values.")
    try:
        for value in values:
            hash(value)
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError(f"Identifier column {column!r} contains non-finite values.")
    except TypeError as error:
        raise ValueError(f"Identifier column {column!r} must contain hashable scalars.") from error


def split_experiments(
    df: pd.DataFrame,
    group_column: str,
    *,
    split_column: str | None,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> ExperimentSplit:
    """Use GroupShuffleSplit or explicit train/validation/test labels; no fitting."""
    _identifier_column(df, group_column)
    groups, identifiers = pd.factorize(df[group_column], sort=False)
    partitions = np.full(len(df), "train", dtype="<U10")
    if split_column is not None:
        labels = df[split_column]
        if labels.isna().any() or not labels.isin(["train", "validation", "test"]).all():
            raise ValueError("Split labels must be exactly 'train', 'validation' or 'test'.")
        if df.groupby(group_column, sort=False, observed=True)[split_column].nunique().max() != 1:
            raise ValueError("An independent experiment crosses split partitions (data leakage).")
        partitions = labels.to_numpy(dtype=str)
        if not {"train", "validation"}.issubset(set(partitions)):
            raise ValueError("Explicit split requires nonempty train and validation partitions.")
        method = "explicit"
    else:
        if len(identifiers) < (3 if test_size > 0 else 2):
            raise ValueError("Too few independent experiments for the requested split.")
        remaining = np.arange(len(df))
        try:
            if test_size > 0:
                splitter = GroupShuffleSplit(
                    n_splits=1, test_size=test_size, random_state=random_state
                )
                remaining, test = next(splitter.split(remaining, groups=groups))
                partitions[test] = "test"
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=validation_size / (1 - test_size), random_state=random_state
            )
            _, validation = next(splitter.split(remaining, groups=groups[remaining]))
            partitions[remaining[validation]] = "validation"
        except ValueError as error:
            raise ValueError(f"Cannot split independent experiments: {error}") from error
        method = "group_shuffle"
    records = []
    for code, identifier in enumerate(identifiers):
        positions = np.flatnonzero(groups == code)
        records.append(
            {
                "experiment_id": identifier,
                "partition": partitions[positions[0]],
                "n_observations": len(positions),
            }
        )
    partitions.setflags(write=False)
    return ExperimentSplit(
        partitions,
        pd.DataFrame(records),
        method,
        random_state if split_column is None else None,
        validation_size if split_column is None else None,
        test_size if split_column is None else None,
    )


def prepare_trajectories(
    df: pd.DataFrame,
    family: BaseFunctionFamily,
    *,
    time_column: str,
    target_column: str,
    group_column: str,
    condition_columns: Sequence[str],
    trajectory_column: str | None,
    split_column: str | None,
    initial_values: Mapping[Hashable, float] | None,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[list[Trajectory], ExperimentSplit]:
    """Validate schema and fixed conditions; retain invalid time/target trajectories as errors.

    For initial-state families, elapsed time starts at the earliest observation.
    y0 is supplied by trajectory ID or the mean of tied earliest measurements.
    Those baseline observations are excluded from every curve score.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or not df.columns.is_unique:
        raise ValueError("df must be a nonempty DataFrame with unique column names.")
    conditions = tuple(condition_columns)
    if not conditions or any(not isinstance(c, str) or not c for c in conditions):
        raise ValueError("condition_columns must contain nonempty string column names.")
    roles = [time_column, target_column, group_column, *conditions]
    if trajectory_column is not None and trajectory_column != group_column:
        roles.append(trajectory_column)
    if split_column is not None:
        roles.append(split_column)
    if len(set(roles)) != len(roles):
        raise ValueError(
            "Time, target, identifiers, conditions and split must use distinct columns."
        )
    missing = set(roles) - set(df.columns)
    if missing:
        raise ValueError(f"Missing DataFrame columns: {sorted(missing)}")
    for column in [time_column, target_column, *conditions]:
        dtype = df[column].dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise ValueError(f"Column {column!r} must be real numeric.")
    trajectory_column = trajectory_column or group_column
    _identifier_column(df, trajectory_column)
    if df.groupby(trajectory_column, sort=False, observed=True)[group_column].nunique().max() != 1:
        raise ValueError("Each trajectory_id must belong to exactly one independent experiment.")
    if initial_values is not None:
        if not family.requires_initial_value:
            raise ValueError("initial_values is only valid for families with an initial state.")
        unknown = set(initial_values) - set(df[trajectory_column])
        if unknown:
            raise ValueError(f"Unknown trajectory IDs in initial_values: {unknown}")
    protocol = split_experiments(
        df,
        group_column,
        split_column=split_column,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )
    trajectories = []
    for identifier, positions in df.groupby(
        trajectory_column, sort=False, observed=True
    ).indices.items():
        frame = df.iloc[positions]
        X = frame[list(conditions)].to_numpy(dtype=float, na_value=np.nan)
        # Time-dependent environments are a different model, not a data-cleaning case.
        for column, values in zip(conditions, X.T, strict=True):
            finite = values[np.isfinite(values)]
            if len(np.unique(finite)) > 1:
                raise ValueError(
                    f"Condition {column!r} varies within trajectory {identifier!r}; unsupported."
                )
        item = Trajectory(
            identifier,
            frame[group_column].iloc[0],
            str(protocol.row_partitions[positions[0]]),
            np.asarray(positions),
            frame[time_column].to_numpy(dtype=float, na_value=np.nan),
            frame[target_column].to_numpy(dtype=float, na_value=np.nan),
            X[0].copy(),
        )
        try:
            item.time, item.target = validate_xy(item.time, item.target)
            if not np.isfinite(X).all():
                raise ValueError("Conditions contain missing or non-finite values.")
            if np.unique(item.time).size < 2:
                raise ValueError("Trajectory needs at least two distinct time points.")
            item.scored = np.ones(len(item.time), dtype=bool)
            if family.requires_initial_value:
                item.time_origin = float(np.min(item.time))
                item.time = item.time - item.time_origin
                baseline = item.time == 0
                supplied = None if initial_values is None else initial_values.get(identifier)
                if supplied is not None:
                    if not np.isscalar(supplied) or not np.isfinite(supplied):
                        raise ValueError("Explicit initial value must be a finite scalar.")
                    item.initial_value = float(supplied)
                else:
                    item.initial_value = float(np.mean(item.target[baseline]))
                item.scored = ~baseline
        except (ValueError, FloatingPointError) as error:
            item.error = str(error)
        trajectories.append(item)
    return trajectories, protocol


@dataclass
class CurveEvaluation:
    """Curve errors with observation/macro-experiment aggregation and full coverage diagnostics."""

    observation: dict[str, float]
    experiment: dict[str, float]
    per_experiment: pd.DataFrame
    predictions: pd.DataFrame
    failures: pd.DataFrame
    complete: bool

    def summary(self) -> dict[str, object]:
        """Return metrics, coverage and failure count suitable for reporting."""
        return {
            "observation": dict(self.observation),
            "experiment": dict(self.experiment),
            "complete": self.complete,
            "n_experiments": len(self.per_experiment),
            "n_scored_observations": int(self.predictions["scored"].sum())
            if not self.predictions.empty
            else 0,
            "n_failed_trajectories": len(self.failures),
        }


def evaluate_curves(
    trajectories: Sequence[Trajectory],
    predictor: Callable[[Trajectory], FloatArray],
) -> CurveEvaluation:
    """Evaluate y predictions; any failed trajectory makes the score incomplete.

    Macro RMSE is the arithmetic mean of per-independent-experiment RMSEs.
    Each experiment pools its trajectories; conditioning baselines are excluded.
    Partial metrics remain visible but must not be used for model selection.
    """
    frames, failures = [], []
    for item in trajectories:
        try:
            if item.error:
                raise ValueError(item.error)
            prediction = np.asarray(predictor(item), dtype=float)
            validate_xy(item.target, prediction)
            if not item.scored.any():
                raise ValueError("No observations remain after excluding the initial state.")
            frames.append(
                pd.DataFrame(
                    {
                        "row_position": item.positions,
                        "trajectory_id": [item.identifier] * len(item.time),
                        "experiment_id": [item.experiment] * len(item.time),
                        "partition": item.partition,
                        "time": item.time,
                        "observed": item.target,
                        "predicted": prediction,
                        "scored": item.scored,
                    }
                )
            )
        except Exception as error:
            failures.append(
                {
                    "trajectory_id": item.identifier,
                    "experiment_id": item.experiment,
                    "partition": item.partition,
                    "message": f"{type(error).__name__}: {error}",
                }
            )
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    observation, experiment, records = {}, {}, []
    if not predictions.empty:
        scored = predictions[predictions["scored"]]
        try:
            observation = regression_metrics(scored["observed"], scored["predicted"])
            for identifier, group in scored.groupby("experiment_id", sort=False):
                records.append(
                    {
                        "experiment_id": identifier,
                        "n_observations": len(group),
                        **regression_metrics(group["observed"], group["predicted"]),
                    }
                )
            experiment = {
                metric: float(np.mean([row[metric] for row in records]))
                for metric in ("mae", "rmse", "r2")
            }
        except (ValueError, FloatingPointError) as error:
            failures.append(
                {
                    "trajectory_id": None,
                    "experiment_id": None,
                    "partition": None,
                    "message": f"Metric computation failed: {error}",
                }
            )
    return CurveEvaluation(
        observation,
        experiment,
        pd.DataFrame(records),
        predictions,
        pd.DataFrame(failures),
        bool(trajectories) and not failures and bool(observation),
    )
