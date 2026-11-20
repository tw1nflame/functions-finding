"""Повторное использование первого этапа с возможной фиксацией физических коэффициентов."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from ..families import BaseFunctionFamily
from ..fitting import fit_family
from ..metrics import regression_metrics
from ..results import FitResult
from .validation import Trajectory


class _FixedParameterFamily(BaseFunctionFamily):
    """Передать существующему оптимизатору только свободные параметры через адаптер."""

    def __init__(self, family: BaseFunctionFamily, fixed: Mapping[str, float]) -> None:
        self.family = family
        self.fixed = dict(fixed)
        self.name = family.name
        self.indices = [i for i, name in enumerate(family.parameter_names) if name not in fixed]
        self.parameter_names = tuple(family.parameter_names[i] for i in self.indices)

    def expand(self, params: FloatArray) -> FloatArray:
        values = {**self.fixed, **dict(zip(self.parameter_names, params, strict=True))}
        return np.array([values[name] for name in self.family.parameter_names])

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        return self.family.evaluate(x, self.expand(params))

    def validate_domain(self, x: FloatArray) -> None:
        self.family.validate_domain(x)

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        lower, upper = self.family.parameter_bounds()
        return lower[self.indices], upper[self.indices]

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        return [guess[self.indices] for guess in self.family.initial_guesses(x, y)]

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        return self.family.formula(self.expand(np.asarray(params)), latex=latex)

    def identifiability_warnings(self, x: FloatArray, params: FloatArray) -> tuple[str, ...]:
        return self.family.identifiability_warnings(x, self.expand(params))


def fit_trajectory(
    family: BaseFunctionFamily,
    trajectory: Trajectory,
    fixed: Mapping[str, float],
    *,
    max_nfev: int,
    max_starts: int,
) -> FitResult:
    """Связать начальное состояние и фиксированные коэффициенты; обучить только по train."""
    if trajectory.partition != "train":
        raise ValueError("Only training trajectories may be fitted.")
    if trajectory.error:
        return FitResult(family, False, message=trajectory.error)
    bound = (
        family.with_initial_value(trajectory.initial_value)
        if family.requires_initial_value
        else family
    )
    try:
        if not fixed:
            return fit_family(
                bound, trajectory.time, trajectory.target, max_nfev=max_nfev, max_starts=max_starts
            )
        adapter = _FixedParameterFamily(bound, fixed)
        if not adapter.parameter_names:
            params = adapter.expand(np.array([]))
            prediction = bound.evaluate(trajectory.time, params)
            return FitResult(
                bound,
                True,
                params=params,
                train_metrics=regression_metrics(trajectory.target, prediction),
                message="All parameters fixed by the user.",
                jacobian_rank=0,
                jacobian_condition=1.0,
            )
        fitted = fit_family(
            adapter, trajectory.time, trajectory.target, max_nfev=max_nfev, max_starts=max_starts
        )
        params = None if fitted.params is None else adapter.expand(fitted.params)
        if params is not None:
            params.setflags(write=False)
        return replace(fitted, family=bound, params=params)
    except Exception as error:
        return FitResult(bound, False, message=f"{type(error).__name__}: {error}")
