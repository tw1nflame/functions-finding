"""Общее связывание известной положительной начальной численности без её подбора."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from .base import BaseFunctionFamily, number


class _PositiveInitialFamily(BaseFunctionFamily):
    requires_initial_value = True

    def __init__(self, initial_value: float | None = None) -> None:
        if initial_value is not None:
            value = np.asarray(initial_value)
            if (
                value.shape != ()
                or value.dtype.kind not in "iuf"
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError("initial_value must be a finite strictly positive scalar.")
        self.initial_value = None if initial_value is None else float(initial_value)

    def with_initial_value(self, value: float) -> Self:
        """Связать известную начальную численность с новым экземпляром семейства."""
        return type(self)(initial_value=value)

    def validate_domain(self, x: FloatArray) -> None:
        """Потребовать известную положительную начальную численность и неотрицательное время."""
        if self.initial_value is None:
            raise ValueError("Bind initial_value before evaluating or fitting this family.")
        if np.any(x < 0):
            raise ValueError("Survival families require elapsed time >= 0.")

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Включить в формулу связанное начальное значение или явный символ y0."""
        values = (number(v, latex) for v in self.validate_parameters(params))
        initial = "y_0" if latex else "y0"
        if self.initial_value is not None:
            initial = number(self.initial_value, latex)
        return "y = " + self.expression(
            dict(zip(self.parameter_names, values, strict=True)),
            variable="x",
            initial=initial,
            latex=latex,
        )
