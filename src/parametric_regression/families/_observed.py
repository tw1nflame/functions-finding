"""Shared binding of a known positive initial population, never fitted."""

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
        """Bind the known initial population on a fresh instance."""
        return type(self)(initial_value=value)

    def validate_domain(self, x: FloatArray) -> None:
        """Require a known positive initial population and nonnegative elapsed time."""
        if self.initial_value is None:
            raise ValueError("Bind initial_value before evaluating or fitting this family.")
        if np.any(x < 0):
            raise ValueError("Survival families require elapsed time >= 0.")

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Include the bound initial value, or an explicit y0 symbol, in the formula."""
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
