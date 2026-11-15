"""Constant coefficient dependency."""

import numpy as np

from .base import _LeastSquaresParameterModel


class ConstantParameterModel(_LeastSquaresParameterModel):
    """Learn a shared constant, or hold a supplied value in the model's target scale.

    For physical fixed time parameters, prefer MultiExperimentRegressor's
    fixed_parameters: it also fixes that coefficient during trajectory fitting.
    """

    name = "constant"
    degree = 0

    def __init__(self, value: float | None = None) -> None:
        if value is not None and (not np.isscalar(value) or not np.isfinite(value)):
            raise ValueError("value must be a finite scalar or None.")
        self.value = None if value is None else float(value)
