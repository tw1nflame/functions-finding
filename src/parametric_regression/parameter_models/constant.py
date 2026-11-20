"""Постоянная зависимость коэффициента."""

import numpy as np

from .base import _LeastSquaresParameterModel


class ConstantParameterModel(_LeastSquaresParameterModel):
    """Оценить общую константу или удерживать заданное значение в целевой шкале модели.

    Для известных физических временных параметров предпочтительнее fixed_parameters
    в MultiExperimentRegressor: они фиксируют коэффициент также при подгонке траекторий.
    """

    name = "constant"
    degree = 0

    def __init__(self, value: float | None = None) -> None:
        if value is not None and (not np.isscalar(value) or not np.isfinite(value)):
            raise ValueError("value must be a finite scalar or None.")
        self.value = None if value is None else float(value)
