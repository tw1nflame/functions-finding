"""Экспоненциальная релаксация с неотрицательной задержкой."""

import numpy as np

from .._validation import FloatArray
from .relaxation import ExponentialRelaxationFamily


class DelayedExponentialFamily(ExponentialRelaxationFamily):
    """Релаксация с задержкой: y = y0 + A*(1-exp(-k*max(t-lag, 0))); k > 0, lag >= 0.

    y0 известно из наблюдения; A может иметь любой знак.
    При lag=0 получается обычная экспоненциальная релаксация.
    """

    name = "delayed_exponential"
    parameter_names = ("A", "k", "lag")
    positive_parameters = ("k",)
    expression_template = "@y0@ + (@A@)*(1-exp(-(@k@)*maximum(@t@-(@lag@), 0)))"
    latex_expression_template = r"@y0@+(@A@)(1-\exp(-(@k@)\max\{@t@-(@lag@),0\}))"

    def with_initial_value(self, value: float) -> "DelayedExponentialFamily":
        """Связать y0, сохранив тип семейства с задержкой."""
        return DelayedExponentialFamily(initial_value=value)

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Сохранить допустимость нулевой задержки; только k строго положителен."""
        return np.array([-np.inf, np.finfo(float).tiny, 0.0]), np.full(3, np.inf)

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        return super()._evaluate(np.maximum(x - params[2], 0.0), params[:2])

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Проверить задержки по наблюдаемому интервалу и линейно оценить амплитуды."""
        self.validate_domain(x)
        maximum = float(np.max(x))
        guesses = []
        for factor in (1.0, 0.2, 5.0):
            rate = factor / maximum
            for fraction in (0.0, 0.15, 0.35, 0.6):
                lag = maximum * fraction
                basis = -np.expm1(-rate * np.maximum(x - lag, 0.0))
                amplitude = np.linalg.lstsq(basis[:, None], y - self.initial_value, rcond=None)[0][
                    0
                ]
                guesses.append(np.array([amplitude, rate, lag]))
        return guesses
