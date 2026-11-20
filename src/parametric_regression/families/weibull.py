"""Выживание по Вейбуллу в исходной положительной шкале численности."""

import numpy as np

from .._validation import FloatArray
from ._observed import _PositiveInitialFamily


class WeibullSurvivalFamily(_PositiveInitialFamily):
    """Модель Вейбулла: y(t) = y0*10**(-(t/delta)**p); delta, p и известное y0 > 0.

    Целевая величина — исходная численность, а не log10 численности и не оптическая плотность.
    При t=delta прогноз равен y0/10. Исчезающий ниже диапазона float64 хвост равен нулю.
    """

    name = "weibull_survival"
    parameter_names = ("delta", "p")
    positive_parameters = parameter_names
    expression_template = "@y0@ * 10**(-(@t@/(@delta@))**(@p@))"
    latex_expression_template = r"@y0@\,10^{-(\frac{@t@}{(@delta@)})^{(@p@)}}"

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Характерное время и параметр формы строго положительны."""
        return np.full(2, np.finfo(float).tiny), np.full(2, np.inf)

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        delta, p = params
        result = np.full_like(x, self.initial_value)
        positive = x > 0
        with np.errstate(over="ignore"):
            log_reduction = np.log(np.log(10.0)) + p * (np.log(x[positive]) - np.log(delta))
        log_initial = np.log(self.initial_value)
        maximum_reduction = log_initial - np.log(np.nextafter(0.0, 1.0)) + 2.0
        representable = log_reduction < np.log(maximum_reduction)
        tail = np.zeros_like(log_reduction)
        tail[representable] = np.exp(log_initial - np.exp(log_reduction[representable]))
        result[positive] = tail
        return result

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Вернуть первую оценку параметров выживания по обучающим данным."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Начать с регрессии логарифма лог-снижения, затем проверить фиксированные формы."""
        self.validate_domain(x)
        valid = (x > 0) & (y > 0) & (y < self.initial_value)
        guesses = []
        if np.count_nonzero(valid) >= 2 and np.unique(x[valid]).size >= 2:
            reduction = (np.log(self.initial_value) - np.log(y[valid])) / np.log(10.0)
            intercept, p = np.linalg.lstsq(
                np.column_stack([np.ones(np.count_nonzero(valid)), np.log(x[valid])]),
                np.log(reduction),
                rcond=None,
            )[0]
            if p > 0:
                with np.errstate(over="ignore", under="ignore"):
                    delta = np.exp(-intercept / p)
                if np.isfinite(delta) and delta >= np.finfo(float).tiny:
                    guesses.append(np.array([delta, p]))
        delta = float(np.median(x[x > 0]))
        guesses.extend(np.array([delta, p]) for p in (1.0, 0.5, 1.5, 2.5))
        return guesses
