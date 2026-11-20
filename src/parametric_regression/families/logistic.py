"""Логистические переходы с устойчивыми вычислениями."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from .._validation import FloatArray
from .base import BaseFunctionFamily, number, transition_guesses


class LogisticFamily(BaseFunctionFamily):
    """Логистическая функция: f(x) = c + L*expit(k*(x-x0)); L >= 0, знак k произволен."""

    name = "logistic"
    parameter_names = ("c", "L", "k", "x0")
    expression_template = "(@c@) + (@L@)/(1 + exp(-(@k@)*(@t@ - (@x0@))))"
    latex_expression_template = r"(@c@) + \frac{(@L@)}{1+\exp(-(@k@)(@t@-(@x0@)))}"

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Неотрицательная L устраняет эквивалентную параметризацию с обращёнными знаками."""
        return np.array([-np.inf, 0, -np.inf, -np.inf]), np.full(4, np.inf)

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        c, amplitude, k, x0 = params
        return c + amplitude * expit(k * (x - x0))

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Оценить уровни, направление перехода и положение половины высоты."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Вернуть девять детерминированных стартов только по обучающим данным."""
        return transition_guesses(x, y, 0.5)

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Вернуть численную логистическую формулу."""
        c, amplitude, k, x0 = (number(v, latex) for v in self.validate_parameters(params))
        if latex:
            return rf"y = {c} + \frac{{{amplitude}}}{{1+e^{{-({k})(x-({x0}))}}}}"
        return f"y = {c} + ({amplitude})/(1 + exp(-({k})*(x - ({x0}))))"
