"""Рациональное насыщение на неотрицательной полуоси."""

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from .base import BaseFunctionFamily, number


class RationalSaturationFamily(BaseFunctionFamily):
    """Рациональное насыщение: f(x) = c + a*x/(K+x); x >= 0, K > 0."""

    name = "rational_saturation"
    parameter_names = ("c", "a", "K")
    positive_parameters = ("K",)
    expression_template = "(@c@) + (@a@)*@t@/((@K@) + @t@)"
    latex_expression_template = r"(@c@) + \frac{(@a@)@t@}{(@K@)+@t@}"

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Использовать минимальное положительное нормальное float как нижнюю границу K."""
        return np.array([-np.inf, -np.inf, np.finfo(float).tiny]), np.full(3, np.inf)

    def validate_domain(self, x: FloatArray) -> None:
        """Отклонить отрицательные x перед вычислением или подбором семейства."""
        if np.any(x < 0):
            raise ValueError("rational_saturation requires x >= 0.")

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        c, a, saturation = params
        # Scaling the ratio prevents overflow in K+x and a*x.
        scale = np.maximum(x, saturation)
        return c + a * ((x / scale) / (saturation / scale + x / scale))

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Вернуть старт насыщения, оценённый по обучающим данным."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Проверить три характерных масштаба полунасыщения."""
        scale = float(np.median(x[x > 0]))
        guesses = []
        for factor in (1.0, 0.1, 10.0):
            saturation = scale * factor
            basis = x / (saturation + x)
            c, a = np.linalg.lstsq(np.column_stack([np.ones_like(x), basis]), y, rcond=None)[0]
            guesses.append(np.array([c, a, saturation]))
        return guesses

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Вернуть численную формулу рационального насыщения."""
        c, a, saturation = (number(v, latex) for v in self.validate_parameters(params))
        if latex:
            return rf"y = {c} + \frac{{({a})x}}{{{saturation}+x}}"
        return f"y = {c} + ({a})*x/(({saturation}) + x)"
