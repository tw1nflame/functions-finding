"""Полиномиальные функции первой, второй и третьей степеней."""

from collections.abc import Mapping

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike

from .._validation import FloatArray
from .base import BaseFunctionFamily, number


class PolynomialFamily(BaseFunctionFamily):
    """Полином: f(x) = a0 + a1*x + ... + an*x**n; допустимые степени — 1, 2, 3."""

    def __init__(self, degree: int = 1) -> None:
        if isinstance(degree, bool) or not isinstance(degree, int) or degree not in (1, 2, 3):
            raise ValueError("degree must be one of 1, 2, 3.")
        self.degree = degree
        self.name = f"polynomial_{degree}"
        self.parameter_names = tuple(f"a{i}" for i in range(degree + 1))

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        return np.polynomial.polynomial.polyval(x, params)

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Вернуть старт только со свободным членом; линейному МНК он не требуется."""
        return np.array([float(np.mean(y)), *([0.0] * self.degree)])

    @staticmethod
    def _coordinate(x: FloatArray) -> tuple[float, float]:
        return float(np.min(x) + np.ptp(x) / 2), float(np.ptp(x) / 2)

    def design_matrix(self, x: FloatArray) -> FloatArray:
        """Построить матрицу Вандермонда в центрированной и масштабированной координате."""
        center, scale = self._coordinate(x)
        return np.polynomial.polynomial.polyvander((x - center) / scale, self.degree)

    def linear_parameters(self, coefficients: FloatArray, x: FloatArray) -> FloatArray:
        """Перевести коэффициенты обратно к степеням исходного x."""
        center, scale = self._coordinate(x)
        converted = Polynomial(coefficients)(Polynomial([-center / scale, 1 / scale])).coef
        return np.pad(converted, (0, self.n_parameters - len(converted)))

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Вернуть полиномиальную формулу в исходной координате входа."""
        params = self.validate_parameters(params)
        terms = [number(params[0], latex)]
        for power, coefficient in enumerate(params[1:], 1):
            sign = "+" if coefficient >= 0 else "-"
            variable = "x" if power == 1 else (f"x^{{{power}}}" if latex else f"x**{power}")
            multiply = r"\," if latex else "*"
            terms.append(f"{sign} ({number(abs(coefficient), latex)}){multiply}{variable}")
        return "y = " + " ".join(terms)

    def expression(
        self,
        parameters: Mapping[str, str],
        *,
        variable: str = "t",
        initial: str = "y0",
        latex: bool = False,
    ) -> str:
        """Подставить зависимости коэффициентов от условий во временной полином."""
        terms = [f"({parameters['a0']})"]
        for power in range(1, self.degree + 1):
            factor = (
                variable
                if power == 1
                else (f"{variable}^{{{power}}}" if latex else f"{variable}**{power}")
            )
            multiply = r"\," if latex else "*"
            terms.append(f"({parameters[f'a{power}']}){multiply}{factor}")
        return " + ".join(terms)
