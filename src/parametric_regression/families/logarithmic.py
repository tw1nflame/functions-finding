"""Logarithmic slowing with a fitted positive time scale."""

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from .base import BaseFunctionFamily, number


def logarithmic_basis(x: FloatArray, tau: float) -> FloatArray:
    """Compute log1p(x/tau), including ratios beyond the float64 range."""
    with np.errstate(over="ignore"):
        ratio = x / tau
    result = np.log1p(ratio)
    overflow = np.isinf(ratio)
    result[overflow] = np.log(x[overflow]) - np.log(tau)
    return result


class LogarithmicFamily(BaseFunctionFamily):
    """y(t) = c + a*log1p(t/tau), t >= 0 and tau > 0."""

    name = "logarithmic"
    parameter_names = ("c", "a", "tau")
    positive_parameters = ("tau",)
    expression_template = "(@c@) + (@a@)*log(1 + @t@/(@tau@))"
    latex_expression_template = r"(@c@) + (@a@)\ln(1+\frac{@t@}{(@tau@)})"

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Only the time scale is constrained."""
        return np.array([-np.inf, -np.inf, np.finfo(float).tiny]), np.full(3, np.inf)

    def validate_domain(self, x: FloatArray) -> None:
        """Require nonnegative time."""
        if np.any(x < 0):
            raise ValueError("Logarithmic family requires time >= 0.")

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        c, a, tau = params
        return c + a * logarithmic_basis(x, tau)

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Return the first time-scale/level guess."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Try four positive scales and estimate c/a linearly."""
        scale = float(np.median(x[x > 0]))
        guesses = []
        for factor in (1.0, 0.1, 10.0, 100.0):
            with np.errstate(over="ignore"):
                tau = factor * scale
            if not np.isfinite(tau) or tau <= 0:
                continue
            basis = logarithmic_basis(x, tau)
            c, a = np.linalg.lstsq(np.column_stack([np.ones_like(x), basis]), y, rcond=None)[0]
            guesses.append(np.array([c, a, tau]))
        return guesses

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Return the original-scale logarithmic expression."""
        values = (number(v, latex) for v in self.validate_parameters(params))
        return "y = " + self.expression(
            dict(zip(self.parameter_names, values, strict=True)), variable="x", latex=latex
        )
