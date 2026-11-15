"""Exponential growth/decay with an offset."""

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from .base import BaseFunctionFamily, number


class ExponentialFamily(BaseFunctionFamily):
    """f(x) = c + a*exp(b*x); parameters are (c, a, b)."""

    name = "exponential"
    parameter_names = ("c", "a", "b")
    expression_template = "(@c@) + (@a@)*exp((@b@)*@t@)"
    latex_expression_template = r"(@c@) + (@a@)\,\exp((@b@)@t@)"

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        c, a, b = params
        if a == 0:
            return np.full_like(x, c)
        # Compute the product in log space: small a can offset a large b*x.
        return c + np.sign(a) * np.exp(np.log(abs(a)) + b * x)

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Return the first deterministic exponential starting vector."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Estimate levels by linear least squares for six fixed trial rates."""
        span, center = float(np.ptp(x)), float(np.mean(x))
        guesses = []
        for rate in (-1.0, 1.0, -3.0, 3.0, -0.25, 0.25):
            b = rate / span
            basis = np.exp(b * (x - center))
            c, amplitude = np.linalg.lstsq(
                np.column_stack([np.ones_like(x), basis]), y, rcond=None
            )[0]
            with np.errstate(over="ignore", under="ignore"):
                a = amplitude * np.exp(-b * center)
            if np.isfinite(a):
                guesses.append(np.array([c, a, b]))
        return guesses

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Return the exponential formula with numeric coefficients."""
        c, a, b = (number(v, latex) for v in self.validate_parameters(params))
        if latex:
            return rf"y = {c} + ({a})\,e^{{({b})x}}"
        return f"y = {c} + ({a})*exp(({b})*x)"
