"""Empirical power-law trends on nonnegative time."""

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from .base import BaseFunctionFamily, number


class PowerLawFamily(BaseFunctionFamily):
    """y(t) = c + a*t**b, t >= 0, b > 0; a and c are unrestricted."""

    name = "power_law"
    parameter_names = ("c", "a", "b")
    positive_parameters = ("b",)
    expression_template = "(@c@) + (@a@)*@t@**(@b@)"
    latex_expression_template = r"(@c@) + (@a@)@t@^{(@b@)}"

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Require a strictly positive exponent."""
        return np.array([-np.inf, -np.inf, np.finfo(float).tiny]), np.full(3, np.inf)

    def validate_domain(self, x: FloatArray) -> None:
        """Reject negative times, including for integral fitted exponents."""
        if np.any(x < 0):
            raise ValueError("Power law requires time >= 0.")

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        c, a, b = params
        result = np.full_like(x, c)
        if a != 0:
            positive = x > 0
            with np.errstate(over="ignore"):
                exponent = np.log(abs(a)) + b * np.log(x[positive])
            result[positive] += np.sign(a) * np.exp(exponent)
        return result

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Return the first training-only power/level guess."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Try six exponents and solve levels in a scaled time coordinate."""
        guesses = []
        scale = float(np.max(x))
        for b in (1.0, 0.5, 1.5, 2.0, 0.25, 3.0):
            basis = (x / scale) ** b
            c, amplitude = np.linalg.lstsq(
                np.column_stack([np.ones_like(x), basis]), y, rcond=None
            )[0]
            with np.errstate(over="ignore", under="ignore"):
                a = (
                    np.sign(amplitude) * np.exp(np.log(abs(amplitude)) - b * np.log(scale))
                    if amplitude != 0
                    else 0.0
                )
            if np.isfinite(a):
                guesses.append(np.array([c, a, b]))
        return guesses

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Return coefficients in the original time coordinate."""
        values = (number(v, latex) for v in self.validate_parameters(params))
        return "y = " + self.expression(
            dict(zip(self.parameter_names, values, strict=True)), variable="x", latex=latex
        )
