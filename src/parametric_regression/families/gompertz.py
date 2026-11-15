"""Asymmetric Gompertz transitions."""

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from .base import BaseFunctionFamily, number, transition_guesses


class GompertzFamily(BaseFunctionFamily):
    """f(x) = c + L*exp(-exp(-k*(x-x0))); parameters (c, L, k, x0)."""

    name = "gompertz"
    parameter_names = ("c", "L", "k", "x0")
    expression_template = "(@c@) + (@L@)*exp(-exp(-(@k@)*(@t@ - (@x0@))))"
    latex_expression_template = r"(@c@) + (@L@)\,\exp(-\exp(-(@k@)(@t@-(@x0@))))"

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        c, amplitude, k, x0 = params
        z = -k * (x - x0)
        # For z > log(746), the outer exp underflows to zero in float64.
        shape = np.zeros_like(x)
        mask = z <= np.log(746.0)
        shape[mask] = np.exp(-np.exp(z[mask]))
        return c + amplitude * shape

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate a transition with positive amplitude."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Try both amplitude signs to cover both asymmetric orientations."""
        positive = transition_guesses(x, y, float(np.exp(-1)))
        negative = transition_guesses(x, -y, float(np.exp(-1)))
        for guess in negative:
            guess[:2] *= -1
        return [guess for pair in zip(positive, negative, strict=True) for guess in pair]

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Return a numeric Gompertz formula."""
        c, amplitude, k, x0 = (number(v, latex) for v in self.validate_parameters(params))
        if latex:
            return rf"y = {c} + ({amplitude})\,\exp(-\exp(-({k})(x-({x0}))))"
        return f"y = {c} + ({amplitude})*exp(-exp(-({k})*(x - ({x0}))))"
