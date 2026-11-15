"""Logistic transitions with stable evaluation."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from .._validation import FloatArray
from .base import BaseFunctionFamily, number, transition_guesses


class LogisticFamily(BaseFunctionFamily):
    """f(x) = c + L*expit(k*(x-x0)); L >= 0, either sign of k."""

    name = "logistic"
    parameter_names = ("c", "L", "k", "x0")
    expression_template = "(@c@) + (@L@)/(1 + exp(-(@k@)*(@t@ - (@x0@))))"
    latex_expression_template = r"(@c@) + \frac{(@L@)}{1+\exp(-(@k@)(@t@-(@x0@)))}"

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Nonnegative L removes the equivalent sign-reversed parameterization."""
        return np.array([-np.inf, 0, -np.inf, -np.inf]), np.full(4, np.inf)

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        c, amplitude, k, x0 = params
        return c + amplitude * expit(k * (x - x0))

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Estimate the levels, direction and half-height location."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Return nine deterministic training-only starts."""
        return transition_guesses(x, y, 0.5)

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Return a numeric logistic formula."""
        c, amplitude, k, x0 = (number(v, latex) for v in self.validate_parameters(params))
        if latex:
            return rf"y = {c} + \frac{{{amplitude}}}{{1+e^{{-({k})(x-({x0}))}}}}"
        return f"y = {c} + ({amplitude})/(1 + exp(-({k})*(x - ({x0}))))"
