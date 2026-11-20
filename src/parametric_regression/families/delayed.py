"""Exponential relaxation following a nonnegative lag."""

import numpy as np

from .._validation import FloatArray
from .relaxation import ExponentialRelaxationFamily


class DelayedExponentialFamily(ExponentialRelaxationFamily):
    """y = y0 + A*(1-exp(-k*max(t-lag, 0))); k > 0, lag >= 0.

    y0 is observed, A may have either sign. lag=0 recovers ordinary relaxation.
    """

    name = "delayed_exponential"
    parameter_names = ("A", "k", "lag")
    positive_parameters = ("k",)
    expression_template = "@y0@ + (@A@)*(1-exp(-(@k@)*maximum(@t@-(@lag@), 0)))"
    latex_expression_template = r"@y0@+(@A@)(1-\exp(-(@k@)\max\{@t@-(@lag@),0\}))"

    def with_initial_value(self, value: float) -> "DelayedExponentialFamily":
        """Bind y0 without dropping the delayed-family type."""
        return DelayedExponentialFamily(initial_value=value)

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Preserve the zero-lag boundary; only k is strictly positive."""
        return np.array([-np.inf, np.finfo(float).tiny, 0.0]), np.full(3, np.inf)

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        return super()._evaluate(np.maximum(x - params[2], 0.0), params[:2])

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Try delays across the observed interval and fit amplitudes linearly."""
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
