"""Weibull survival in the original, positive population scale."""

import numpy as np

from .._validation import FloatArray
from ._observed import _PositiveInitialFamily


class WeibullSurvivalFamily(_PositiveInitialFamily):
    """y(t) = y0*10**(-(t/delta)**p); delta, p, known y0 > 0.

    Targets are original counts, not log10(counts) and not optical density.
    At t=delta the prediction is y0/10. Underflow of the survival tail yields zero.
    """

    name = "weibull_survival"
    parameter_names = ("delta", "p")
    positive_parameters = parameter_names
    expression_template = "@y0@ * 10**(-(@t@/(@delta@))**(@p@))"
    latex_expression_template = r"@y0@\,10^{-(\frac{@t@}{(@delta@)})^{(@p@)}}"

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Both characteristic time and shape are strictly positive."""
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
        """Return the first training-derived survival guess."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Use a log-reduction regression for initialization, then fixed-shape starts."""
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
