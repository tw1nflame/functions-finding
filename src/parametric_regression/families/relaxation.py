"""Exponential relaxation with an observed, non-fitted initial value."""

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from .base import BaseFunctionFamily, number


class ExponentialRelaxationFamily(BaseFunctionFamily):
    """y(t) = y0 + A*(1-exp(-k*t)), t >= 0, k > 0; A may have either sign.

    initial_value binds an observed y0 for standalone fit_family. The multi-
    experiment regressor binds a separate initial value for each trajectory.
    """

    name = "exponential_relaxation"
    parameter_names = ("A", "k")
    positive_parameters = ("k",)
    requires_initial_value = True
    expression_template = "@y0@ + (@A@)*(1 - exp(-(@k@)*@t@))"
    latex_expression_template = r"@y0@ + (@A@)\,(1-\exp(-(@k@)@t@))"

    def __init__(self, initial_value: float | None = None) -> None:
        if initial_value is not None and (
            not np.isscalar(initial_value) or not np.isfinite(initial_value)
        ):
            raise ValueError("initial_value must be a finite scalar.")
        self.initial_value = None if initial_value is None else float(initial_value)

    def with_initial_value(self, value: float) -> "ExponentialRelaxationFamily":
        """Return a new bound family, leaving this template unchanged."""
        return ExponentialRelaxationFamily(initial_value=value)

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """A is unrestricted; k is strictly positive."""
        return np.array([-np.inf, np.finfo(float).tiny]), np.full(2, np.inf)

    def validate_domain(self, x: FloatArray) -> None:
        """Require elapsed nonnegative time and an explicit initial state."""
        if self.initial_value is None:
            raise ValueError("Bind initial_value before evaluating or fitting relaxation.")
        if np.any(x < 0):
            raise ValueError("Relaxation requires elapsed time >= 0.")

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        amplitude, rate = params
        # A large nonnegative rate*time is a finite saturated response, not overflow.
        with np.errstate(over="ignore"):
            exponent = -rate * x
        return self.initial_value - amplitude * np.expm1(exponent)

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Return the first training-only rate/amplitude guess."""
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Try four rates and estimate amplitude linearly for each rate."""
        self.validate_domain(x)
        guesses = []
        for factor in (1.0, 0.2, 5.0, 20.0):
            rate = factor / float(np.max(x))
            basis = -np.expm1(-rate * x)
            amplitude = np.linalg.lstsq(basis[:, None], y - self.initial_value, rcond=None)[0][0]
            guesses.append(np.array([amplitude, rate]))
        return guesses

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Return a numeric formula with the observed initial state."""
        values = self.validate_parameters(params)
        initial = "y_0" if latex else "y0"
        if self.initial_value is not None:
            initial = number(self.initial_value, latex)
        return "y = " + self.expression(
            dict(zip(self.parameter_names, (number(v, latex) for v in values), strict=True)),
            variable="x",
            initial=initial,
            latex=latex,
        )
