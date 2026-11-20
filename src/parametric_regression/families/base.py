"""Contract for stateless, extensible mathematical families."""

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray, as_vector


class BaseFunctionFamily(ABC):
    """Define a function, named parameters, bounds and training-only initial guesses.

    Subclasses implement ``_evaluate``, ``initial_guess`` and ``formula``.
    Validation and finite-output checks are shared by all built-in families.
    """

    name: str
    parameter_names: tuple[str, ...]
    positive_parameters: tuple[str, ...] = ()
    requires_initial_value: bool = False
    expression_template: str | None = None
    latex_expression_template: str | None = None

    def with_initial_value(self, value: float) -> "BaseFunctionFamily":
        """Return a family bound to an observed initial value, when supported."""
        raise ValueError(f"{self.name} does not use an initial value.")

    def expression(
        self,
        parameters: Mapping[str, str],
        *,
        variable: str = "t",
        initial: str = "y0",
        latex: bool = False,
    ) -> str:
        """Compose a right-hand-side expression without evaluating user strings.

        Optional stage-two extension: templates use @parameter@, @t@, @y0@.
        Custom families may override this method instead of supplying templates.
        """
        template = self.latex_expression_template if latex else self.expression_template
        if template is None:
            raise NotImplementedError(f"{self.name} needs an expression template for composition.")
        values = {**parameters, "t": variable, "y0": initial}
        return re.sub(r"@([^@]+)@", lambda match: values[match.group(1)], template)

    @property
    def n_parameters(self) -> int:
        """Number of free coefficients, including the intercept."""
        return len(self.parameter_names)

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Return inclusive lower/upper bounds in parameter_names order."""
        return np.full(self.n_parameters, -np.inf), np.full(self.n_parameters, np.inf)

    def validate_domain(self, x: FloatArray) -> None:
        """Raise ValueError if the finite x vector is outside the family domain."""
        return None

    def validate_parameters(self, params: ArrayLike) -> FloatArray:
        """Return a finite parameter vector satisfying the family bounds."""
        params = as_vector(params, "params")
        if params.size != self.n_parameters:
            raise ValueError(f"{self.name} requires {self.n_parameters} parameters.")
        lower, upper = self.parameter_bounds()
        if np.any(params < lower) or np.any(params > upper):
            raise ValueError(f"Parameters outside bounds for {self.name}.")
        return params

    def evaluate(self, x: ArrayLike, params: ArrayLike) -> FloatArray:
        """Evaluate at finite 1-D x; reject invalid parameters or numerical overflow.

        Returns an array with the same shape as x; an empty x is allowed.
        """
        x = as_vector(x, "x", allow_empty=True)
        params = self.validate_parameters(params)
        self.validate_domain(x)
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            result = np.asarray(self._evaluate(x, params), dtype=float)
        if result.shape != x.shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{self.name} returned non-finite values or an invalid shape.")
        return result

    @abstractmethod
    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        """Return function values for validated inputs."""

    @abstractmethod
    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Return one feasible parameter vector using training data only."""

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Return deterministic initial vectors for shared multi-start fitting."""
        return [self.initial_guess(x, y)]

    @abstractmethod
    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Return an explicit formula in x with numeric coefficients."""

    def design_matrix(self, x: FloatArray) -> FloatArray | None:
        """Return a linear design matrix, or None to use nonlinear optimization."""
        return None

    def identifiability_warnings(self, x: FloatArray, params: FloatArray) -> tuple[str, ...]:
        """Return optional family-specific warnings about practically inseparable parameters.

        Ordinary fits retain the warning. Stage two excludes such estimates when
        parameters were fitted, in addition to its existing Jacobian checks.
        """
        return ()

    def linear_parameters(self, coefficients: FloatArray, x: FloatArray) -> FloatArray:
        """Convert fitted linear-basis coefficients to public parameters."""
        return coefficients


def transition_guesses(x: FloatArray, y: FloatArray, fraction: float) -> list[FloatArray]:
    """Initialize c, L, k, x0 for monotonic transitions from training data."""
    order = np.argsort(x)
    span = float(np.ptp(x))
    low, high = float(np.min(y)), float(np.max(y))
    direction = 1.0 if y[order[-1]] >= y[order[0]] else -1.0
    center = float(x[np.argmin(np.abs(y - (low + fraction * (high - low))))])
    centers = [center, float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))]
    return [
        np.array([low, max(high - low, 1e-8), direction * rate / span, location])
        for rate in (4.0, 1.0, 10.0)
        for location in centers
    ]


def number(value: float, latex: bool = False) -> str:
    """Format a coefficient, including proper LaTeX scientific notation."""
    result = f"{value:.12g}"
    if latex and "e" in result:
        mantissa, exponent = result.split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return result
