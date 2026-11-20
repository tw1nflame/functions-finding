"""Concentration-response saturation in the incoming coefficient target scale."""

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from .._validation import FloatArray
from ..families import BaseFunctionFamily
from ._specialized import _OneFactorModel, curve_formula


class _HillCurve(BaseFunctionFamily):
    name = "hill_curve"
    parameter_names = ("c", "A", "K", "h")
    expression_template = "(@c@) + (@A@)*@t@**(@h@)/((@K@)**(@h@) + @t@**(@h@))"
    latex_expression_template = r"(@c@)+(@A@)\frac{@t@^{(@h@)}}{(@K@)^{(@h@)}+@t@^{(@h@)}}"

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        return np.array([-np.inf, -np.inf, np.finfo(float).tiny, np.finfo(float).tiny]), np.full(
            4, np.inf
        )

    def validate_domain(self, x: FloatArray) -> None:
        if np.any(x < 0):
            raise ValueError("Hill requires concentration >= 0.")

    @staticmethod
    def basis(x: FloatArray, K: float, h: float) -> FloatArray:
        result = np.zeros_like(x)
        positive = x > 0
        with np.errstate(over="ignore"):
            result[positive] = expit(h * (np.log(x[positive]) - np.log(K)))
        return result

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        c, A, K, h = params
        return c + A * self.basis(x, K, h)

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        guesses = []
        for h in (1.0, 2.0, 0.5, 4.0):
            for K in np.quantile(x[x > 0], [0.25, 0.5, 0.75]):
                c, A = np.linalg.lstsq(
                    np.column_stack([np.ones_like(x), self.basis(x, K, h)]), y, rcond=None
                )[0]
                guesses.append(np.array([c, A, K, h]))
        return guesses

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        return curve_formula(self, params, latex)


class HillParameterModel(_OneFactorModel):
    """g(C) = c + A*C**h/(K**h+C**h), C >= 0, K/h > 0, signed A.

    Operates in the incoming target scale. If the surrounding relationship uses
    log(k), the physical prediction is exp(g(C)), not an ordinary Hill law for k.
    At least four distinct concentration levels are needed; more are preferable.
    """

    name = "hill"
    n_coefficients = 4

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        feature_names: Sequence[str] | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Use shared multi-start fitting, honoring weights and identifiability checks."""
        x, target, weights = self._prepare_fit(X, y, feature_names, sample_weight)
        self._fit_curve(_HillCurve(), x, target, weights)
        if not np.min(x[x > 0]) <= self.parameters_["K"] <= np.max(x):
            self.warnings_ += (
                "Hill K lies outside observed positive concentrations; "
                "saturation is poorly covered.",
            )
        return self
