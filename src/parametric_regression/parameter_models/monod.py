"""Physical nonnegative Monod growth rate; exact zero at zero substrate."""

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from ..families import RationalSaturationFamily
from ._specialized import _OneFactorModel, curve_formula


class _MonodCurve(RationalSaturationFamily):
    name = "monod_curve"
    parameter_names = ("mu_max", "K_s")
    positive_parameters = parameter_names
    expression_template = "(@mu_max@)*@t@/((@K_s@) + @t@)"
    latex_expression_template = r"(@mu_max@)\frac{@t@}{(@K_s@)+@t@}"

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        return np.full(2, np.finfo(float).tiny), np.full(2, np.inf)

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        return super()._evaluate(x, np.array([0.0, *params]))

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        guesses = []
        for factor in (1.0, 0.1, 10.0):
            K = float(np.median(x[x > 0])) * factor
            basis = super()._evaluate(x, np.array([0.0, 1.0, K]))
            amplitude = max(float(np.linalg.lstsq(basis[:, None], y, rcond=None)[0][0]), 1e-12)
            guesses.append(np.array([amplitude, K]))
        return guesses

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        return curve_formula(self, params, latex)


class MonodParameterModel(_OneFactorModel):
    """mu(S) = mu_max*S/(K_s+S), S >= 0; mu_max, K_s > 0.

    Requires identity target scale to preserve physical zero at S=0. The general
    regressor rejects this zero if the chosen time parameter must be strictly positive.
    """

    name = "monod"
    n_coefficients = 2
    required_transform = "identity"

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        feature_names: Sequence[str] | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Fit original nonnegative rates through the existing rational evaluator."""
        x, target, weights = self._prepare_fit(X, y, feature_names, sample_weight)
        if np.any(target < 0):
            raise ValueError("Monod targets must be nonnegative rates in identity scale.")
        if not np.any(target > 0):
            raise ValueError(
                "Monod requires at least one positive rate; all-zero data are unidentifiable."
            )
        self._fit_curve(_MonodCurve(), x, target, weights)
        self.mu_max_, self.K_s_ = map(float, self._params)
        if not np.min(x[x > 0]) <= self.K_s_ <= np.max(x):
            self.warnings_ += (
                "Monod K_s lies outside observed positive substrate levels; "
                "saturation is poorly covered.",
            )
        return self
