"""One-factor validation and weighted reuse of the existing family fitter."""

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray, as_vector
from ..families import BaseFunctionFamily
from ..families.base import number
from ..fitting import fit_family
from .base import BaseParameterModel, condition_matrix, latex_name


class _WeightedFamily(BaseFunctionFamily):
    """Multiply targets and predictions by sqrt(weight) for the shared fitter."""

    def __init__(self, family: BaseFunctionFamily, weights: FloatArray) -> None:
        self.family = family
        self.root_weights = np.sqrt(weights / np.max(weights))
        if np.any(self.root_weights == 0):
            raise ValueError("Sample-weight dynamic range exceeds float64 precision.")
        self.name = family.name
        self.parameter_names = family.parameter_names

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        return self.family.parameter_bounds()

    def validate_domain(self, x: FloatArray) -> None:
        self.family.validate_domain(x)

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        return self.root_weights * self.family.evaluate(x, params)

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        return self.family.initial_guesses(x, y / self.root_weights)

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        return self.family.formula(params, latex=latex)


def curve_formula(family: BaseFunctionFamily, params: ArrayLike, latex: bool) -> str:
    values = (number(v, latex) for v in family.validate_parameters(params))
    return "y = " + family.expression(
        dict(zip(family.parameter_names, values, strict=True)), variable="x", latex=latex
    )


class _OneFactorModel(BaseParameterModel):
    """Common input contract; derived classes retain explicit scientific semantics."""

    n_coefficients: int

    def complexity(self, n_features: int) -> int:
        """Return parameter count, rejecting multivariate use of a one-factor law."""
        if n_features != 1:
            raise ValueError(f"{self.name} requires exactly one condition factor.")
        return self.n_coefficients

    def _prepare_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        feature_names: Sequence[str] | None,
        sample_weight: ArrayLike | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        self._params = None
        self._curve = None
        self.parameters_ = {}
        self.warnings_ = ()
        matrix, target = condition_matrix(X), as_vector(y, "y")
        self.complexity(matrix.shape[1])
        if len(matrix) != len(target):
            raise ValueError("X and y must have the same number of observations.")
        names = tuple(feature_names) if feature_names is not None else ("z1",)
        if len(names) != 1 or not isinstance(names[0], str) or not names[0]:
            raise ValueError("feature_names must contain exactly one nonempty string.")
        weights = (
            np.ones(len(target)) if sample_weight is None else as_vector(sample_weight, "weights")
        )
        if len(weights) != len(target) or np.any(weights <= 0):
            raise ValueError(
                "sample_weight must contain one finite positive weight per observation."
            )
        if np.unique(matrix[:, 0]).size < self.n_coefficients:
            raise ValueError(
                f"{self.name} needs at least {self.n_coefficients} distinct conditions."
            )
        self.feature_names_in_ = names
        return matrix[:, 0], target, weights

    def _fit_curve(
        self,
        curve: BaseFunctionFamily,
        x: FloatArray,
        target: FloatArray,
        weights: FloatArray,
    ) -> Self:
        adapter = _WeightedFamily(curve, weights)
        result = fit_family(adapter, x, target * adapter.root_weights)
        if not result.success:
            raise ValueError(f"{self.name} fitting failed: {result.message}")
        if result.jacobian_rank < curve.n_parameters or result.jacobian_condition > 1e8:
            raise ValueError(f"{self.name} parameters are unidentifiable or ill-conditioned.")
        self.jacobian_rank_ = result.jacobian_rank
        self.jacobian_condition_ = result.jacobian_condition
        self.warnings_ = result.warnings
        self._curve, self._params = curve, result.params.copy()
        self.parameters_ = result.coefficients
        return self

    def _require_fit(self) -> None:
        if getattr(self, "_params", None) is None:
            raise RuntimeError("Parameter model must be successfully fitted first.")

    def _prediction_input(self, X: ArrayLike) -> FloatArray:
        self._require_fit()
        matrix = condition_matrix(X)
        self.complexity(matrix.shape[1])
        return matrix[:, 0]

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict in the declared target scale using exactly one factor."""
        x = self._prediction_input(X)
        return self._curve.evaluate(x, self._params)

    def formula(self, *, latex: bool = False) -> str:
        """Return an explicit RHS in the declared target scale."""
        self._require_fit()
        variable = self.feature_names_in_[0]
        if latex:
            variable = latex_name(variable)
        return self._curve.expression(
            {key: number(value, latex) for key, value in self.parameters_.items()},
            variable=variable,
            latex=latex,
        )
