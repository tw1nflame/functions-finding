"""Small extension contract and shared sklearn implementation for coefficient models."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .._validation import FloatArray, as_vector
from ..families.base import number


def condition_matrix(X: ArrayLike) -> FloatArray:
    """Validate a finite real matrix of shape (observations, conditions)."""
    array = np.asarray(X)
    if array.ndim != 2 or array.dtype.kind not in "iuf" or min(array.shape) < 1:
        raise ValueError("X must be a nonempty 2-D matrix of real numeric conditions.")
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Conditions must be finite.")
    return array


def latex_name(name: str) -> str:
    """Render an arbitrary column name as literal LaTeX text."""
    escapes = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "\\": r"\backslash ",
        "^": r"\hat{}",
    }
    return r"\mathrm{" + "".join(escapes.get(char, char) for char in name) + "}"


class BaseParameterModel(ABC):
    """A condition-to-coefficient model; instances are copied before training.

    Implement fit/predict/formula and complexity. fit must use only supplied
    training data and honor sample_weight (equal independent-experiment weights).
    required_transform optionally declares 'identity' or 'log' target scale.
    The surrounding regressor applies that transform and its inverse exactly once.
    """

    name: str
    required_transform: str | None = None

    @abstractmethod
    def complexity(self, n_features: int) -> int:
        """Return the number of estimated coefficients for n_features conditions."""

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        feature_names: Sequence[str] | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Fit on training conditions and coefficient estimates; return self."""

    @abstractmethod
    def predict(self, X: ArrayLike) -> FloatArray:
        """Return one predicted coefficient per condition row."""

    @abstractmethod
    def formula(self, *, latex: bool = False) -> str:
        """Return the fitted right-hand-side expression in named conditions."""


class _LeastSquaresParameterModel(BaseParameterModel):
    """Shared scaled polynomial basis and sklearn ordinary least squares."""

    degree: int
    interactions: bool = True
    value: float | None = None

    def _basis(self, n_features: int) -> tuple[PolynomialFeatures, np.ndarray]:
        transformer = PolynomialFeatures(degree=self.degree, include_bias=True)
        transformer.fit(np.zeros((1, n_features)))
        mask = np.ones(len(transformer.powers_), dtype=bool)
        if not self.interactions:
            mask = np.count_nonzero(transformer.powers_, axis=1) <= 1
        return transformer, mask

    def complexity(self, n_features: int) -> int:
        """Count fitted basis coefficients, including the intercept."""
        if self.value is not None:
            return 0
        return int(self._basis(n_features)[1].sum())

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        feature_names: Sequence[str] | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Fit a full-rank scaled design using sklearn LinearRegression.

        X is (n, m), y is (n,), feature_names names the m conditions. Positive
        sample_weight can give equal total weight to each independent experiment.
        Insufficient observations or a rank-deficient design raise ValueError.
        """
        # A failed refit must not leave a previously fitted model usable.
        self._regression = None
        X, y = condition_matrix(X), as_vector(y, "y")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of observations.")
        names = (
            tuple(feature_names)
            if feature_names is not None
            else tuple(f"z{i + 1}" for i in range(X.shape[1]))
        )
        if len(names) != X.shape[1] or len(set(names)) != len(names):
            raise ValueError("feature_names must uniquely name all condition columns.")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Condition names must be nonempty strings.")
        weights = np.ones(len(y)) if sample_weight is None else as_vector(sample_weight, "weights")
        if weights.shape != y.shape or np.any(weights <= 0):
            raise ValueError("sample_weight must have one positive finite value per observation.")
        self.feature_names_in_ = names
        self._scaler = StandardScaler().fit(X, sample_weight=weights)
        self._transformer, self._mask = self._basis(X.shape[1])
        design = self._transformer.transform(self._scaler.transform(X))[:, self._mask]
        if len(y) < self.complexity(X.shape[1]):
            raise ValueError(f"{self.name} has more coefficients than training observations.")
        response = y if self.value is None else np.full_like(y, self.value)
        regression = LinearRegression(fit_intercept=False).fit(
            design, response, sample_weight=weights
        )
        if regression.rank_ < design.shape[1]:
            raise ValueError(f"{self.name} condition design is rank deficient.")
        if not np.isfinite(regression.coef_).all():
            raise ValueError("Non-finite parameter-model coefficients.")
        self._regression = regression
        self.coefficients_ = regression.coef_.copy()
        return self

    def _require_fit(self) -> None:
        if getattr(self, "_regression", None) is None:
            raise RuntimeError("Parameter model must be successfully fitted first.")

    def predict(self, X: ArrayLike) -> FloatArray:
        """Return finite coefficient predictions, preserving training feature order."""
        self._require_fit()
        X = condition_matrix(X)
        if X.shape[1] != len(self.feature_names_in_):
            raise ValueError("Number of condition columns differs from training.")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            design = self._transformer.transform(self._scaler.transform(X))[:, self._mask]
            prediction = self._regression.predict(design)
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite parameter predictions.")
        return prediction

    def formula(self, *, latex: bool = False) -> str:
        """Return an explicit expression, including training-derived scaling constants."""
        self._require_fit()
        variables = []
        for name, center, scale in zip(
            self.feature_names_in_, self._scaler.mean_, self._scaler.scale_, strict=True
        ):
            label = latex_name(name) if latex else name
            if latex:
                variables.append(
                    rf"\left(\frac{{{label}-({number(center, True)})}}"
                    rf"{{{number(scale, True)}}}\right)"
                )
            else:
                variables.append(f"(({label} - ({number(center)}))/({number(scale)}))")
        terms = []
        for coefficient, powers in zip(
            self.coefficients_, self._transformer.powers_[self._mask], strict=True
        ):
            factors = []
            for variable, power in zip(variables, powers, strict=True):
                if power:
                    factors.append(
                        variable
                        if power == 1
                        else (f"{variable}^{{{power}}}" if latex else f"{variable}**{power}")
                    )
            multiply = r"\," if latex else "*"
            terms.append(multiply.join([f"({number(coefficient, latex)})", *factors]))
        return " + ".join(terms)


def independent_weights(groups: Sequence[object]) -> FloatArray:
    """Give all trajectories of each independent experiment total weight one."""
    codes, _ = pd.factorize(pd.Series(list(groups), dtype=object), sort=False)
    counts = np.bincount(codes)
    return 1.0 / counts[codes]
