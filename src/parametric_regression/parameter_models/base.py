"""Контракт расширения и общая реализация моделей коэффициентов через sklearn."""

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
    """Проверить конечную вещественную матрицу формы (число наблюдений, число условий)."""
    array = np.asarray(X)
    if array.ndim != 2 or array.dtype.kind not in "iuf" or min(array.shape) < 1:
        raise ValueError("X must be a nonempty 2-D matrix of real numeric conditions.")
    array = np.asarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Conditions must be finite.")
    return array


def latex_name(name: str) -> str:
    """Представить произвольное имя столбца как буквальный текст LaTeX."""
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
    """Модель зависимости коэффициента от условий; перед обучением создаётся копия экземпляра.

    Реализуйте fit, predict, formula и complexity. fit использует только переданные
    обучающие данные и учитывает sample_weight для равного вклада независимых экспериментов.
    Необязательный required_transform задаёт шкалу 'identity' или 'log'.
    Регрессор применяет это преобразование и его обратное ровно по одному разу.
    """

    name: str
    required_transform: str | None = None

    @abstractmethod
    def complexity(self, n_features: int) -> int:
        """Вернуть число оцениваемых коэффициентов для n_features внешних факторов."""

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        feature_names: Sequence[str] | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Обучить по условиям и оценкам коэффициентов; вернуть текущий экземпляр."""

    @abstractmethod
    def predict(self, X: ArrayLike) -> FloatArray:
        """Вернуть по одному прогнозу коэффициента на строку условий."""

    @abstractmethod
    def formula(self, *, latex: bool = False) -> str:
        """Вернуть правую часть обученного выражения через именованные условия."""


class _LeastSquaresParameterModel(BaseParameterModel):
    """Общий масштабированный полиномиальный базис и обычный МНК sklearn."""

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
        """Посчитать оцениваемые коэффициенты базиса, включая свободный член."""
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
        """Обучить полноранговую масштабированную модель через sklearn LinearRegression.

        X имеет форму (n, m), y — (n,), feature_names задаёт имена m факторов.
        Положительные sample_weight позволяют выровнять суммарный вес независимых экспериментов.
        Недостаток наблюдений или неполный ранг матрицы вызывают ValueError.
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
        """Вернуть конечные прогнозы коэффициентов с сохранением обучающего порядка факторов."""
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
        """Вернуть явное выражение с константами масштабирования, оценёнными по train."""
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
    """Назначить всем траекториям одного независимого эксперимента суммарный вес, равный единице."""
    codes, _ = pd.factorize(pd.Series(list(groups), dtype=object), sort=False)
    counts = np.bincount(codes)
    return 1.0 / counts[codes]
