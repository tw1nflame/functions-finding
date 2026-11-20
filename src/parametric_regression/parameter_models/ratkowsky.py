"""Ограниченный субоптимальный температурный закон роста по модели квадратного корня."""

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from ..families import BaseFunctionFamily
from ._specialized import _OneFactorModel, curve_formula


class _RatkowskyCurve(BaseFunctionFamily):
    name = "ratkowsky_log_curve"
    parameter_names = ("log_b", "T_min")
    expression_template = "2*(@log_b@) + 2*log(@t@ - (@T_min@))"
    latex_expression_template = r"2(@log_b@)+2\ln(@t@-(@T_min@))"

    def __init__(self, minimum: float, maximum: float) -> None:
        self.minimum, self.maximum = minimum, maximum

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        return np.array([-np.inf, -np.inf]), np.array([np.inf, np.nextafter(self.minimum, -np.inf)])

    def validate_domain(self, x: FloatArray) -> None:
        if np.any(x > self.maximum):
            raise ValueError("Ratkowsky temperature exceeds the declared suboptimal upper limit.")

    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        log_b, minimum = params
        if np.any(x <= minimum):
            raise ValueError("Ratkowsky requires T > fitted T_min.")
        return 2 * log_b + 2 * np.log(x - minimum)

    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        return self.initial_guesses(x, y)[0]

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        guesses = []
        span = float(np.ptp(x))
        for factor in (0.1, 0.5, 1.0, 3.0, 10.0):
            minimum = self.minimum - factor * span
            log_b = float(np.mean(y / 2 - np.log(x - minimum)))
            guesses.append(np.array([log_b, minimum]))
        return guesses

    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        return curve_formula(self, params, latex)


class RatkowskyParameterModel(_OneFactorModel):
    """Модель Ратковского: log(mu) = 2*log(b) + 2*log(T-T_min), b > 0, T > T_min.

    Температуры задаются в градусах Цельсия.
    При самостоятельном вызове целевая величина и прогнозы имеют шкалу log(mu).
    Модель применима только в субоптимальном диапазоне роста.

    Parameters
    ----------
    temperature_max : float или None
        Известная верхняя граница применимости в градусах Цельсия ниже биологического оптимума.
        Если не задана, прогноз выше максимальной обучающей температуры запрещён.
    """

    name = "ratkowsky"
    n_coefficients = 2
    required_transform = "log"

    def __init__(self, temperature_max: float | None = None) -> None:
        if temperature_max is not None:
            value = np.asarray(temperature_max)
            if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):
                raise ValueError("temperature_max must be finite Celsius or None.")
        self.temperature_max = temperature_max

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        feature_names: Sequence[str] | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Оценить нижнюю границу температуры и масштаб по взвешенным ошибкам лог-скорости."""
        x, target, weights = self._prepare_fit(X, y, feature_names, sample_weight)
        maximum = float(np.max(x)) if self.temperature_max is None else float(self.temperature_max)
        self._fit_curve(_RatkowskyCurve(float(np.min(x)), maximum), x, target, weights)
        try:
            with np.errstate(over="raise", under="ignore"):
                b = np.exp(self._params[0])
        except FloatingPointError as error:
            self._params = None
            raise ValueError("Ratkowsky b is outside the positive float64 range.") from error
        if not np.isfinite(b) or b <= 0:
            self._params = None
            raise ValueError("Ratkowsky b is outside the positive float64 range.")
        self.b_, self.T_min_ = float(b), float(self._params[1])
        return self
