"""Закон Аррениуса с явной логарифмической шкалой целевой величины и температурой в Кельвинах."""

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray
from ..families.base import number
from ._specialized import _OneFactorModel
from .base import latex_name


class ArrheniusParameterModel(_OneFactorModel):
    """Закон Аррениуса: log(k) = log(k_ref) - E_a/R * (1/T - 1/T_ref), E_a >= 0.

    T и T_ref задаются в Кельвинах, R — в Дж/(моль·К), E_a — в Дж/моль.
    При самостоятельном вызове fit принимает log(k), predict возвращает log(k).
    MultiExperimentRegressor выполняет обратное преобразование exp ровно один раз.
    Автоматического перевода из градусов Цельсия и определения единиц нет.

    Parameters
    ----------
    T_ref : float
        Известная положительная опорная температура в Кельвинах; по умолчанию 298.15.
    """

    name = "arrhenius"
    n_coefficients = 2
    required_transform = "log"
    R = 8.314462618

    def __init__(self, T_ref: float = 298.15) -> None:
        value = np.asarray(T_ref)
        if (
            value.shape != ()
            or value.dtype.kind not in "iuf"
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError("T_ref must be a finite temperature > 0 Kelvin.")
        self.T_ref = float(T_ref)

    def _coordinate(self, x: FloatArray) -> FloatArray:
        if np.any(x <= 0):
            raise ValueError("Arrhenius requires temperature > 0 Kelvin; no Celsius inference.")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            return -(1.0 / x - 1.0 / self.T_ref) / self.R

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        feature_names: Sequence[str] | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> Self:
        """Оценить логарифм скорости взвешенным линейным МНК при E_a >= 0."""
        x, target, weights = self._prepare_fit(X, y, feature_names, sample_weight)
        z = self._coordinate(x)
        weights = weights / np.max(weights)
        center = np.average(z, weights=weights)
        mean = np.average(target, weights=weights)
        scale = np.max(np.abs(z - center))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Arrhenius temperature design is rank deficient.")
        coordinate = (z - center) / scale
        denominator = np.sum(weights * coordinate**2)
        energy = max(
            float(np.sum(weights * coordinate * (target - mean)) / denominator / scale), 0.0
        )
        log_reference = float(mean - energy * center)
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            reference = np.exp(log_reference)
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("Arrhenius k_ref is outside the positive float64 range.")
        self._params = np.array([log_reference, energy])
        self.k_ref_, self.E_a_ = float(reference), energy
        self.parameters_ = {"k_ref": self.k_ref_, "E_a": energy}
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Вернуть log(k) для температур в Кельвинах; физическая скорость равна exp результата."""
        x = self._prediction_input(X)
        with np.errstate(over="raise", invalid="raise"):
            result = self._params[0] + self._params[1] * self._coordinate(x)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite Arrhenius log-rate prediction.")
        return result

    def formula(self, *, latex: bool = False) -> str:
        """Вернуть log(k); связь общей модели применяет exp к этому выражению ровно один раз."""
        self._require_fit()
        name = self.feature_names_in_[0]
        reference, energy, gas, log_reference = (
            number(v, latex) for v in (self.T_ref, self.E_a_, self.R, self._params[0])
        )
        if latex:
            return (
                rf"({log_reference})-\frac{{({energy})}}{{{gas}}}"
                rf"(\frac{{1}}{{{latex_name(name)}}}-\frac{{1}}{{{reference}}})"
            )
        return f"({log_reference}) - ({energy})/({gas})*(1/({name}) - 1/({reference}))"
