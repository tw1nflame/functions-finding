"""Линейные зависимости от произвольного числа внешних условий."""

from .base import _LeastSquaresParameterModel


class LinearParameterModel(_LeastSquaresParameterModel):
    """Линейная зависимость: g(z) = a0 + sum(a_j*z_j), где a0 — свободный член."""

    name = "linear"
    degree = 1
