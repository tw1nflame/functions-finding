"""Available parametric families and their extension contract."""

from .base import BaseFunctionFamily
from .exponential import ExponentialFamily
from .gompertz import GompertzFamily
from .logistic import LogisticFamily
from .polynomial import PolynomialFamily
from .rational import RationalSaturationFamily
from .relaxation import ExponentialRelaxationFamily

__all__ = [
    "BaseFunctionFamily",
    "ExponentialFamily",
    "ExponentialRelaxationFamily",
    "GompertzFamily",
    "LogisticFamily",
    "PolynomialFamily",
    "RationalSaturationFamily",
]
