"""Linear dependencies on an arbitrary number of conditions."""

from .base import _LeastSquaresParameterModel


class LinearParameterModel(_LeastSquaresParameterModel):
    """g(z) = intercept + sum(a_j*z_j)."""

    name = "linear"
    degree = 1
