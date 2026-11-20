"""Квадратичные модели условий с необязательными попарными взаимодействиями."""

from .base import _LeastSquaresParameterModel


class PolynomialParameterModel(_LeastSquaresParameterModel):
    """Полный квадратичный базис; interactions=False исключает произведения z_i*z_j при i != j."""

    degree = 2

    def __init__(self, degree: int = 2, *, interactions: bool = True) -> None:
        if isinstance(degree, bool) or degree != 2:
            raise ValueError("PolynomialParameterModel supports degree=2 only.")
        if not isinstance(interactions, bool):
            raise ValueError("interactions must be a bool.")
        self.interactions = interactions
        self.name = "quadratic" if interactions else "quadratic_no_interactions"
