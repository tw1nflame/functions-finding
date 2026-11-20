"""Контракт расширяемых математических семейств без состояния обучения."""

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from .._validation import FloatArray, as_vector


class BaseFunctionFamily(ABC):
    """Задать функцию, имена параметров, границы и начальные оценки по обучающим данным.

    Подклассы реализуют ``_evaluate``, ``initial_guess`` и ``formula``.
    Проверка входов и конечности результата общая для всех встроенных семейств.
    """

    name: str
    parameter_names: tuple[str, ...]
    positive_parameters: tuple[str, ...] = ()
    requires_initial_value: bool = False
    expression_template: str | None = None
    latex_expression_template: str | None = None

    def with_initial_value(self, value: float) -> "BaseFunctionFamily":
        """Вернуть семейство с известным начальным значением, если оно поддерживается."""
        raise ValueError(f"{self.name} does not use an initial value.")

    def expression(
        self,
        parameters: Mapping[str, str],
        *,
        variable: str = "t",
        initial: str = "y0",
        latex: bool = False,
    ) -> str:
        """Составить правую часть выражения без исполнения пользовательских строк.

        Необязательное расширение для второго этапа: шаблоны используют
        @parameter@, @t@ и @y0@. Пользовательское семейство может переопределить
        этот метод вместо задания шаблонов.
        """
        template = self.latex_expression_template if latex else self.expression_template
        if template is None:
            raise NotImplementedError(f"{self.name} needs an expression template for composition.")
        values = {**parameters, "t": variable, "y0": initial}
        return re.sub(r"@([^@]+)@", lambda match: values[match.group(1)], template)

    @property
    def n_parameters(self) -> int:
        """Число свободных коэффициентов, включая свободный член."""
        return len(self.parameter_names)

    def parameter_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Вернуть включительные нижние и верхние границы в порядке parameter_names."""
        return np.full(self.n_parameters, -np.inf), np.full(self.n_parameters, np.inf)

    def validate_domain(self, x: FloatArray) -> None:
        """Вызвать ValueError, если конечный вектор x выходит за область определения семейства."""
        return None

    def validate_parameters(self, params: ArrayLike) -> FloatArray:
        """Вернуть конечный вектор параметров, удовлетворяющий границам семейства."""
        params = as_vector(params, "params")
        if params.size != self.n_parameters:
            raise ValueError(f"{self.name} requires {self.n_parameters} parameters.")
        lower, upper = self.parameter_bounds()
        if np.any(params < lower) or np.any(params > upper):
            raise ValueError(f"Parameters outside bounds for {self.name}.")
        return params

    def evaluate(self, x: ArrayLike, params: ArrayLike) -> FloatArray:
        """Вычислить значения для конечного одномерного x с проверкой параметров и переполнений.

        Возвращается массив той же формы, что и x. Пустой массив x допустим.
        """
        x = as_vector(x, "x", allow_empty=True)
        params = self.validate_parameters(params)
        self.validate_domain(x)
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            result = np.asarray(self._evaluate(x, params), dtype=float)
        if result.shape != x.shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{self.name} returned non-finite values or an invalid shape.")
        return result

    @abstractmethod
    def _evaluate(self, x: FloatArray, params: FloatArray) -> FloatArray:
        """Вычислить значения функции для уже проверенных входов."""

    @abstractmethod
    def initial_guess(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Вернуть один допустимый вектор параметров только по обучающим данным."""

    def initial_guesses(self, x: FloatArray, y: FloatArray) -> list[FloatArray]:
        """Вернуть детерминированные начальные векторы для общего подбора с несколькими стартами."""
        return [self.initial_guess(x, y)]

    @abstractmethod
    def formula(self, params: ArrayLike, *, latex: bool = False) -> str:
        """Вернуть явную формулу по x с численными коэффициентами."""

    def design_matrix(self, x: FloatArray) -> FloatArray | None:
        """Вернуть матрицу линейной модели или None для нелинейной оптимизации."""
        return None

    def identifiability_warnings(self, x: FloatArray, params: FloatArray) -> tuple[str, ...]:
        """Вернуть предупреждения семейства о практически неразличимых параметрах.

        Обычный подбор сохраняет предупреждение в результате.
        Второй этап исключает такие подогнанные оценки дополнительно к проверкам якобиана.
        """
        return ()

    def linear_parameters(self, coefficients: FloatArray, x: FloatArray) -> FloatArray:
        """Перевести коэффициенты линейного базиса в публичную параметризацию."""
        return coefficients


def transition_guesses(x: FloatArray, y: FloatArray, fraction: float) -> list[FloatArray]:
    """Оценить c, L, k, x0 для монотонных переходов по обучающим данным."""
    order = np.argsort(x)
    span = float(np.ptp(x))
    low, high = float(np.min(y)), float(np.max(y))
    direction = 1.0 if y[order[-1]] >= y[order[0]] else -1.0
    center = float(x[np.argmin(np.abs(y - (low + fraction * (high - low))))])
    centers = [center, float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))]
    return [
        np.array([low, max(high - low, 1e-8), direction * rate / span, location])
        for rate in (4.0, 1.0, 10.0)
        for location in centers
    ]


def number(value: float, latex: bool = False) -> str:
    """Отформатировать коэффициент, включая научную запись в LaTeX."""
    result = f"{value:.12g}"
    if latex and "e" in result:
        mantissa, exponent = result.split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return result
