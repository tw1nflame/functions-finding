"""Результаты с полным списком кандидатов, диагностикой и точным протоколом оценки."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray
from .families import BaseFunctionFamily

SplitMethod = Literal["random", "time", "group"]


@dataclass(frozen=True)
class SplitInfo:
    """Позиции исходных строк для обучения и валидации, а также настройки разбиения."""

    method: SplitMethod
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    validation_size: float
    random_state: int | None

    def __post_init__(self) -> None:
        for name in ("train_indices", "validation_indices"):
            indices = np.array(getattr(self, name), dtype=np.int64, copy=True)
            indices.setflags(write=False)
            object.__setattr__(self, name, indices)


@dataclass
class FitResult:
    """Результат подбора одного семейства или неудачная попытка с диагностикой.

    Коэффициенты всегда получены только на обучающей части.
    При оценке через ModelSelector свойство ``metrics`` содержит метрики валидации,
    в остальных случаях — обучающие метрики.
    """

    family: BaseFunctionFamily
    success: bool
    params: FloatArray | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    message: str = ""
    warnings: tuple[str, ...] = ()
    split_protocol: SplitInfo | None = None
    nfev: int = 0
    jacobian_rank: int | None = None
    jacobian_condition: float | None = None

    @property
    def family_name(self) -> str:
        """Стабильный идентификатор семейства."""
        return self.family.name

    @property
    def n_parameters(self) -> int:
        """Число свободных параметров для предпочтения более простой модели."""
        return self.family.n_parameters

    @property
    def coefficients(self) -> dict[str, float]:
        """Именованные коэффициенты; пустой словарь, если решение не получено."""
        if self.params is None:
            return {}
        return dict(zip(self.family.parameter_names, map(float, self.params), strict=True))

    @property
    def formula(self) -> str | None:
        """Текстовая формула в исходной координате входной переменной."""
        return None if self.params is None else self.family.formula(self.params)

    @property
    def latex_formula(self) -> str | None:
        """Формула LaTeX без ограничителей математической разметки."""
        return None if self.params is None else self.family.formula(self.params, latex=True)

    @property
    def metrics(self) -> dict[str, float]:
        """Метрики валидации, если они доступны; иначе обучающие метрики."""
        return dict(self.validation_metrics or self.train_metrics)

    def predict(self, x: ArrayLike) -> FloatArray:
        """Предсказать значения по одномерному x; при неудачном подборе вызвать RuntimeError."""
        if not self.success or self.params is None:
            raise RuntimeError(f"Cannot predict with {self.family_name}: {self.message}")
        return self.family.evaluate(x, self.params)


@dataclass
class SelectionResult:
    """Результат выбора с неудачными кандидатами и выбранной моделью, если она существует."""

    candidates: tuple[FitResult, ...]
    best_index: int | None
    split_protocol: SplitInfo
    relative_tolerance: float
    absolute_tolerance: float
    criterion: str = "validation_rmse"

    @property
    def success(self) -> bool:
        """Удалось ли хотя бы одной модели обучиться и предсказать данные валидации."""
        return self.best_index is not None

    @property
    def best_result(self) -> FitResult:
        """Вернуть выбранный результат; если все кандидаты неудачны — ошибку с диагностикой."""
        if self.best_index is None:
            details = "; ".join(f"{r.family_name}: {r.message}" for r in self.candidates)
            raise RuntimeError(f"No successful candidate. {details}")
        return self.candidates[self.best_index]

    @property
    def best_family(self) -> str | None:
        """Имя выбранного семейства или None, если подходящих кандидатов нет."""
        return self.best_result.family_name if self.success else None

    @property
    def formula(self) -> str | None:
        """Численная формула выбранной модели или None."""
        return self.best_result.formula if self.success else None

    @property
    def latex_formula(self) -> str | None:
        """Формула выбранной модели в LaTeX или None."""
        return self.best_result.latex_formula if self.success else None

    @property
    def coefficients(self) -> dict[str, float]:
        """Именованные коэффициенты выбранной модели."""
        return self.best_result.coefficients if self.success else {}

    @property
    def metrics(self) -> dict[str, float]:
        """Метрики валидации выбранной модели; пустой словарь, если все кандидаты неудачны."""
        return self.best_result.metrics if self.success else {}

    def predict(self, x: ArrayLike) -> FloatArray:
        """Вычислить прогноз выбранной обученной моделью без скрытого переобучения."""
        return self.best_result.predict(x)

    def to_dataframe(self) -> pd.DataFrame:
        """Вернуть всех кандидатов по порядку с отдельными метриками обучения и валидации."""
        return pd.DataFrame(
            [
                {
                    "family": result.family_name,
                    "success": result.success,
                    "selected": index == self.best_index,
                    "n_parameters": result.n_parameters,
                    **{f"train_{key}": value for key, value in result.train_metrics.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in result.validation_metrics.items()
                    },
                    "message": result.message,
                    "warnings": "; ".join(result.warnings),
                }
                for index, result in enumerate(self.candidates)
            ]
        )
