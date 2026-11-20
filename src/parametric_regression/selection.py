"""Выбор по общей отложенной части с предпочтением простоты, без переобучения."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ._validation import FloatArray, validate_xy
from .families import (
    BaseFunctionFamily,
    ExponentialFamily,
    GompertzFamily,
    LogisticFamily,
    PolynomialFamily,
    RationalSaturationFamily,
)
from .fitting import _positive_integer, fit_family
from .metrics import regression_metrics
from .results import SelectionResult, SplitInfo, SplitMethod


class ModelSelector:
    """Сравнить семейства на общей отложенной части по RMSE валидации.

    Parameters
    ----------
    families : последовательность или None
        Явный список кандидатов. При None используются семь стандартных вариантов:
        полиномы степеней 1/2/3, экспонента, логистика, Гомпертц и рациональное насыщение.
    validation_size : float
        Доля в интервале (0, 1). Для временного или группового разбиения — доля
        уникальных временных отметок или групп соответственно.
    random_state : int
        Начальное значение генератора для воспроизводимых случайных и групповых разбиений.
    relative_tolerance, absolute_tolerance : float
        Допускаются RMSE <= min_RMSE * (1 + relative_tolerance) + absolute_tolerance.
        Среди них выбирается меньшее число параметров, затем меньшая ошибка,
        затем исходный порядок кандидатов.
    max_nfev, max_starts : int
        Бюджет вычислений на один старт и максимальное число начальных приближений.
    """

    def __init__(
        self,
        families: Sequence[BaseFunctionFamily] | None = None,
        *,
        validation_size: float = 0.2,
        random_state: int = 42,
        relative_tolerance: float = 0.01,
        absolute_tolerance: float = 1e-12,
        max_nfev: int = 2000,
        max_starts: int = 12,
    ) -> None:
        if families is None:
            families = [
                PolynomialFamily(1),
                PolynomialFamily(2),
                PolynomialFamily(3),
                ExponentialFamily(),
                LogisticFamily(),
                GompertzFamily(),
                RationalSaturationFamily(),
            ]
        self.families = tuple(families)
        if not self.families or any(not isinstance(f, BaseFunctionFamily) for f in self.families):
            raise ValueError("families must be a non-empty sequence of BaseFunctionFamily objects.")
        if not np.isfinite(validation_size) or not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        for name, value in (
            ("relative_tolerance", relative_tolerance),
            ("absolute_tolerance", absolute_tolerance),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise ValueError("random_state must be an integer seed.")
        if not 0 <= random_state < 2**32:
            raise ValueError("random_state must be between 0 and 2**32 - 1.")
        _positive_integer(max_nfev, "max_nfev")
        _positive_integer(max_starts, "max_starts")
        self.validation_size = validation_size
        self.random_state = random_state
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.max_nfev, self.max_starts = max_nfev, max_starts

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        split: SplitMethod = "random",
        groups: ArrayLike | None = None,
    ) -> SelectionResult:
        """Подобрать всех кандидатов по обучающим строкам и оценить на общей отложенной части.

        x/y — конечные вещественные одномерные массивы. При split='group' массив groups
        должен содержать по одному непустому ID эксперимента на строку.
        Временное разбиение сортирует x и сохраняет одинаковые отметки в одной части.
        В каждой части нужно не менее двух наблюдений.
        Все попытки возвращаются даже при отсутствии успешного кандидата;
        автоматического переобучения после выбора нет.
        """
        x, y = validate_xy(x, y)
        protocol = self._split(x, split, groups)
        train, validation = protocol.train_indices, protocol.validation_indices
        candidates = []
        for family in self.families:
            candidate = fit_family(
                family, x[train], y[train], max_nfev=self.max_nfev, max_starts=self.max_starts
            )
            candidate.split_protocol = protocol
            if candidate.success:
                try:
                    prediction = candidate.predict(x[validation])
                    candidate.validation_metrics = regression_metrics(y[validation], prediction)
                except Exception as error:
                    candidate.success = False
                    candidate.message = f"Validation failed: {type(error).__name__}: {error}"
            candidates.append(candidate)
        successful = [i for i, candidate in enumerate(candidates) if candidate.success]
        winner = None
        if successful:
            minimum = min(candidates[i].validation_metrics["rmse"] for i in successful)
            threshold = minimum * (1 + self.relative_tolerance) + self.absolute_tolerance
            eligible = [
                i for i in successful if candidates[i].validation_metrics["rmse"] <= threshold
            ]
            winner = min(
                eligible,
                key=lambda i: (
                    candidates[i].n_parameters,
                    candidates[i].validation_metrics["rmse"],
                    i,
                ),
            )
        return SelectionResult(
            tuple(candidates), winner, protocol, self.relative_tolerance, self.absolute_tolerance
        )

    def fit_dataframe(
        self,
        df: pd.DataFrame,
        *,
        x_column: str,
        y_column: str,
        split: SplitMethod = "random",
        group_column: str | None = None,
    ) -> SelectionResult:
        """Выбрать модель по именованным столбцам DataFrame; group_column задаёт ID экспериментов.

        Индексы разбиения — позиции строк, не зависящие от df.index.
        Отсутствующие столбцы, повторяющиеся имена и пропущенные значения вызывают ValueError.
        """
        columns = [x_column, y_column] + ([group_column] if group_column is not None else [])
        if not df.columns.is_unique:
            raise ValueError("DataFrame column names must be unique.")
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing DataFrame columns: {missing}")
        for column in (x_column, y_column):
            if not pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(
                df[column]
            ):
                raise ValueError(f"Column {column!r} must be numeric.")
            if pd.api.types.is_complex_dtype(df[column]):
                raise ValueError(f"Column {column!r} must contain real numbers.")
        return self.fit(
            df[x_column].to_numpy(dtype=float, na_value=np.nan),
            df[y_column].to_numpy(dtype=float, na_value=np.nan),
            split=split,
            groups=None if group_column is None else df[group_column].to_numpy(),
        )

    def _split(self, x: FloatArray, split: SplitMethod, groups: ArrayLike | None) -> SplitInfo:
        if split not in ("random", "time", "group"):
            raise ValueError("split must be 'random', 'time' or 'group'.")
        if groups is not None and split != "group":
            raise ValueError("groups requires split='group' to prevent experiment leakage.")
        indices = np.arange(x.size)
        if split == "random":
            train, validation = train_test_split(
                indices, test_size=self.validation_size, random_state=self.random_state
            )
        elif split == "time":
            times = np.unique(x)
            n_validation = int(np.ceil(times.size * self.validation_size))
            if times.size - n_validation < 1:
                raise ValueError("Time split requires at least two distinct timestamps.")
            boundary = times[-n_validation]
            order = np.argsort(x, kind="stable")
            train, validation = order[x[order] < boundary], order[x[order] >= boundary]
        else:
            if groups is None:
                raise ValueError("split='group' requires groups (experiment identifiers).")
            group_array = np.asarray(groups)
            if group_array.ndim != 1 or group_array.shape != x.shape:
                raise ValueError("groups must be one-dimensional and have the same length as x.")
            if np.any(pd.isna(group_array)):
                raise ValueError("groups must not contain missing identifiers.")
            if group_array.dtype.kind in "f" and not np.all(np.isfinite(group_array)):
                raise ValueError("groups must not contain infinite identifiers.")
            codes, unique = pd.factorize(group_array, sort=False)
            if len(unique) < 2:
                raise ValueError("Group split requires at least two distinct experiments.")
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=self.validation_size, random_state=self.random_state
            )
            train, validation = next(splitter.split(indices, groups=codes))
        if min(len(train), len(validation)) < 2:
            raise ValueError("At least two observations are required in each split partition.")
        return SplitInfo(
            split,
            train,
            validation,
            self.validation_size,
            self.random_state if split != "time" else None,
        )
