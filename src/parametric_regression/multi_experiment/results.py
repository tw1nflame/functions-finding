"""Обученная двухэтапная аналитическая функция и отдельные отчёты её проверки."""

import warnings
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .._validation import FloatArray, as_vector
from ..families import BaseFunctionFamily
from ..metrics import regression_metrics
from ..parameter_models import BaseParameterModel
from ..parameter_models.base import condition_matrix, latex_name
from ..results import FitResult
from .validation import CurveEvaluation, ExperimentSplit, Trajectory

Conditions = Mapping[str, float] | pd.DataFrame


class ExtrapolationWarning(UserWarning):
    """Условия прогноза выходят за диапазоны пригодных обучающих условий."""


@dataclass
class ParameterRelationship:
    """Модель коэффициента с явным преобразованием целевой шкалы identity или log."""

    model: BaseParameterModel
    transform: str = "identity"

    def predict(self, X: ArrayLike) -> FloatArray:
        """Вернуть физические коэффициенты, проверив конечность и сохранение положительности."""
        value = as_vector(self.model.predict(X), "predicted parameters")
        if self.transform == "log":
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                value = np.exp(value)
            if not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError("Positive parameter is outside the representable float64 range.")
        return value

    def formula(self, *, latex: bool = False) -> str:
        """Вернуть выражение физического коэффициента с учётом обратного преобразования."""
        expression = self.model.formula(latex=latex)
        if self.transform == "log":
            return rf"\exp\left({expression}\right)" if latex else f"exp({expression})"
        return expression


@dataclass
class MultiExperimentResult:
    """Результат двухэтапного обучения, промежуточные коэффициенты train и ошибки отложенных кривых.

    parameter_models сопоставляет имена временных параметров объектам ParameterRelationship.
    Тестовые данные не участвуют в выборе кандидатов.
    Скрытого переобучения на объединении train и validation нет.
    """

    time_family: BaseFunctionFamily
    condition_columns: tuple[str, ...]
    split_protocol: ExperimentSplit
    aggregation: str
    relative_tolerance: float
    absolute_tolerance: float
    success: bool = False
    message: str = ""
    warnings: tuple[str, ...] = ()
    parameter_models: dict[str, ParameterRelationship] = field(default_factory=dict)
    experiment_parameters: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_fits: dict[Hashable, FitResult] = field(default_factory=dict)
    candidate_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    parameter_candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    evaluations: dict[str, CurveEvaluation] = field(default_factory=dict)
    parameter_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    condition_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    training_conditions: FloatArray | None = None
    _trajectories: tuple[Trajectory, ...] = field(default=(), repr=False)
    criterion: str = "validation_curve_rmse"

    def _require_fit(self) -> None:
        if not self.success:
            raise RuntimeError(f"No successful multi-experiment model: {self.message}")

    @property
    def metrics(self) -> dict[str, dict[str, object]]:
        """Отдельные метрики индивидуальных train-кривых, общей train-модели, валидации и теста."""
        return {name: evaluation.summary() for name, evaluation in self.evaluations.items()}

    @property
    def parameter_formulas(self) -> dict[str, str]:
        """Выражения физических коэффициентов через именованные условия."""
        return {
            name: relationship.formula() for name, relationship in self.parameter_models.items()
        }

    @property
    def parameter_latex_formulas(self) -> dict[str, str]:
        """Выражения коэффициентов в LaTeX без ограничителей математической разметки."""
        return {
            name: relationship.formula(latex=True)
            for name, relationship in self.parameter_models.items()
        }

    def _formula(self, latex: bool) -> str | None:
        if not self.success:
            return None
        parameters = self.parameter_latex_formulas if latex else self.parameter_formulas
        conditions = [latex_name(name) if latex else name for name in self.condition_columns]
        # User-defined condition names must not collide with time or initial state.
        symbols = []
        for base in ("t", "y0"):
            symbol, index = base, 0
            while symbol in self.condition_columns:
                index += 1
                symbol = f"{base}_{index}"
            symbols.append(symbol)
        time_symbol, initial_symbol = symbols
        arguments = [time_symbol, *conditions]
        initial = "y_0" if latex and initial_symbol == "y0" else initial_symbol
        if self.time_family.requires_initial_value:
            arguments.append(initial)
        return (
            "y("
            + ", ".join(arguments)
            + ") = "
            + self.time_family.expression(
                parameters, variable=time_symbol, initial=initial, latex=latex
            )
        )

    @property
    def formula(self) -> str | None:
        """Полностью подставленная численная аналитическая зависимость или None при неудаче."""
        return self._formula(False)

    @property
    def latex_formula(self) -> str | None:
        """Полностью подставленная аналитическая зависимость в LaTeX или None при неудаче."""
        return self._formula(True)

    def _conditions(self, conditions: Conditions) -> pd.DataFrame:
        if isinstance(conditions, Mapping):
            frame = pd.DataFrame([conditions])
        elif isinstance(conditions, pd.DataFrame):
            frame = conditions.copy()
        else:
            raise ValueError("conditions must be a mapping of scalar values or a DataFrame.")
        if not frame.columns.is_unique or set(frame.columns) != set(self.condition_columns):
            raise ValueError(f"conditions must contain exactly {self.condition_columns}.")
        frame = frame.loc[:, list(self.condition_columns)]
        for name in self.condition_columns:
            dtype = frame[name].dtype
            if (
                not pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
                or pd.api.types.is_complex_dtype(dtype)
            ):
                raise ValueError(f"Condition {name!r} must be real numeric.")
        values = frame.to_numpy(dtype=float, na_value=np.nan)
        condition_matrix(values)
        return pd.DataFrame(values, columns=self.condition_columns, index=frame.index)

    def prediction_scenarios(self, conditions: Conditions) -> list[str]:
        """Разметить известные условия, интерполяцию внутри диапазонов и экстраполяцию.

        Метка 'interpolation' относится к прямоугольному диапазону факторов;
        она не гарантирует принадлежности выпуклой оболочке обучающих условий.
        """
        self._require_fit()
        frame = self._conditions(conditions)
        labels = []
        for row in frame.to_numpy():
            outside = any(
                value < self.condition_ranges[name][0] or value > self.condition_ranges[name][1]
                for name, value in zip(self.condition_columns, row, strict=True)
            )
            seen = np.any(
                np.all(np.isclose(self.training_conditions, row, rtol=1e-10, atol=1e-12), axis=1)
            )
            labels.append(
                "extrapolation" if outside else ("seen_conditions" if seen else "interpolation")
            )
        return labels

    def predict_parameters(
        self,
        conditions: Conditions,
        *,
        warn_extrapolation: bool = True,
    ) -> pd.DataFrame:
        """Предсказать временные параметры для каждой строки условий в порядке семейства."""
        self._require_fit()
        frame = self._conditions(conditions)
        if warn_extrapolation and "extrapolation" in self.prediction_scenarios(frame):
            warnings.warn(
                f"Conditions are outside training ranges: {self.condition_ranges}",
                ExtrapolationWarning,
                stacklevel=2,
            )
        X = frame.to_numpy()
        values = {}
        for name in self.time_family.parameter_names:
            prediction = self.parameter_models[name].predict(X)
            if prediction.shape != (len(frame),):
                raise ValueError(f"Parameter model for {name} returned an invalid shape.")
            if name in self.time_family.positive_parameters and np.any(prediction <= 0):
                model_name = self.parameter_models[name].model.name
                raise ValueError(
                    f"{model_name} predicts a nonpositive value for strictly positive "
                    f"time parameter {name}; these conditions are incompatible with the family."
                )
            values[name] = prediction
        parameters = pd.DataFrame(values, index=frame.index)
        for row in parameters.to_numpy():
            self.time_family.validate_parameters(row)
        return parameters

    def predict(
        self,
        time: ArrayLike,
        conditions: Conditions,
        *,
        initial_value: float | ArrayLike | None = None,
        warn_extrapolation: bool = True,
    ) -> FloatArray:
        """Предсказать новую траекторию или построчные сочетания времени и условий.

        Словарь или одна строка DataFrame распространяются на все моменты времени.
        Иначе на каждый момент должна приходиться одна строка условий.
        Семействам с начальным состоянием нужен конечный скаляр y0 или y0 для каждой строки;
        время задаётся как прошедшее с начала наблюдений.
        """
        self._require_fit()
        time = as_vector(time, "time")
        params = self.predict_parameters(
            conditions, warn_extrapolation=warn_extrapolation
        ).to_numpy()
        if len(params) not in (1, len(time)):
            raise ValueError("Provide one condition row or one row per time value.")
        initial = None
        if self.time_family.requires_initial_value:
            if initial_value is None:
                raise ValueError("This family requires initial_value for a new trajectory.")
            initial = as_vector(np.atleast_1d(initial_value), "initial_value")
            if len(initial) not in (1, len(time)):
                raise ValueError("initial_value must be scalar or have the same length as time.")
        elif initial_value is not None:
            raise ValueError("This time family does not use initial_value.")
        if len(params) == 1 and (initial is None or len(initial) == 1):
            family = (
                self.time_family
                if initial is None
                else self.time_family.with_initial_value(initial[0])
            )
            return family.evaluate(time, params[0])
        prediction = np.empty_like(time)
        for i in range(len(time)):
            family = self.time_family
            if initial is not None:
                family = family.with_initial_value(initial[0 if len(initial) == 1 else i])
            prediction[i] = family.evaluate(time[i : i + 1], params[0 if len(params) == 1 else i])[
                0
            ]
        return prediction

    def _predict_trajectory(self, item: Trajectory) -> FloatArray:
        return self.predict(
            item.time,
            dict(zip(self.condition_columns, item.conditions, strict=True)),
            initial_value=item.initial_value,
            warn_extrapolation=False,
        )

    def score_parameters(
        self,
        conditions: Conditions,
        true_parameters: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """Сравнить прогнозы с внешними известными физическими параметрами без подбора.

        Строки сопоставляются по позиции. Эта дополнительная диагностика не выбирает модель.
        """
        prediction = self.predict_parameters(conditions)
        if len(prediction) != len(true_parameters):
            raise ValueError("conditions and true_parameters need the same number of rows.")
        if not set(self.time_family.parameter_names).issubset(true_parameters.columns):
            raise ValueError("true_parameters must name every time-family parameter.")
        return {
            name: regression_metrics(true_parameters[name], prediction[name])
            for name in self.time_family.parameter_names
        }
