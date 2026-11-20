"""Двухэтапный подбор с ограниченным перебором сочетаний по отложенным кривым."""

from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from itertools import product
from math import prod

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .._validation import FloatArray
from ..families import BaseFunctionFamily
from ..fitting import _positive_integer
from ..metrics import regression_metrics
from ..parameter_models import (
    BaseParameterModel,
    ConstantParameterModel,
    LinearParameterModel,
    PolynomialParameterModel,
)
from ..parameter_models.base import independent_weights
from .fitting import fit_trajectory
from .results import Conditions, MultiExperimentResult, ParameterRelationship
from .validation import evaluate_curves, prepare_trajectories


class MultiExperimentRegressor:
    """По обучающим траекториям оценить параметры и их зависимости от условий.

    Parameters
    ----------
    time_family : BaseFunctionFamily
        Явно заданное временное семейство. Автоматического выбора его формы нет.
    parameter_models : отображение или None
        Соответствие имён временных параметров моделям коэффициентов.
        Включает ручной режим, если не задано parameter_selection='auto'.
        В автоматическом режиме эти записи закрепляют конкретные модели.
    parameter_selection : {'auto', 'manual'}
        В режиме auto сочетания сравниваются по RMSE кривых валидации.
        В режиме manual для каждого нефиксированного параметра требуется модель.
    parameter_candidates : последовательность или None
        Кандидаты автоматического выбора: по умолчанию константа, линейная
        и полная квадратичная модели.
    parameter_transforms : отображение или None
        Явные преобразования 'identity'/'log'. Для строго положительных параметров
        семейства по умолчанию используется log, для остальных — identity.
        Обязательная шкала специализированной модели учитывается отдельно;
        явный конфликт настроек отклоняется.
    fixed_parameters : отображение или None
        Известные физические коэффициенты, фиксируемые на обоих этапах.
        Позволяют устойчиво оценивать остальные параметры при неполной идентифицируемости.
    validation_size, test_size : float
        Доли независимых групп. При test_size=0 отдельная тестовая часть не создаётся.
    random_state : int
        Начальное значение генератора для воспроизводимого группового разбиения.
    aggregation : {'experiment', 'observation'}
        По умолчанию независимые эксперименты имеют одинаковый вес при выборе модели.
    relative_tolerance, absolute_tolerance : float
        Предпочтение простоты в пределах min_RMSE*(1+relative_tolerance)+absolute_tolerance.
    condition_threshold : float
        Порог обусловленности якобиана после нормировки столбцов.
        Оценки выше порога или с недостаточным рангом исключаются.
    max_combinations : int
        Жёсткое ограничение числа полных сочетаний; перебор не усекается молча.
    max_nfev, max_starts : int
        Бюджет вычислений на один старт и максимальное число начальных приближений
        для подбора отдельных временных кривых.
    """

    def __init__(
        self,
        time_family: BaseFunctionFamily,
        *,
        parameter_models: Mapping[str, BaseParameterModel] | None = None,
        parameter_selection: str | None = None,
        parameter_candidates: Sequence[BaseParameterModel] | None = None,
        parameter_transforms: Mapping[str, str] | None = None,
        fixed_parameters: Mapping[str, float] | None = None,
        validation_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42,
        aggregation: str = "experiment",
        relative_tolerance: float = 0.01,
        absolute_tolerance: float = 1e-8,
        condition_threshold: float = 1e8,
        max_combinations: int = 81,
        max_nfev: int = 2000,
        max_starts: int = 12,
    ) -> None:
        if not isinstance(time_family, BaseFunctionFamily):
            raise TypeError("time_family must implement BaseFunctionFamily.")
        self.time_family = deepcopy(time_family)
        names = set(time_family.parameter_names)
        if len(names) != time_family.n_parameters or not names:
            raise ValueError("Time-family parameter names must be nonempty and unique.")
        self.parameter_models = dict(parameter_models or {})
        self.fixed_parameters = dict(fixed_parameters or {})
        self.transforms = {
            name: "log" if name in time_family.positive_parameters else "identity"
            for name in time_family.parameter_names
        }
        self.transforms.update(parameter_transforms or {})
        self._explicit_transforms = set(parameter_transforms or {})
        for mapping in (self.parameter_models, self.fixed_parameters, self.transforms):
            if set(mapping) - names:
                raise ValueError(f"Unknown time parameters: {set(mapping) - names}")
        if set(self.parameter_models) & set(self.fixed_parameters):
            raise ValueError("A fixed parameter must not also have a coefficient model.")
        if any(transform not in ("identity", "log") for transform in self.transforms.values()):
            raise ValueError("Parameter transforms must be 'identity' or 'log'.")
        lower, upper = time_family.parameter_bounds()
        for name, value in self.fixed_parameters.items():
            index = time_family.parameter_names.index(name)
            if (
                not np.isscalar(value)
                or not np.isfinite(value)
                or value < lower[index]
                or value > upper[index]
                or (self.transforms[name] == "log" and value <= 0)
            ):
                raise ValueError(f"Invalid fixed physical value for {name}.")
        mode = parameter_selection or ("manual" if parameter_models is not None else "auto")
        if mode not in ("manual", "auto"):
            raise ValueError("parameter_selection must be 'manual' or 'auto'.")
        if mode == "manual" and set(self.parameter_models) | set(self.fixed_parameters) != names:
            raise ValueError("Manual mode needs a model or fixed value for every time parameter.")
        self.parameter_selection = mode
        self.parameter_candidates = (
            tuple(parameter_candidates)
            if parameter_candidates is not None
            else (
                ConstantParameterModel(),
                LinearParameterModel(),
                PolynomialParameterModel(),
            )
        )
        if not self.parameter_candidates or any(
            not isinstance(model, BaseParameterModel)
            for model in (*self.parameter_candidates, *self.parameter_models.values())
        ):
            raise ValueError("Coefficient models must implement BaseParameterModel.")
        if (
            not np.isfinite(validation_size)
            or not np.isfinite(test_size)
            or not 0 < validation_size < 1
            or not 0 <= test_size < 1
            or validation_size + test_size >= 1
        ):
            raise ValueError("Split fractions require validation>0, test>=0 and validation+test<1.")
        if (
            isinstance(random_state, bool)
            or not isinstance(random_state, int)
            or not 0 <= random_state < 2**32
        ):
            raise ValueError("random_state must be an integer in [0, 2**32).")
        if aggregation not in ("experiment", "observation"):
            raise ValueError("aggregation must be 'experiment' or 'observation'.")
        if any(not np.isfinite(v) or v < 0 for v in (relative_tolerance, absolute_tolerance)):
            raise ValueError("Selection tolerances must be finite and nonnegative.")
        if not np.isfinite(condition_threshold) or condition_threshold < 1:
            raise ValueError("condition_threshold must be finite and >= 1.")
        for name, value in (
            ("max_combinations", max_combinations),
            ("max_nfev", max_nfev),
            ("max_starts", max_starts),
        ):
            _positive_integer(value, name)
        self.validation_size, self.test_size, self.random_state = (
            validation_size,
            test_size,
            random_state,
        )
        self.aggregation = aggregation
        self.relative_tolerance, self.absolute_tolerance = relative_tolerance, absolute_tolerance
        self.condition_threshold = condition_threshold
        self.max_combinations, self.max_nfev, self.max_starts = (
            max_combinations,
            max_nfev,
            max_starts,
        )
        self.result_: MultiExperimentResult | None = None

    def fit(
        self,
        df: pd.DataFrame,
        *,
        time_column: str,
        target_column: str,
        group_column: str,
        condition_columns: Sequence[str],
        trajectory_column: str | None = None,
        split_column: str | None = None,
        initial_values: Mapping[Hashable, float] | None = None,
    ) -> MultiExperimentResult:
        """Обучить и оценить общую модель без подбора параметров отложенных кривых.

        group_column задаёт независимые эксперименты; необязательный trajectory_column —
        организмы или траектории внутри них. split_column может содержать заранее заданные
        метки train/validation/test, постоянные в пределах независимого эксперимента.
        initial_values при необходимости сопоставляет ID траекторий известным начальным состояниям.
        Ошибки схемы вызывают ValueError; неудачные подгонки сохраняются в результате.
        """
        self.result_ = None
        if isinstance(condition_columns, str):
            raise ValueError("condition_columns must be a sequence of column names, not a string.")
        columns = tuple(condition_columns)
        # Fail early for custom families without a composable expression contract.
        for latex in (False, True):
            self.time_family.expression(
                {name: name for name in self.time_family.parameter_names}, latex=latex
            )
        trajectories, protocol = prepare_trajectories(
            df,
            self.time_family,
            time_column=time_column,
            target_column=target_column,
            group_column=group_column,
            condition_columns=columns,
            trajectory_column=trajectory_column,
            split_column=split_column,
            initial_values=initial_values,
            validation_size=self.validation_size,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        result = MultiExperimentResult(
            deepcopy(self.time_family),
            columns,
            protocol,
            self.aggregation,
            self.relative_tolerance,
            self.absolute_tolerance,
            _trajectories=tuple(trajectories),
        )
        self.result_ = result
        records, usable = [], []
        n_free = self.time_family.n_parameters - len(self.fixed_parameters)
        for item in trajectories:
            record = {
                "trajectory_id": item.identifier,
                "experiment_id": item.experiment,
                "partition": item.partition,
                "n_observations": len(item.time),
                "time_origin": item.time_origin,
                "initial_value": item.initial_value,
                "fit_success": False,
                "used_for_parameter_models": False,
                "message": "Held-out trajectory: not fitted.",
                **{
                    f"condition:{name}": value
                    for name, value in zip(columns, item.conditions, strict=True)
                },
                **{f"param:{name}": np.nan for name in self.time_family.parameter_names},
            }
            if item.partition == "train":
                fitted = fit_trajectory(
                    self.time_family,
                    item,
                    self.fixed_parameters,
                    max_nfev=self.max_nfev,
                    max_starts=self.max_starts,
                )
                result.train_fits[item.identifier] = fitted
                record.update(
                    {
                        "fit_success": fitted.success,
                        "message": fitted.message,
                        "jacobian_rank": fitted.jacobian_rank,
                        "jacobian_condition": fitted.jacobian_condition,
                        "fit_warnings": "; ".join(fitted.warnings),
                        **{f"fit_{key}": value for key, value in fitted.train_metrics.items()},
                        **{f"param:{name}": value for name, value in fitted.coefficients.items()},
                    }
                )
                eligible = fitted.success
                if eligible and (
                    fitted.jacobian_rank is None
                    or fitted.jacobian_rank < n_free
                    or fitted.jacobian_condition is None
                    or fitted.jacobian_condition > self.condition_threshold
                ):
                    eligible = False
                    record["message"] = (
                        "Excluded: time parameters are locally unidentifiable or ill-conditioned."
                    )
                if eligible and any(
                    fitted.coefficients[name] <= 0
                    for name, transform in self.transforms.items()
                    if transform == "log"
                ):
                    eligible = False
                    record["message"] = (
                        "Excluded: log-transformed parameter estimate is not strictly positive."
                    )
                record["used_for_parameter_models"] = eligible
                if eligible and n_free > 0:
                    family_warnings = self.time_family.identifiability_warnings(
                        item.time, fitted.params
                    )
                    if family_warnings:
                        eligible = False
                        record["used_for_parameter_models"] = False
                        record["message"] = "Excluded: " + "; ".join(family_warnings)
                if eligible:
                    usable.append(item)
            elif item.error:
                record["message"] = item.error
            records.append(record)
        result.experiment_parameters = pd.DataFrame(records)
        train = [item for item in trajectories if item.partition == "train"]
        result.evaluations["train_curve_fit"] = evaluate_curves(
            train, lambda item: result.train_fits[item.identifier].predict(item.time)
        )
        excluded = len(train) - len(usable)
        if excluded:
            result.warnings += (
                f"Excluded {excluded}/{len(train)} train trajectories from coefficient regression; "
                "see experiment_parameters.",
            )
        if not usable:
            result.message = (
                "No identifiable training trajectories are available for coefficient models."
            )
            return result
        X = np.array([item.conditions for item in usable])
        groups = [item.experiment for item in usable]
        n_independent = len(set(groups))
        weights = independent_weights(groups)
        result.training_conditions = np.unique(X, axis=0)
        result.condition_ranges = {
            name: (float(X[:, i].min()), float(X[:, i].max())) for i, name in enumerate(columns)
        }
        choices: dict[str, list[ParameterRelationship]] = {}
        model_records = []
        for name in self.time_family.parameter_names:
            targets = np.array(
                [result.train_fits[item.identifier].coefficients[name] for item in usable]
            )
            if name in self.fixed_parameters:
                value = self.fixed_parameters[name]
                if self.transforms[name] == "log":
                    value = float(np.log(value))
                templates = (ConstantParameterModel(value),)
            elif name in self.parameter_models:
                templates = (self.parameter_models[name],)
            else:
                templates = self.parameter_candidates
            choices[name] = []
            for template in templates:
                candidate = deepcopy(template)
                transform = candidate.required_transform or self.transforms[name]
                info = {
                    "parameter": name,
                    "model": candidate.name,
                    "n_coefficients": None,
                    "transform": transform,
                    "success": False,
                    "message": "",
                }
                try:
                    if transform not in ("identity", "log"):
                        raise ValueError("Model required_transform must be identity, log or None.")
                    if name in self._explicit_transforms and transform != self.transforms[name]:
                        raise ValueError(
                            f"{candidate.name} requires {transform} target scale, "
                            f"but {name} explicitly uses {self.transforms[name]}."
                        )
                    if transform == "log" and np.any(targets <= 0):
                        raise ValueError(
                            "Log-target models require strictly positive time parameters."
                        )
                    transformed = np.log(targets) if transform == "log" else targets
                    complexity = candidate.complexity(len(columns))
                    if not isinstance(complexity, int) or complexity < 0:
                        raise ValueError("Model complexity must be a nonnegative integer.")
                    info["n_coefficients"] = complexity
                    if n_independent < max(1, complexity):
                        raise ValueError(
                            f"Need {complexity} independent training experiments; "
                            f"have {n_independent}."
                        )
                    candidate.fit(X, transformed, feature_names=columns, sample_weight=weights)
                    relationship = ParameterRelationship(candidate, transform)
                    prediction = relationship.predict(X)
                    regression_metrics(targets, prediction)
                    relationship.formula()
                    relationship.formula(latex=True)
                    choices[name].append(relationship)
                    info["success"] = True
                    info["message"] = "; ".join(getattr(candidate, "warnings_", ()))
                except Exception as error:
                    info["message"] = f"{type(error).__name__}: {error}"
                model_records.append(info)
        result.parameter_candidates = pd.DataFrame(model_records)
        if any(not available for available in choices.values()):
            result.message = "No usable coefficient model for: " + ", ".join(
                name for name, values in choices.items() if not values
            )
            return result
        combination_count = prod(len(available) for available in choices.values())
        if combination_count > self.max_combinations:
            result.message = (
                f"Search requires {combination_count} combinations; "
                f"max_combinations={self.max_combinations}. "
                "Reduce candidates, pin models, or explicitly increase the limit."
            )
            return result
        validation = [item for item in trajectories if item.partition == "validation"]
        combinations, evaluations, scores = [], [], []
        # Temporary success enables the same prediction code used by final results.
        result.success = True
        for combination in product(*choices.values()):
            relationships = dict(zip(choices, combination, strict=True))
            result.parameter_models = relationships
            evaluation = evaluate_curves(validation, result._predict_trajectory)
            aggregate = (
                evaluation.experiment
                if self.aggregation == "experiment"
                else evaluation.observation
            )
            score = aggregate.get("rmse", np.inf) if evaluation.complete else np.inf
            complexity = sum(r.model.complexity(len(columns)) for r in combination)
            scores.append(
                {
                    "candidate": len(scores),
                    "models": {name: r.model.name for name, r in relationships.items()},
                    "n_coefficients": complexity,
                    "validation_score": score,
                    "validation_observation_rmse": evaluation.observation.get("rmse", np.nan),
                    "validation_experiment_rmse": evaluation.experiment.get("rmse", np.nan),
                    "success": evaluation.complete and np.isfinite(score),
                    "selected": False,
                    "message": ""
                    if evaluation.complete
                    else evaluation.failures.to_string(index=False),
                }
            )
            combinations.append(relationships)
            evaluations.append(evaluation)
        result.candidate_scores = pd.DataFrame(scores)
        successful = [i for i, record in enumerate(scores) if record["success"]]
        if not successful:
            result.success = False
            result.parameter_models = {}
            result.message = (
                "No combination could predict all validation trajectories. See candidate_scores."
            )
            return result
        minimum = min(scores[i]["validation_score"] for i in successful)
        threshold = minimum * (1 + self.relative_tolerance) + self.absolute_tolerance
        eligible = [i for i in successful if scores[i]["validation_score"] <= threshold]
        best = min(
            eligible, key=lambda i: (scores[i]["n_coefficients"], scores[i]["validation_score"], i)
        )
        result.parameter_models = combinations[best]
        result.candidate_scores.loc[best, "selected"] = True
        result.evaluations["validation"] = evaluations[best]
        result.evaluations["train_general"] = evaluate_curves(train, result._predict_trajectory)
        # This is the first time test predictions/targets participate in an evaluation.
        test = [item for item in trajectories if item.partition == "test"]
        if test:
            result.evaluations["test"] = evaluate_curves(test, result._predict_trajectory)
            if not result.evaluations["test"].complete:
                result.warnings += (
                    "Test evaluation is incomplete; inspect evaluations['test'].failures.",
                )
        result.parameter_metrics = {
            name: regression_metrics(
                [result.train_fits[item.identifier].coefficients[name] for item in usable],
                relationship.predict(X),
            )
            for name, relationship in result.parameter_models.items()
        }
        result.message = (
            f"Selected {self.parameter_selection} coefficient models using "
            f"{self.aggregation}-aggregated validation curve RMSE."
        )
        return result

    def predict(
        self,
        time: ArrayLike,
        conditions: Conditions,
        *,
        initial_value: float | ArrayLike | None = None,
        warn_extrapolation: bool = True,
    ) -> FloatArray:
        """Использовать последний результат; до успешного обучения вызвать RuntimeError."""
        if self.result_ is None:
            raise RuntimeError("Call fit before predict.")
        return self.result_.predict(
            time, conditions, initial_value=initial_value, warn_extrapolation=warn_extrapolation
        )
