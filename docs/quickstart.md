# Быстрый старт

## Установка

Нужен Python 3.11 или новее. Из корня скачанного проекта:

```console
python -m pip install -e .
```

Для ноутбуков и разработки:

```console
uv sync --extra notebook --extra dev
```

`numpy`, `scipy`, `scikit-learn`, `pandas` и `matplotlib` устанавливаются как зависимости.

## Подобрать одну кривую

```python
import numpy as np
from parametric_regression import ModelSelector

rng = np.random.default_rng(42)
time = np.linspace(0, 20, 100)
response = 2 + 5 * np.exp(-0.3 * time) + rng.normal(0, 0.03, time.size)

result = ModelSelector(random_state=42).fit(time, response, split="time")
assert result.success, result.to_dataframe().to_string(index=False)

print(result.best_family)
print(result.formula)
print(result.coefficients)
print(result.metrics)
print(result.predict([21, 22]))
```

`split="time"` оставляет последние временные отметки для validation. Все коэффициенты
победителя получены только на train: скрытого переобучения после выбора нет.
Другие варианты — `random` и `group`; при `group` передайте `groups`.

Сравнение **всех** кандидатов, включая неудачные попытки:

```python
print(result.to_dataframe())
```

График наблюдений и нескольких кандидатов:

```python
from parametric_regression.visualization import plot_fit

figure = plot_fit(result, time, response, top_n=3)
figure.savefig("fit.png", dpi=150, bbox_inches="tight")
```

## Подобрать конкретное семейство

```python
from parametric_regression import ExponentialFamily, fit_family

fit = fit_family(ExponentialFamily(), time, response)
assert fit.success, fit.message
print(fit.coefficients)  # c, a, b в c + a*exp(b*x)
print(fit.jacobian_rank, fit.jacobian_condition)
```

`fit_family` использует все переданные ему точки. Для validation и выбора семейства
используйте `ModelSelector` или задайте собственное разбиение до вызова.

## Общая модель с двумя условиями

Синтетический генератор из репозитория создаёт 30 независимых экспериментов с
постоянными температурой и pH внутри каждого. Подготовьте такую же long-таблицу
для своих измерений: одна строка — одно наблюдение во времени.

```python
from examples.multi_experiment_usage import make_synthetic_data
from parametric_regression import ExponentialRelaxationFamily, MultiExperimentRegressor

data = make_synthetic_data(random_state=42)
result = MultiExperimentRegressor(
    ExponentialRelaxationFamily(),
    random_state=42,
).fit(
    data,
    time_column="time",
    target_column="response",
    group_column="experiment_id",
    condition_columns=["temperature", "pH"],
)
assert result.success, result.message

print(result.parameter_formulas)
print(result.formula)
print(result.metrics["test"])

prediction = result.predict(
    [0, 5, 10, 20],
    {"temperature": 25.0, "pH": 7.0},
    initial_value=1.1,
)
```

Этот пример запускается из корня проекта. Начальное значение `initial_value` нужно
семейству `ExponentialRelaxationFamily`; оно не является подбираемым коэффициентом.
В прогноз передаётся прошедшее время от начала траектории.

Групповое разбиение выполняется до подбора параметров. Если в одном независимом
эксперименте есть несколько кривых, используйте отдельный `trajectory_column` с
глобально уникальным ID каждой кривой.

Дальше: [формат таблицы и полный протокол](multi_experiment_regression.md),
[модели коэффициентов](parameter_models.md), [справочник API](api/index.rst).
