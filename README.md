# Parametric Regression

Python-библиотека для автоматического подбора простой аналитической зависимости
`y = f(x)` по числовым наблюдениям. Это исследовательский MVP: параметры заданных
семейств оцениваются на обучающей выборке, а семейства сравниваются по RMSE на
общей отложенной выборке. Результат содержит формулу, коэффициенты, прогнозы и
диагностику каждого кандидата.

Начиная с версии **0.2**, библиотека также строит общую зависимость по нескольким
независимым экспериментам: `y(t,z) = f(t;g(z))`, где `z` — числовые условия, а
`g` описывает коэффициенты временной кривой. Старый одномерный API сохранён.
Подробности: [двухэтапная регрессия](docs/multi_experiment_regression.md).

## HTML-документация

Опубликованная документация:
[добавление нового семейства](https://tw1nflame.github.io/functions-finding/adding_new_family.html).

Руководство, математические методы, примеры, поиск и справочник API собираются
через Sphinx с темой Furo:

```console
uv sync --extra docs
uv run --extra docs python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Откройте `docs/_build/html/index.html`. Подробности и локальный сервер:
[сборка документации](docs/building_docs.md).

Для проверки исполняемых примеров:

```console
uv run --extra docs python -m sphinx -W --keep-going -b doctest docs docs/_build/doctest
```

Реальный эксперимент на опубликованных кривых бактериального роста ST14028:
[выполненный ноутбук](examples/real_bacterial_growth.ipynb) и
[отчёт с протоколом, метриками и ограничениями](docs/real_data_experiment.md).
В нём проверены оба этапа, предсказание исключённой концентрации 0,2 мМ и сравнение
с прямым полиномом и ближайшей концентрацией. Исходный Excel и подготовленные CSV
сохранены в `data/`. Повторный запуск: `uv run python examples/run_real_bacterial_growth.py`.

## Семейства

| Класс | Формула | Порядок коэффициентов |
|---|---|---|
| `PolynomialFamily(degree=1/2/3)` | `a0 + a1*x + ... + an*x**n` | `a0, ..., an` |
| `ExponentialFamily` | `c + a*exp(b*x)` | `c, a, b` |
| `LogisticFamily` | `c + L/(1 + exp(-k*(x-x0)))` | `c, L, k, x0` |
| `GompertzFamily` | `c + L*exp(-exp(-k*(x-x0)))` | `c, L, k, x0` |
| `RationalSaturationFamily` | `c + a*x/(K+x)` | `c, a, K` |
| `ExponentialRelaxationFamily(initial_value=y0)` | `y0 + A*(1-exp(-k*x))` | `A, k`; `y0` наблюдается |

Для рационального семейства требуется `x >= 0`, `K > 0`. В логистическом
семействе `L >= 0`, знак `k` задаёт направление перехода. Гомпертц допускает
оба знака `L` и `k`. Подробнее: [математические модели](docs/mathematical_models.md).

## Установка

Требуется Python 3.11+. Из корня проекта:

```bash
uv sync --extra dev
uv run python examples/basic_usage.py
```

Альтернатива с pip в виртуальном окружении:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Для Jupyter: `uv sync --extra dev --extra notebook`, затем
`uv run jupyter lab examples/synthetic_experiment.ipynb`.

## Быстрый пример

```python
import numpy as np
from parametric_regression import ModelSelector
from parametric_regression.visualization import plot_fit

rng = np.random.default_rng(42)
x = np.linspace(0, 20, 100)
y = 2 + 5 * np.exp(-0.3 * x) + rng.normal(0, 0.03, x.size)

result = ModelSelector(random_state=42).fit(x, y, split="time")
print(result.best_family)
print(result.formula)
print(result.coefficients)
print(result.metrics)  # MAE, RMSE, R² на проверочной выборке
print(result.to_dataframe())  # Все кандидаты, включая неудачные
print(result.predict([21, 22, 23]))

figure = plot_fit(result, x, y, top_n=3)
figure.savefig("comparison.png", dpi=150)
```

По умолчанию рассматриваются семь кандидатов: три степени полинома и четыре
нелинейных семейства. Можно передать свой список:

```python
from parametric_regression import ExponentialFamily, PolynomialFamily, fit_family

selector = ModelSelector(
    families=[PolynomialFamily(1), ExponentialFamily()],
    validation_size=0.2,
    relative_tolerance=0.01,
    absolute_tolerance=1e-12,
    random_state=42,
)
result = selector.fit(x, y, split="random")
single_fit = fit_family(ExponentialFamily(), x, y)
print(single_fit.train_metrics)
```

`fit_family` оценивает одно семейство на всех **переданных ему** наблюдениях.
`ModelSelector.fit` передаёт ему только обучающую часть. Выбранная модель
автоматически не переобучается на полной выборке: её формула и прогнозы
соответствуют тем же коэффициентам, для которых получены проверочные метрики.
Если после оценки нужна модель на всех данных, это отдельное явное действие:

```python
final_fit = fit_family(result.best_result.family, x, y)
# final_fit.metrics — обучающие метрики; не новая оценка обобщения.
```

## Выбор модели и разбиение

Все кандидаты используют одинаковые индексы, сохранённые в
`result.split_protocol.train_indices` и `.validation_indices`.

| `split` | Поведение |
|---|---|
| `"random"` | Случайное разбиение наблюдений с фиксированным `random_state`. |
| `"time"` | Сортировка по `x`; последние 20% уникальных значений `x` по умолчанию идут в проверку. Одинаковые отметки времени не разделяются. |
| `"group"` | Разбиение целых экспериментов, заданных `groups`; 20% групп по умолчанию идут в проверку. |

`validation_size` задаёт долю строк, уникальных времён или групп соответственно;
число проверочных элементов округляется вверх. В каждой части нужны минимум
две строки. Для каждого кандидата в обучении требуется не менее числа его
параметров различных значений `x`. Неподходящий кандидат сохраняется как неудачный.

Пусть минимальный проверочный RMSE равен `m`. К рассмотрению допускаются модели с
`RMSE <= m * (1 + relative_tolerance) + absolute_tolerance`. Среди них выбирается
модель с меньшим числом параметров, затем меньшим RMSE, затем более ранняя во
входном списке. По умолчанию относительный допуск — 1%; абсолютный `1e-12`
устраняет влияние округления на практически точных зависимостях и имеет единицы
целевой переменной. Он настраивается, особенно для очень малых значений `y`.

В результате явно сохранены `criterion="validation_rmse"`, оба допуска и протокол
разбиения. Критерий выбирает семейство на одной проверочной выборке; для независимой
итоговой оценки исследования нужна дополнительная тестовая выборка.

## DataFrame и независимые эксперименты

```python
import pandas as pd

df = pd.DataFrame({"time": x, "aspect_ratio": y})
result = selector.fit_dataframe(df, x_column="time", y_column="aspect_ratio", split="time")

# Когда в df есть идентификаторы нескольких независимых экспериментов:
# result = selector.fit_dataframe(
#     df, x_column="time", y_column="aspect_ratio",
#     group_column="experiment_id", split="group",
# )
```

Имена столбцов произвольные. Индексы разбиения — позиции строк, а не `df.index`.
Передача `groups` при другом протоколе отклоняется, чтобы случайно не разделить
наблюдения одного эксперимента. NaN, бесконечности, комплексные значения,
нечисловые столбцы и массивы размерности выше одной отклоняются; строки не удаляются
и значения не заполняются автоматически.

## Результаты и ошибки

- `FitResult`: `params`, `coefficients`, `formula`, `latex_formula`, `success`,
  `message`, `warnings`, `n_parameters`, `nfev`, `train_metrics`, `validation_metrics`.
- `SelectionResult`: `best_result`, `best_family`, `candidates`, `split_protocol`,
  `criterion`, `to_dataframe()`, формулы, коэффициенты, метрики и `predict(x)` победителя.
- Ошибки общих входных данных дают `ValueError`. Ошибки домена, оптимизации и
  прогнозов отдельного кандидата сохраняются в его результате.
- Если все кандидаты неуспешны, `result.success == False`, `best_family` и формула
  равны `None`; попытка `predict` или доступа к `best_result` даёт `RuntimeError`
  с диагностикой. Все попытки остаются в `candidates`.
- R² для постоянной цели: 1 при точном совпадении, иначе 0 (конвенция scikit-learn).
  Для единственного наблюдения `regression_metrics` возвращает R² = NaN; селектор
  требует минимум две проверочные строки.
- Нелинейный подбор использует детерминированные стартовые приближения.
  `max_starts=12` и `max_nfev=2000` можно менять. Отказы отдельных стартов и
  недостаточный ранг якобиана сохраняются в `warnings`.

Функция `plot_fit` возвращает `matplotlib.figure.Figure`, не вызывает `show()` и
не сохраняет файлы сама. При отображении разбиения ей нужны исходные строки в
исходном порядке. `top_n` позволяет сравнить несколько успешных моделей.

## Общая модель по условиям эксперимента

Для таблицы `df` со столбцами времени, отклика, ID независимого эксперимента и
постоянными внутри траектории факторами:

```python
from parametric_regression import MultiExperimentRegressor, ExponentialRelaxationFamily

model = MultiExperimentRegressor(
    time_family=ExponentialRelaxationFamily(),
    parameter_selection="auto",
    random_state=42,
)
result = model.fit(
    df,
    time_column="time",
    target_column="response",
    group_column="experiment_id",
    condition_columns=["temperature", "pH"],
)
print(result.formula)
print(result.parameter_formulas)
print(result.metrics["test"]["experiment"])
predictions = result.predict(
    time=[0, 5, 10, 20],
    conditions={"temperature": 25, "pH": 7},
    initial_value=1.1,
)
```

Второй этап использует `ConstantParameterModel`, `LinearParameterModel` и
`PolynomialParameterModel(degree=2, interactions=True/False)`. Режим Manual
принимает словарь моделей по именам коэффициентов. Auto сравнивает комбинации
по ошибке **итоговых кривых** на validation, с предпочтением простых зависимостей.
Положительные `k` у релаксации и `K` у рационального насыщения по умолчанию
моделируются через логарифм; остальные коэффициенты автоматически не логарифмируются.

Разбиение независимых экспериментов выполняется **до fitting**: по умолчанию
примерно 60% train, 20% validation, 20% test. Для нескольких организмов внутри
эксперимента передайте `trajectory_column`, сохранив общий `group_column`.
`split_column` позволяет использовать готовые метки train/validation/test.
Временные параметры validation/test не подбираются и не участвуют в обучении.

В релаксации `y0` определяется для каждой траектории отдельно, время начинается
с первого наблюдения. Начальные точки не входят в оценочные метрики. Отчёты
содержат MAE/RMSE/R² как по всем наблюдениям, так и в среднем по независимым
экспериментам; второй вариант используется для выбора по умолчанию.

`experiment_parameters` сохраняет все траектории и причины исключения;
`candidate_scores` — проверенные комбинации; `condition_ranges` — диапазоны
пригодных обучающих условий. Выход за диапазоны сопровождается `ExtrapolationWarning`.
Неидентифицируемые коэффициенты не используются бесконтрольно: проверяются ранг
и обусловленность якобиана. Известные нестабильные параметры можно закрепить
через `fixed_parameters`, а общую константу обучить через `ConstantParameterModel`.

Самодостаточная демонстрация с четырьмя графиками:

```bash
uv run python examples/multi_experiment_usage.py
uv run jupyter lab examples/multi_experiment.ipynb
```

Для Jupyter предварительно установите extra `notebook`, как описано выше.
Документация: [двухэтапный алгоритм](docs/multi_experiment_regression.md),
[модели коэффициентов](docs/parameter_models.md), [валидация](docs/validation.md).

## Расширение и разработка

Новый класс наследует `BaseFunctionFamily`, задаёт функцию, коэффициенты,
начальные приближения и формулу. Для включения достаточно передать экземпляр в
`ModelSelector(families=[...])`; глобального реестра нет. Полный пример:
[добавление семейства](docs/adding_new_family.md).

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python examples/basic_usage.py --output artifacts/demo.png
uv run python -m build
```

Тесты проверяют формулы, восстановление коэффициентов и кривых на синтетических
данных, утечку проверочных значений, разделение экспериментов, отказы оптимизации,
расширение новым семейством, крайние случаи и сохранение графиков.

Документация: [архитектура](docs/architecture.md),
[математика](docs/mathematical_models.md), [расширение](docs/adding_new_family.md).
Примеры: [скрипт](examples/basic_usage.py), [ноутбук](examples/synthetic_experiment.ipynb).

В демонстрации временная проверка приходится на почти постоянный хвост
экспоненты: RMSE около `0.025` при шуме `0.03`, но R² может быть отрицательным
из-за очень малой вариации цели на этом участке. Это не ошибка расчёта;
метрики нужно интерпретировать вместе с диапазоном и протоколом проверки.

## Ограничения и развитие

Одномерный API подбирает зависимость от одной переменной. Двухэтапный API
поддерживает одну временную координату и произвольное число числовых факторов,
постоянных внутри траектории. Временное семейство второго этапа задаётся явно.
Нелинейная оптимизация локальная: несколько стартов повышают устойчивость, но не
гарантируют глобальный минимум или однозначность коэффициентов. Слишком короткий
участок насыщения часто не позволяет восстановить асимптоты. Очень большие
смещения/масштабы `x` и `y` могут приводить к потере точности или диагностируемому
отказу; рекомендуется выбирать разумные единицы измерения. Выход за диапазон
данных допустим в домене семейства, но точность экстраполяции не гарантируется.
Истинное переполнение экспоненты вызывает ошибку, а не подмену формулы обрезанием.

В текущей версии нет доверительных интервалов, пользовательских весов отдельных
кадров, автоматического заполнения пропусков, общей символьной регрессии или
совместной оптимизации всей двухэтапной функции по исходным наблюдениям.
Повторные траектории учитываются равными суммарными весами независимых экспериментов
при регрессии коэффициентов. Проверки выполнены на синтетических данных и на
[опубликованном наборе ST14028](docs/real_data_experiment.md), с отдельно описанными
ограничениями метаданных, контроля и идентификации биологических запусков.
Возможные дальнейшие расширения: оценка неопределённости коэффициентов, совместное
уточнение общей функции, дополнительные протоколы проверки и ограничения
предметной области после появления реальных данных.
