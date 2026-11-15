# Parametric Regression

```{raw} html
<div class="hero-kicker">Python library · v0.2.0 · NumPy / SciPy / scikit-learn</div>
<p class="hero-lead">От экспериментальных точек к аналитической формуле.
Подбирайте временную зависимость, сравнивайте модели и описывайте их коэффициенты
через условия эксперимента.</p>
<div class="doc-cards">
  <a class="doc-card" href="quickstart.html"><strong>Начать за 5 минут →</strong>
    <span>Установка, первая модель, коэффициенты и прогноз.</span></a>
  <a class="doc-card" href="fitting_explained.html"><strong>Как работает подбор →</strong>
    <span>Наименьшие квадраты, несколько стартов и выбор сложности.</span></a>
  <a class="doc-card" href="api/index.html"><strong>Справочник API →</strong>
    <span>Сигнатуры, параметры, результаты и визуализации из исходного кода.</span></a>
  <a class="doc-card" href="real_data_experiment.html"><strong>Реальные данные →</strong>
    <span>Кривые роста ST14028, отложенная концентрация и сравнение с baseline.</span></a>
</div>
```

## Два уровня моделирования

**Одна кривая.** Задайте наблюдения `x, y`. `ModelSelector` подберёт коэффициенты
каждого кандидата на обучающей части и выберет семейство по проверочной RMSE.
Результат включает численную формулу, прогноз, метрики и диагностику всех попыток.

**Несколько экспериментов.** Задайте временное семейство и таблицу независимых
экспериментов. `MultiExperimentRegressor` оценит параметры тренировочных кривых,
построит их зависимости от внешних условий и вернёт общую формулу:

$$
\widehat y(t,\mathbf z)=F\bigl(t;G_1(\mathbf z),\ldots,G_p(\mathbf z)\bigr).
$$

Число условий не ограничено двумя: это могут быть температура, концентрация, pH
и другие численные факторы. В текущем API они должны быть постоянны внутри каждой
траектории. [Условия, меняющиеся во времени](scope.md), требуют другой постановки.

## Минимальный пример

```python
import numpy as np
from parametric_regression import ModelSelector

x = np.linspace(0, 10, 40)
y = 1 + 3 * np.exp(-0.4 * x)
result = ModelSelector(random_state=42).fit(x, y, split="time")

if result.success:
    print(result.formula)
    print(result.coefficients)
    print(result.predict([10.5, 11.0]))
else:
    print(result.to_dataframe())
```

```{note}
Версия 0.2 — исследовательская библиотека. Успешная оптимизация не гарантирует
идентифицируемости параметров. Validation участвует в выборе модели; для независимой
оценки нужна отдельная test-выборка. Автоматического refit на train+validation нет.
```

```{toctree}
:hidden:
:caption: Начало работы

quickstart
examples
scope
```

```{toctree}
:hidden:
:caption: Методы и руководство

fitting_explained
mathematical_models
multi_experiment_regression
parameter_models
validation
```

```{toctree}
:hidden:
:caption: Исследования

real_data_experiment
```

```{toctree}
:hidden:
:caption: Справочник и разработка

api/index
architecture
adding_new_family
building_docs
```
