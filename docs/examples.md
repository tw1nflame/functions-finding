# Примеры и ноутбуки

## Одна кривая

Скрипт создаёт синтетический ряд, сравнивает кандидатов и сохраняет PNG и CSV.
Запускайте из корня проекта:

```console
uv run python examples/basic_usage.py
```

```{literalinclude} ../examples/basic_usage.py
:language: python
:start-at:     rng =
:end-at:     result.to_dataframe().to_csv
:dedent: 4
```

Скачать: {download}`basic_usage.py <../examples/basic_usage.py>` или
{download}`synthetic_experiment.ipynb <../examples/synthetic_experiment.ipynb>`.

## Несколько экспериментов и два условия

30 независимых синтетических экспериментов с известной зависимостью амплитуды и
скорости релаксации от температуры и pH. Скрипт сохраняет четыре диагностических
графика, формулу, метрики, промежуточные коэффициенты и split-ID.

```console
uv run python examples/multi_experiment_usage.py
```

```{literalinclude} ../examples/multi_experiment_usage.py
:language: python
:start-at:     df = make_synthetic_data()
:end-at:     print("Test:",
:dedent: 4
```

Скачать: {download}`multi_experiment_usage.py <../examples/multi_experiment_usage.py>` или
{download}`multi_experiment.ipynb <../examples/multi_experiment.ipynb>`.

## Реальные измерения ST14028

[Полный отчёт](real_data_experiment.md) описывает аудит исходного Excel, отдельные
кривые, предсказание исключённой концентрации 0,2 мМ, baseline и ограничения.
Исходные данные доступны в `data/external`, long-таблицы — в `data/processed`.

```console
uv run python examples/run_real_bacterial_growth.py
```

Скачать выполненный {download}`real_bacterial_growth.ipynb <../examples/real_bacterial_growth.ipynb>`.
Для запуска рядом нужен модуль
{download}`bacterial_growth_support.py <../examples/bacterial_growth_support.py>`;
проще открыть ноутбук непосредственно в checkout проекта.

```{note}
Загрузка ноутбука из документации сама по себе не устанавливает библиотеку и данные.
Выполнение предполагает структуру исходного проекта. Инструкции находятся в первом
разделе каждого ноутбука и в отчёте реального эксперимента.
```
