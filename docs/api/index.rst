Справочник API
==============

Сигнатуры и описания ниже получаются из исходного кода при каждой сборке.
Обучающие примеры находятся в :doc:`../quickstart`, математическая логика —
в :doc:`../fitting_explained`.

.. toctree::
   :maxdepth: 2

   single
   families
   multi
   parameters
   results
   visualization

Выбор точки входа
-----------------

* :class:`~parametric_regression.ModelSelector` — сравнение временных семейств.
* :func:`~parametric_regression.fit_family` — подбор заданного семейства.
* :class:`~parametric_regression.MultiExperimentRegressor` — общая модель нескольких экспериментов.
* :class:`~parametric_regression.BaseFunctionFamily` — расширение набора семейств.
* :class:`~parametric_regression.BaseParameterModel` — собственные зависимости коэффициентов.
