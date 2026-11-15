Результат общей модели
======================

Прежде чем запрашивать прогнозы и формулы, проверяйте ``success`` и ``message``.
Таблицы диагностики сохраняются и при неуспешном обучении.

.. autoclass:: parametric_regression.multi_experiment.results.MultiExperimentResult
   :members:

Связь одного параметра с условиями
----------------------------------

.. autoclass:: parametric_regression.multi_experiment.results.ParameterRelationship
   :members:

Оценка кривых
-------------

Объекты ``CurveEvaluation`` находятся в ``result.evaluations``. Они содержат
предсказания, ошибки, покрытие и отдельные неудачные траектории.

.. autoclass:: parametric_regression.multi_experiment.validation.CurveEvaluation
   :members:

Протокол разделения
-------------------

.. autoclass:: parametric_regression.multi_experiment.validation.ExperimentSplit
   :members:
