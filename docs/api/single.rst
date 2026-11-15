Одна кривая
===========

Подбор семейства
----------------

Общие ошибки входных данных вызывают ``ValueError``. Ошибка конкретного семейства
сохраняется как ``FitResult(success=False)``. Вызов ``predict`` неуспешного
результата вызывает ``RuntimeError``.

.. autofunction:: parametric_regression.fit_family

.. autofunction:: parametric_regression.regression_metrics

Выбор модели
------------

.. autoclass:: parametric_regression.ModelSelector
   :members:

Результаты
----------

.. autoclass:: parametric_regression.FitResult
   :members:

.. autoclass:: parametric_regression.SelectionResult
   :members:

.. autoclass:: parametric_regression.SplitInfo
   :members:
